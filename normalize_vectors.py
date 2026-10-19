#!/usr/bin/env python3
"""
Normalize steering vectors to have the same L2 norm across all behaviors.

Reads the raw vectors written by generate_vectors.py
(artifacts/caa_vectors_raw/) and rescales every behavior's vector, layer by
layer, to one target norm per layer. The result goes to artifacts/caa_vectors/,
which is what train_cast.py and the evaluation scripts read. Equalising the
norms is what makes one steering multiplier mean the same thing across
behaviors.

The target comes from configs/normalization_target_norms.yaml, which records the
per-model, per-layer norm the released vectors carry. Deriving it instead from
the mean over whatever behaviors happen to be on disk would tie the scale to the
behavior set -- a different set gives a different target, and every steering
multiplier downstream is read against that scale. Reading the recorded value
keeps the scale fixed whichever behaviors are being normalized.

--behaviors therefore chooses *which* behaviors to rescale, not what to rescale
them to. Pass the literal `all` to normalize every behavior that has raw
vectors, or name them explicitly. A behavior that is named but has no raw vector
for some layer is an error, not something to skip.

Example usage:
python normalize_vectors.py --model_family llama --model_size 8b --n_layers 32 \
    --behaviors all
python normalize_vectors.py --model_family qwen --model_size 7b --n_layers 28 \
    --behaviors myopic-reward sycophancy
"""

import argparse
import os
from typing import Dict, List, Optional

import torch
from dotenv import load_dotenv

from cast.paths import CAA_VECTOR_DIR, REPO_ROOT

# Import the same functions used in generate_vectors.py to ensure consistency
from generate_vectors import RAW_CAA_VECTOR_DIR, get_vector_path

# Picks up HF_TOKEN from .env, as the rest of the pipeline does.
load_dotenv()

# Per-model, per-layer target norms recovered from the released vectors. Keys are
# the model identifiers built by resolve_model_name_path().
TARGET_NORMS_PATH = REPO_ROOT / "configs" / "normalization_target_norms.yaml"


def resolve_model_name_path(model_family: str, model_size: str) -> str:
    """
    Build the model identifier exactly as used by generate_vectors.py so that
    filenames match and vectors can be found.
    """
    model_family = model_family.lower()
    model_size_lower = model_size.lower()

    if model_family == "qwen":
        if model_size_lower in ("7b", "14b"):
            return f"qwen/qwen2.5-{model_size_lower}-instruct"
        raise ValueError(f"Unsupported model size: {model_size_lower}")
    elif model_family == "llama":
        if model_size_lower == "8b":
            return "meta-llama/llama-3.1-8b-instruct"
        else:
            raise ValueError(f"Unsupported model size: {model_size_lower}")
    else:
        raise ValueError(f"Unsupported model family: {model_family}")


def load_recorded_target_norms(model_name_path: str) -> Dict[int, float]:
    """Read the recorded per-layer target norms for one model.

    PyYAML is imported here rather than at module scope so that
    --compute_target_from_behaviors and --target_norm still work without it.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "Reading the recorded target norms needs PyYAML (pip install "
            "PyYAML). Without it, the scale has to be given explicitly with "
            "--target_norm or --compute_target_from_behaviors."
        ) from exc

    if not TARGET_NORMS_PATH.exists():
        raise FileNotFoundError(
            f"Recorded target norms not found at {TARGET_NORMS_PATH}. The file "
            "is part of this repository rather than a downloaded artifact; "
            "restore it, or set the scale explicitly with --target_norm / "
            "--compute_target_from_behaviors."
        )

    with open(TARGET_NORMS_PATH) as f:
        table = yaml.safe_load(f) or {}

    entry = table.get(model_name_path)
    if entry is None or "target_norms" not in entry:
        raise KeyError(
            f"{TARGET_NORMS_PATH} records no target norms for "
            f"{model_name_path}. Recorded models: {', '.join(sorted(table))}.\n"
            "Normalizing a model that is not in the table requires an explicit "
            "scale: pass --target_norm, or --compute_target_from_behaviors to "
            "take the mean over the behavior set."
        )

    return {int(layer): float(norm) for layer, norm in entry["target_norms"].items()}


def list_raw_behaviors() -> List[str]:
    """Behaviors that generate_vectors.py has written raw vectors for."""
    if not RAW_CAA_VECTOR_DIR.exists():
        return []
    return sorted(
        child.name for child in RAW_CAA_VECTOR_DIR.iterdir() if child.is_dir()
    )


def normalize_vectors(
    model_size: str,
    n_layers: int,
    model_family: str = "llama",
    behaviors: Optional[List[str]] = None,
    target_norm: Optional[float] = None,
    compute_target_from_behaviors: bool = False,
):
    """
    Normalize all steering vectors to have the same L2 norm.

    Args:
        model_size: Size of the model
        n_layers: Number of layers in the model
        model_family: Model family
        behaviors: Behaviors to rescale. ["all"] means every behavior found
            under artifacts/caa_vectors_raw/.
        target_norm: One explicit norm for every layer, instead of the recorded
            per-layer values.
        compute_target_from_behaviors: Take the target at each layer to be the
            mean raw norm over `behaviors`, instead of the recorded value. The
            scale then depends on the behavior set.
    """
    if not behaviors:
        raise ValueError(
            "normalize_vectors() needs an explicit behavior set. Pass ['all'] "
            "to rescale every behavior that has raw vectors."
        )

    if target_norm is not None and compute_target_from_behaviors:
        raise ValueError(
            "target_norm and compute_target_from_behaviors both set a target; "
            "pass at most one."
        )

    if "all" in behaviors:
        behaviors = list_raw_behaviors()
        if not behaviors:
            raise FileNotFoundError(
                f"No raw steering vectors found under {RAW_CAA_VECTOR_DIR}.\n"
                "Run generate_vectors.py first, or see the README section "
                "'Obtaining data and artifacts'."
            )
    else:
        # A repeated name is redundant work, and weights that behavior twice in
        # the mean under --compute_target_from_behaviors.
        duplicates = sorted({b for b in behaviors if behaviors.count(b) > 1})
        if duplicates:
            raise ValueError(
                f"--behaviors lists {', '.join(duplicates)} more than once; "
                "each behavior must appear exactly once."
            )

    # Resolve once per run so paths match those used during vector generation
    model_name_path = resolve_model_name_path(model_family=model_family, model_size=model_size)

    # Read the table before touching any vector, so that a model or layer the
    # table does not cover fails before artifacts/caa_vectors/ is half written.
    recorded_norms: Dict[int, float] = {}
    if target_norm is None and not compute_target_from_behaviors:
        recorded_norms = load_recorded_target_norms(model_name_path)
        absent_layers = [
            layer for layer in range(n_layers) if layer not in recorded_norms
        ]
        if absent_layers:
            raise KeyError(
                f"{TARGET_NORMS_PATH} records no target norm for "
                f"{model_name_path} at layer(s) "
                f"{', '.join(str(layer) for layer in absent_layers)} "
                f"(it covers layers 0-{max(recorded_norms)}).\n"
                "Check --n_layers, or set the scale explicitly with "
                "--target_norm / --compute_target_from_behaviors."
            )

    # Check the whole set before writing anything, so that a missing raw vector
    # fails the run rather than leaving artifacts/caa_vectors/ holding some
    # behaviors at some layers and not others.
    missing = {}
    for behavior in behaviors:
        absent = [
            layer
            for layer in range(n_layers)
            if not os.path.exists(get_vector_path(behavior, layer, model_name_path))
        ]
        if absent:
            missing[behavior] = absent
    if missing:
        lines = [
            f"  {behavior}: {len(layers)}/{n_layers} layers missing "
            f"(first: layer {layers[0]})"
            for behavior, layers in sorted(missing.items())
        ]
        raise FileNotFoundError(
            f"Raw steering vectors for {model_name_path} are missing under "
            f"{RAW_CAA_VECTOR_DIR}:\n"
            + "\n".join(lines)
            + "\n\nNothing was written. Generate the missing vectors with "
            "generate_vectors.py, or name a behavior set you have in full.\n"
            "This repository ships code only. See the README section "
            "'Obtaining data and artifacts'."
        )

    if target_norm is not None:
        target_source = f"explicit --target_norm {target_norm}"
    elif compute_target_from_behaviors:
        target_source = "mean raw norm over the behavior set"
    else:
        target_source = f"recorded target norms ({TARGET_NORMS_PATH})"

    print(f"Normalizing {len(behaviors)} behaviors over {n_layers} layers:")
    print(f"  in:  {RAW_CAA_VECTOR_DIR}")
    print(f"  out: {CAA_VECTOR_DIR}")
    print(f"  behaviors: {behaviors}")
    print(f"  target:    {target_source}")

    for layer in range(n_layers):
        norms = {}
        vecs = {}
        new_paths = {}

        for behavior in behaviors:
            vec_path = get_vector_path(behavior, layer, model_name_path)
            vec = torch.load(vec_path)
            norm = vec.norm().item()
            vecs[behavior] = vec
            norms[behavior] = norm
            new_paths[behavior] = str(
                CAA_VECTOR_DIR / behavior / os.path.basename(vec_path)
            )

        if target_norm is not None:
            layer_target = target_norm
        elif compute_target_from_behaviors:
            layer_target = torch.tensor(list(norms.values())).mean().item()
        else:
            layer_target = recorded_norms[layer]

        # normalize all vectors to have the same norm
        for behavior in behaviors:
            vecs[behavior] = vecs[behavior] * layer_target / norms[behavior]
        # save the normalized vectors
        for behavior in behaviors:
            os.makedirs(os.path.dirname(new_paths[behavior]), exist_ok=True)
            torch.save(vecs[behavior], new_paths[behavior])


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rescale raw CAA steering vectors so that every behavior has the "
            "same L2 norm at each layer. The norm is read per model and per "
            "layer from configs/normalization_target_norms.yaml, the scale the "
            "released vectors carry."
        )
    )
    parser.add_argument("--model_family", type=str, choices=["llama", "qwen"], default="llama",
                       help="Model family (llama or qwen)")
    parser.add_argument("--model_size", type=str,
                       choices=["7b", "8b", "14b"],
                       default="8b",
                       help="Model size (qwen: 7b or 14b; llama: 8b)")
    parser.add_argument("--n_layers", type=int, required=True,
                       help="Number of layers in the model")
    parser.add_argument("--behaviors", nargs="+", type=str, required=True,
                       help=(
                           "Behaviors to rescale, or 'all' for every behavior "
                           "found under artifacts/caa_vectors_raw/. This picks "
                           "which vectors are written, not what they are "
                           "rescaled to. Missing raw vectors are an error, not "
                           "a skip."
                       ))
    parser.add_argument("--target_norm", type=float, default=None,
                       help=(
                           "Escape hatch: rescale every layer to this norm "
                           "instead of the recorded per-layer value. Only for "
                           "vectors that are not meant to match the released "
                           "scale -- a different norm changes what every "
                           "steering multiplier downstream means."
                       ))
    parser.add_argument("--compute_target_from_behaviors", action="store_true",
                       help=(
                           "Escape hatch: take each layer's target to be the "
                           "mean raw norm over --behaviors, as this script did "
                           "before the norms were recorded. Normalizing over a "
                           "different behavior set silently changes what every "
                           "steering multiplier means, which is why the "
                           "recorded table is the default."
                       ))

    args = parser.parse_args()

    normalize_vectors(
        model_size=args.model_size,
        n_layers=args.n_layers,
        model_family=args.model_family,
        behaviors=args.behaviors,
        target_norm=args.target_norm,
        compute_target_from_behaviors=args.compute_target_from_behaviors,
    )


if __name__ == "__main__":
    main()
    
