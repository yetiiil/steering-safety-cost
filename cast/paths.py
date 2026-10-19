"""Repository-relative path resolution.

Every path in this repository is resolved from the location of the source file,
never from the current working directory.  This lets any script be launched from
anywhere without the caller having to `cd` into the repository first.

Layout
------
    <repo>/data/train/            harmful prompt splits used to optimise CAST
    <repo>/data/benign/           benign prompts for the false-refusal constraint
    <repo>/data/behaviors/<b>/    CAA behaviour datasets (generate / test splits)
    <repo>/data/attacks/          static attack prompts (AutoDAN, PAIR, GCG suffixes)
    <repo>/artifacts/DIM/         refusal directions from the DIM pipeline
    <repo>/artifacts/caa_vectors/ normalised CAA steering vectors
    <repo>/artifacts/cast_vectors/ vectors produced by train_cast.py

Extracted inputs and generated artifacts are ignored. The fixed input archive
under data/paper is tracked; scripts/00_download_data.py restores its inputs.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
TRAIN_DATA_DIR = DATA_DIR / "train"
BENIGN_DATA_DIR = DATA_DIR / "benign"
BEHAVIOR_DATA_DIR = DATA_DIR / "behaviors"
ATTACK_DATA_DIR = DATA_DIR / "attacks"

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
CAA_VECTOR_DIR = ARTIFACTS_DIR / "caa_vectors"
CAST_VECTOR_DIR = ARTIFACTS_DIR / "cast_vectors"
REFUSAL_ABLATED_VECTOR_DIR = ARTIFACTS_DIR / "refusal_ablated_vectors"


def dim_direction_dir(model_id: str) -> Path:
    """Directory holding the DIM refusal direction for `model_id`.

    Honours SAVE_DIR/DIM_DIR from the environment (as the upstream DIM pipeline
    does) and falls back to `<repo>/artifacts/DIM/<model_id>`.

    A relative SAVE_DIR is anchored to the repository root, not to the working
    directory: the DIM pipeline is launched after `cd third_party/refusal_direction`,
    so a CWD-relative `SAVE_DIR=artifacts` would write the direction where the
    later stages never look for it.
    """
    save_dir = os.getenv("SAVE_DIR")
    dim_dir = os.getenv("DIM_DIR", "DIM")
    if save_dir:
        base = Path(save_dir)
        if not base.is_absolute():
            base = REPO_ROOT / base
        return base / dim_dir / model_id
    return ARTIFACTS_DIR / dim_dir / model_id


def behavior_dir(behavior: str) -> Path:
    """Directory holding the datasets for one CAA behaviour."""
    return BEHAVIOR_DATA_DIR / behavior


def caa_vector_path(behavior: str, layer: int, model_path: str) -> Path:
    """Path of the normalised CAA steering vector for (behaviour, layer, model).

    The filename convention is inherited from the CAA reference implementation:
    the HuggingFace model id has `/` and `-` replaced by `_`.
    """
    model_filename = model_path.replace("/", "_").replace("-", "_")
    return CAA_VECTOR_DIR / behavior / f"vec_layer_{layer}_{model_filename}.pt"


def require(path: Path, what: str) -> Path:
    """Return `path`, or raise with a message pointing at the download script."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{what} not found at {path}.\n"
            "See the README section 'Obtaining data and artifacts'. "
            "Run scripts/00_download_data.py for the fixed paper inputs; "
            "produce model artifacts with stages 1-5."
        )
    return Path(path)
