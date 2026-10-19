"""
Generate completions on benign prompts with a CAA / CAST steering vector applied.

This is the false-refusal half of the evaluation: the same generation path as
`run_steering.py` but on harmless requests and with no attack template, so the
judge can count how often steering makes the model refuse something it should
answer. The paper reports XSTest-safe and Alpaca.

Usage (CAA vectors from artifacts/caa_vectors):
python run_benign.py \
    --layers 13 \
    --multipliers -3 -2 -1 0 1 2 3 \
    --behaviors sycophancy myopic-reward length_conciseness \
    --model_size 8b \
    --model_family llama \
    --benign_source xtest

Usage (a single vector file, e.g. one produced by train_cast.py):
python run_benign.py \
    --layers 19 \
    --multipliers -1.5 1.5 \
    --behaviors power-seeking-inclination \
    --model_size 7b \
    --model_family qwen \
    --custom_vector_path artifacts/cast_vectors/qwen2.5-7b-instruct/main/final_vector_{behavior}.pt \
    --batch_size 16

When --custom_vector_path is provided the vector is loaded directly from that file
(for every layer in --layers) and output filenames use the prefix ``results_single_``
instead of ``results_``.

This script only processes benign user inputs without any attack methods.
"""

import json
import pandas as pd
import os
from dotenv import load_dotenv
import argparse
from typing import List, Dict, Optional
from tqdm import tqdm
import numpy as np
import torch
import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM

# Removed legacy CAA path dependency

# Import from steering-vectors library
from steering_vectors import SteeringVector
from tokenizer import create_chat_prompt, detect_model_family

from steering_settings import SteeringSettings, PAPER_BEHAVIORS

from cast.paths import (
    BENIGN_DATA_DIR,
    REFUSAL_ABLATED_VECTOR_DIR,
    REPO_ROOT,
    caa_vector_path,
    require,
)

# Local functions to use steering-vectors directory paths
def make_tensor_save_suffix(layer, model_name_path: str) -> str:
    """Convert model name to tensor save suffix format (matches steering-vectors naming convention)"""
    # Convert model path to match the actual file naming convention
    # e.g., "meta-llama/llama-3.1-8b-instruct" -> "meta_llama_llama_3.1_8b_instruct"
    model_name_clean = model_name_path.replace('/', '_').replace('-', '_')
    return f"{layer}_{model_name_clean}"

def get_vector_path(behavior: str, layer, model_name_path: str, use_ablated: bool = False):
    """Path of the normalised steering vector for (behaviour, layer, model).

    The plain CAA vector lives in `artifacts/caa_vectors`; the direction-ablated
    baseline lives in `artifacts/refusal_ablated_vectors` under the same name
    plus a `_direction` suffix, which is how `run_steering.py` loads it too.
    """
    if use_ablated:
        return (
            REFUSAL_ABLATED_VECTOR_DIR
            / behavior
            / f"vec_layer_{make_tensor_save_suffix(layer, model_name_path)}_direction.pt"
        )
    return caa_vector_path(behavior, layer, model_name_path)

def get_steering_vector(behavior, layer, model_name_path, use_ablated: bool = False):
    """Load the normalised steering vector for (behaviour, layer, model)."""
    vector_path = get_vector_path(behavior, layer, model_name_path, use_ablated=use_ablated)
    print(f"Loading vector from {vector_path}")
    require(vector_path, "Steering vector")
    return t.load(vector_path, weights_only=True)

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

# Local benign prompt set (see the README section "Obtaining data and artifacts")
XTEST_DATASET_PATH = BENIGN_DATA_DIR / "xtest_safe.json"

# Helper: build simple question from benign item (no attack methods)
def build_benign_user_input(local_item: Dict[str, str]) -> str:
    """Extract the benign goal/question from the item without any attack modifications."""
    return local_item.get("goal", "")


def load_benign_data(source: str = "xtest"):
    """Load and prepare the benign behaviors data for false positive testing.

    The two prompt sets the paper reports:
    source: "xtest" to use local data/benign/xtest_safe.json (XSTest safe split),
            "alpaca" to use the fixed 100 Alpaca inputs recorded for the paper.
    """
    if source == "xtest":
        # Local JSON benign dataset, one prompt per "instruction" field
        dataset_path = XTEST_DATASET_PATH
        require(dataset_path, f"Benign prompt set '{source}'")
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        benign_data = []
        for i, item in enumerate(data):
            instruction = item.get("instruction", "")
            benign_data.append(
                {
                    "goal": instruction,
                    # No explicit target field in these JSON datasets; leave empty.
                    "target": "",
                    "behavior_id": i,
                }
            )

        print(
            f"Loaded {len(benign_data)} benign behaviors from {dataset_path} for false positive testing"
        )
        return benign_data

    if source == "alpaca":
        path = require(BENIGN_DATA_DIR / "alpaca_100.json", "Paper Alpaca inputs")
        with open(path, encoding="utf-8") as stream:
            benign_data = json.load(stream)
        print(f"Loaded {len(benign_data)} fixed Alpaca inputs from {path}")
        return benign_data

    raise ValueError(f"Unknown benign data source: {source}")




def get_model_path(size: str, model_family: str = "llama") -> str:
    """Get the model path based on size and model family"""
    if model_family == "llama":
        if size == "8b":
            return "meta-llama/llama-3.1-8b-instruct"
        else:
            raise ValueError(f"Unsupported model size: {size}")
    elif model_family == "qwen":
        return f"qwen/qwen2.5-{size.lower()}-instruct"
    else:
        raise ValueError(f"Unsupported model family: {model_family}")



def get_model_and_tokenizer(model_name: str, use_8bit: bool = True):
    """Load model and tokenizer with optional 8-bit quantization"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if use_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            load_in_8bit=False, 
            # device_map='auto',
            device_map='cuda',
            torch_dtype=torch.float16,
            token=HUGGINGFACE_TOKEN
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            # device_map='auto',
            device_map='cuda',
            token=HUGGINGFACE_TOKEN
        )
    return model, tokenizer


def process_item_benign_open_ended(
    item: Dict[str, str],
    model,
    tokenizer,
    system_prompt: Optional[str],
    save_residuals: bool = False,
) -> Dict[str, str]:
    """Process benign items for open-ended generation without any attacks.

    `save_residuals` only adds a separate forward pass whose hidden states are
    returned alongside the completion; generation is greedy and does not depend
    on it, so the completions are identical either way.
    """
    goal = item.get("goal", "")
    
    # Use simple benign user input without any attack modifications
    question =  (goal)
    
    # Create chat prompt (no answer prefix for benign testing)
    prompt = create_chat_prompt(
        tokenizer=tokenizer,
        question=question,
        answer=None,
        system_prompt=system_prompt,
        use_chat_format=True
    )
    
    # Tokenize and generate
    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Capture residual stream at the last prompt token (after steering is applied)
    last_token_residual = None
    if save_residuals:
        with torch.no_grad():
            preforward = model(**inputs, output_hidden_states=True, use_cache=True)
            # Hidden states of the last prompt token across all transformer layers (exclude embeddings)
            last_token_residual = (
                torch.stack([hs[:, -1, :].float() for hs in preforward.hidden_states[1:]], dim=1)
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    model_output = response.strip()

    result = {
        "model_output": model_output,
        "goal": item.get("goal", ""),
        "target": item.get("target", ""),
        "behavior_id": item.get("behavior_id", ""),
    }
    if save_residuals:
        result["last_token_residual"] = last_token_residual
    return result


def process_batch_benign_open_ended(
    items: List[Dict[str, str]],
    model,
    tokenizer,
    system_prompt: Optional[str],
    save_residuals: bool = False,
) -> List[Dict[str, str]]:
    """Process a batch of benign items for open-ended generation."""
    batch_size = len(items)

    # Build prompts for all items in the batch
    prompts = []
    for item in items:
        question = build_benign_user_input(item)
        prompt = create_chat_prompt(
            tokenizer=tokenizer,
            question=question,
            answer=None,
            system_prompt=system_prompt,
            use_chat_format=True,
        )
        prompts.append(prompt)

    # Tokenize batch with left padding for generation
    padding_side_original = tokenizer.padding_side
    tokenizer.padding_side = "left"

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    # Forward pass to extract residuals
    batch_residuals = []
    if save_residuals:
        with torch.no_grad():
            preforward = model(**inputs, output_hidden_states=True)
            for batch_idx in range(batch_size):
                last_token_hidden = torch.stack(
                    [
                        hs[batch_idx, -1, :].float()
                        for hs in preforward.hidden_states[1:]
                    ],
                    dim=0,
                )  # Shape: [num_layers, hidden_dim]
                last_token_residual = (
                    last_token_hidden.unsqueeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                batch_residuals.append(last_token_residual)

    # Batched generation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    tokenizer.padding_side = padding_side_original

    # Decode responses (skip input tokens)
    results: List[Dict[str, str]] = []
    input_seq_length = inputs["input_ids"].shape[1]
    for i, item in enumerate(items):
        response = tokenizer.decode(
            outputs[i][input_seq_length:], skip_special_tokens=True
        )
        model_output = response.strip()
        result = {
            "model_output": model_output,
            "goal": item.get("goal", ""),
            "target": item.get("target", ""),
            "behavior_id": item.get("behavior_id", ""),
        }
        if save_residuals:
            result["last_token_residual"] = batch_residuals[i]
        results.append(result)

    return results


def run_steering(
    layers: List[int],
    multipliers: List[int],
    settings: SteeringSettings,
    overwrite: bool = False,
    batch_size: int = 1,
    results_root: str = "benign_results",
    run_tag: Optional[str] = None,
    save_residuals: bool = False,
):
    """
    layers: List of layers to test steering on.
    multipliers: List of multipliers to test steering with.
    settings: SteeringSettings object.
    results_root: Directory the results tree is written under. A relative path is
        taken relative to the repository root, so the caller need not cd anywhere.
    run_tag: Optional label inserted into the result filename prefix. Use it to
        keep the runs of an ablation apart; without it filenames are unchanged.
    save_residuals: Also dump the last-prompt-token hidden states per run
        (~40 MB of .npy per configuration). Off by default.

    When settings.custom_vector_path is set the vector is loaded directly from
    that file (shared across all layers) and result filenames use the prefix
    ``results_single_``.
    """
    base_results_dir = str(REPO_ROOT / results_root)
    save_results_dir = os.path.join(
        base_results_dir,
        settings.model_family,
        settings.model_size.replace(".", "-"),
        settings.behavior,
    )
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir, exist_ok=True)

    # Only support benign open_ended testing
    if settings.type != "open_ended":
        raise ValueError(
            "This script only supports open_ended testing with benign data"
        )

    # Load benign behaviors data for false positive testing
    benign_source = getattr(settings, "benign_source", "xtest")
    test_data = load_benign_data(source=benign_source)

    # Initialize model
    model_name_path = get_model_path(
        settings.model_size,
        settings.model_family
    )
    
    model, tokenizer = get_model_and_tokenizer(model_name_path)
    # Ensure no sampling params are set when using greedy decoding
    if hasattr(model, "generation_config"):
        try:
            gc = model.generation_config
            gc.do_sample = False
            # Unset sampling-only params to avoid warnings
            gc.top_k = None
            gc.top_p = None
            gc.temperature = None
        except Exception:
            pass

    # Determine whether we are using a custom (single) vector file
    custom_vector_path = getattr(settings, "custom_vector_path", None)

    # Match vector dtype to model parameters once
    model_dtype = next(model.parameters()).dtype

    for layer in layers:
        if custom_vector_path is not None:
            # --- custom vector mode ---
            # Use str.replace (not str.format) so unrelated braces in the path do not
            # raise KeyError, and so a path can contain only one of the placeholders.
            # A relative path is taken relative to the repository root; an absolute
            # one is used as given.
            vector_path = REPO_ROOT / custom_vector_path.replace(
                "{layer}", str(layer)
            ).replace("{behavior}", settings.behavior)
            print(f"Loading custom steering vector from: {vector_path}")
            require(vector_path, "Custom steering vector")
            vector = t.load(vector_path, weights_only=True)
            vector = vector.to(model_dtype).to(model.device)
        else:
            # --- standard vector mode ---
            name_path = model_name_path
            if settings.override_vector_model is not None:
                name_path = settings.override_vector_model
            if settings.override_vector is not None:
                vector = get_steering_vector(
                    settings.behavior,
                    settings.override_vector,
                    name_path,
                    use_ablated=getattr(settings, "use_ablated_vectors", False),
                )
            else:
                vector = get_steering_vector(
                    settings.behavior,
                    layer,
                    name_path,
                    use_ablated=getattr(settings, "use_ablated_vectors", False),
                )
            if settings.model_size != "7b":
                vector = vector.half()
            vector = vector.to(model.device)

        for multiplier in multipliers:
            result_save_suffix = settings.make_result_save_suffix(
                layer=layer, multiplier=multiplier, model_family=settings.model_family
            )
            if custom_vector_path is not None:
                filename_prefix = "results_single_"
            elif getattr(settings, "use_ablated_vectors", False):
                filename_prefix = "results_ablation_"
            else:
                filename_prefix = "results_"
            if run_tag:
                filename_prefix = f"{filename_prefix}{run_tag}_"
            save_filename = os.path.join(
                save_results_dir,
                f"{filename_prefix}{result_save_suffix}_benign_source={benign_source}.json",
            )
            if os.path.exists(save_filename) and not overwrite:
                print("Found existing", save_filename, "- skipping")
                continue
            results = []
            residuals_list = []

            # Apply steering vector using SteeringVector.apply (steering-vectors approach)
            steering_vector = SteeringVector(
                layer_activations={layer: vector},
                layer_type="decoder_block",
            )

            with steering_vector.apply(
                model, multiplier=multiplier, min_token_index=0
            ):
                for batch_start in tqdm(
                    range(0, len(test_data), batch_size),
                    desc=f"Layer {layer}, multiplier {multiplier}",
                ):
                    batch_end = min(batch_start + batch_size, len(test_data))
                    batch_items = test_data[batch_start:batch_end]

                    if len(batch_items) == 1:
                        batch_results = [
                            process_item_benign_open_ended(
                                item=batch_items[0],
                                model=model,
                                tokenizer=tokenizer,
                                system_prompt=None,
                                save_residuals=save_residuals,
                            )
                        ]
                    else:
                        batch_results = process_batch_benign_open_ended(
                            items=batch_items,
                            model=model,
                            tokenizer=tokenizer,
                            system_prompt=None,
                            save_residuals=save_residuals,
                        )

                    for result in batch_results:
                        # Collect residual and remove from JSON result for size efficiency
                        if "last_token_residual" in result:
                            try:
                                residuals_list.append(
                                    np.array(
                                        result["last_token_residual"],
                                        dtype=np.float32,
                                    )
                                )
                            except Exception:
                                pass
                            result.pop("last_token_residual", None)
                        results.append(result)

            # Save residual tensors separately as a single .npy file
            if residuals_list:
                try:
                    residuals_array = np.stack(residuals_list, axis=0)
                except Exception:
                    residuals_array = np.array(residuals_list, dtype=object)
                tensor_filename = save_filename.replace(".json", "_residuals.npy")
                np.save(tensor_filename, residuals_array)
                print(f"Saved residuals to {tensor_filename}")

            with open(save_filename, "w",) as f:
                json.dump(results, f, indent=4)

            print(f"Saved {len(results)} results to {save_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    parser.add_argument(
        "--behaviors", type=str, nargs="+", choices=PAPER_BEHAVIORS,
        default=["sycophancy"],
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        choices=["pos", "neg"],
        required=False,
    )
    parser.add_argument("--override_vector", type=int, default=None)
    parser.add_argument("--override_vector_model", type=str, default=None)
    parser.add_argument(
        "--custom_vector_path",
        type=str,
        default=None,
        help=(
            "Path to a custom steering vector .pt file, e.g. one produced by "
            "train_cast.py. When set, this vector is used directly (for every layer "
            "in --layers) and output filenames use the prefix 'results_single_'. "
            "The path may contain the placeholders '{behavior}' and '{layer}', which "
            "are substituted with the current behaviour and layer so that one "
            "invocation can sweep several of them."
        ),
    )
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["7b", "8b", "14b"],
        default="7b",
    )
    parser.add_argument(
        "--model_family", type=str, choices=["llama", "qwen"], default="llama"
    )
    parser.add_argument("--override_model_weights_path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument(
        "--use_ablated_vectors",
        action="store_true",
        default=False,
        help="If set, load the direction-ablated baseline vectors from artifacts/refusal_ablated_vectors instead of the CAA vectors in artifacts/caa_vectors.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing benign items. Default is 1 (no batching).",
    )
    parser.add_argument(
        "--benign_source",
        type=str,
        choices=["xtest", "alpaca"],
        default="xtest",
        help=(
            "Source of benign prompts, the two the paper reports: "
            "'xtest' for the XSTest safe split in data/benign/xtest_safe.json "
            "(default), 'alpaca' for the first 100 rows of tatsu-lab/alpaca "
            "(HF parquet)."
        ),
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="benign_results",
        help=(
            "Directory the results tree is written under; a relative path is taken "
            "relative to the repository root. Give each ablation its own root "
            "(e.g. benign_results_no_frr) instead of relying on the vector path."
        ),
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help=(
            "Optional label inserted into the result filename prefix "
            "(results_single_<run_tag>_...). Without it filenames are unchanged."
        ),
    )
    parser.add_argument(
        "--save_residuals",
        action="store_true",
        default=False,
        help=(
            "Also dump the last-prompt-token hidden states next to each results "
            "JSON as _residuals.npy (~40 MB per configuration). Off by default and "
            "not needed for any number in the paper; the analysis that consumed "
            "these dumps is not part of this release. Completions are identical "
            "either way."
        ),
    )

    args = parser.parse_args()

    steering_settings = SteeringSettings()
    steering_settings.type = "open_ended"  # Fixed to open_ended for benign testing
    steering_settings.system_prompt = args.system_prompt
    steering_settings.override_vector = args.override_vector
    steering_settings.override_vector_model = args.override_vector_model
    steering_settings.custom_vector_path = args.custom_vector_path
    steering_settings.model_size = args.model_size
    steering_settings.override_model_weights_path = args.override_model_weights_path
    steering_settings.model_family = args.model_family
    steering_settings.use_ablated_vectors = bool(args.use_ablated_vectors)
    # Configure benign dataset source (see --benign_source)
    steering_settings.benign_source = args.benign_source

    for behavior in args.behaviors:
        steering_settings.behavior = behavior
        print(f"Run benign steering evaluation for behavior: {behavior}")
        run_steering(
            layers=args.layers,
            multipliers=args.multipliers,
            settings=steering_settings,
            overwrite=args.overwrite,
            batch_size=args.batch_size,
            results_root=args.results_root,
            run_tag=args.run_tag,
            save_residuals=args.save_residuals,
        )
