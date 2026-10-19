"""
Measure the *behavioural* effect of a steering vector, with no attack involved.

For each (behaviour, layer, multiplier) the script applies one steering vector and
either scores the forced A/B choice (`--type ab`, the CAA behaviour score reported
in Figure 2 and the layer sweep of Figure 3) or generates a free-form continuation
(`--type open_ended`, the completions judged for Table 10 and summarised by
`length_effect_stats.py`).

Usage:
    python run_open_ended.py --behaviors sycophancy --layers 18 \\
        --multipliers 0.1 0.5 1 2 5 10 --type ab --model_size 7b --model_family qwen

    # CAST vector for the same behaviour, results kept in their own tree
    python run_open_ended.py --behaviors corrigible-more-HHH --layers 13 \\
        --multipliers -1 0 1 --type open_ended \\
        --custom_vector 'artifacts/cast_vectors/qwen2.5-7b-instruct/dim_1_masked_kl/sanitized_{behavior}_qwen2.5-7b-instruct_masked_kl.pt' \\
        --results_root results_dim_1 --run_tag dim_1

Datasets and vectors are resolved relative to the repository root (see `cast.paths`
and the README section "Obtaining data and artifacts"); the working directory does
not matter.
"""

import json
import os
import argparse
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import torch
import math
from dataclasses import dataclass
from collections.abc import Iterable

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from steering_vectors import SteeringVector
from dotenv import load_dotenv
from tokenizer import create_chat_prompt

from cast.paths import CAA_VECTOR_DIR, REPO_ROOT, behavior_dir, require
# The raw (un-normalised) CAA tree is defined where it is written, so the two
# cannot drift apart; normalize_vectors.py imports it from the same place.
from generate_vectors import RAW_CAA_VECTOR_DIR
# steering_settings.py is the single source of truth for the behaviour surface.
from steering_settings import MAIN_BEHAVIORS, PAPER_BEHAVIORS

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

# Swept when --behaviors is not given: the three behaviours carrying the main
# results.  The three Appendix B.7 behaviours have to be asked for by name.
DEFAULT_BEHAVIORS = list(MAIN_BEHAVIORS)

@dataclass
class TokenProb:
    token_id: int
    logprob: float
    text: str

@dataclass
class TextProbs:
    text: str
    token_probs: List[TokenProb]

    @property
    def sum_logprobs(self) -> float:
        return sum([tp.logprob for tp in self.token_probs])

    def __repr__(self) -> str:
        return f"TextProbs({self.text}:{self.sum_logprobs:.2f})"

def _single_token_id_for(text: str, tokenizer: PreTrainedTokenizerBase) -> Optional[int]:
    """Return the token id if `text` encodes to exactly one token; else None."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    return None


def get_robust_a_b_probs(logits, tokenizer):
    last_token_logits = logits[0, -1, :]
    last_token_probs = torch.softmax(last_token_logits, dim=-1)

    a_prob, b_prob = 0.0, 0.0


    for text, target in [(" A", "a"), ("A", "a"), (" B", "b"), ("B", "b")]:
        token_id = _single_token_id_for(text, tokenizer)
        if token_id is not None:
            prob = last_token_probs[token_id].item()
            if target == "a":
                a_prob = max(a_prob, prob)
            else:
                b_prob = max(b_prob, prob)

    return a_prob, b_prob


def _compute_min_token_index_for_prompt(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    model_family: str,
    question: str,
    answer: Optional[str] = "Choice: (",
) -> int:
    """Compute the token index of the last token of the assistant header.

    Steering starts from this index so the first generated token is steered.

    Pass answer="Choice: (" for A/B (matches the actual prompt sent to the model).
    Pass answer=None for open-ended generation (no forced prefix).

    Uses a string-split approach: find the assistant header in the prompt string,
    encode only the prefix up to and including it, and return len(prefix_ids)-1.
    This avoids standalone-encoding artefacts (e.g. spurious leading tokens) that
    would arise from encoding the header string in isolation.
    """
    prompt = create_chat_prompt(
        tokenizer=tokenizer,
        question=question,
        answer=answer,
        use_chat_format=True,
    )

    # Choose assistant header string per family
    family = (model_family or "").lower()
    if family == "llama":
        assistant_hdr = "<|start_header_id|>assistant<|end_header_id|>\n"
    elif family == "qwen":
        assistant_hdr = "<|im_start|>assistant\n"
    else:
        # Unknown family: fallback to applying from start
        return 0

    # Find the last occurrence of the header in the prompt string, then encode
    # only the prefix up to and including it.  Because we encode a contiguous
    # slice of the full prompt, the tokenization is identical to how those
    # tokens appear in the full sequence — no standalone-encoding artefacts.
    hdr_start = prompt.rfind(assistant_hdr)
    if hdr_start == -1:
        return 0
    prefix = prompt[: hdr_start + len(assistant_hdr)]
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    if not prefix_ids:
        return 0
    return len(prefix_ids) - 1


def _get_model_dtype(model_name: str) -> torch.dtype:
    """Get appropriate dtype for model to avoid NaN issues"""
    # float16 for every model evaluated here (Qwen2.5 and Llama-3.1 Instruct)
    return torch.float16

def get_model_and_tokenizer(model_name: str):
    """Load model and tokenizer with optional 8-bit quantization"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_dtype = _get_model_dtype(model_name)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=model_dtype,
        device_map='auto'
    )
    return model, tokenizer

def get_model_path(size: str, model_family: str = "llama") -> str:
    """Get the model path based on size and model family"""
    if model_family == "llama":
        return f"meta-llama/llama-3.1-8b-instruct"
    elif model_family == "qwen":
        return f"qwen/qwen2.5-{size.lower()}-instruct"
    else:
        raise ValueError(f"Unsupported model family: {model_family}")


def make_tensor_save_suffix(layer, model_name_path: str) -> str:
    """Convert model name to tensor save suffix format (matches CAA implementation)"""
    return f"{layer}_{model_name_path.replace('/', '_').replace('-', '_')}"

def get_vector_path(behavior: str, layer, model_name_path: str, normalized=False) -> str:
    """Get the path to a steering vector file (matches CAA implementation).

    `normalized=True` resolves to the per-layer normalised tree written by
    normalize_vectors.py (`artifacts/caa_vectors`), which is what every caller in
    this repository asks for; `normalized=False` resolves to the raw tree written
    by generate_vectors.py (`artifacts/caa_vectors_raw`). Normalisation rescales
    every behaviour to a shared per-layer norm, so the same multiplier means
    different things in the two trees -- they are not interchangeable.
    """
    vector_dir = CAA_VECTOR_DIR if normalized else RAW_CAA_VECTOR_DIR
    return str(
        vector_dir
        / behavior
        / f"vec_layer_{make_tensor_save_suffix(layer, model_name_path)}.pt"
    )

def get_steering_vector(behavior, layer, model_name_path, normalized=False, custom_vector_path=None):
    """Load a steering vector (matches CAA implementation exactly)
    
    Args:
        behavior: Behavior name
        layer: Layer number
        model_name_path: Model name/path
        normalized: True for the normalized vectors in artifacts/caa_vectors,
                    False for the raw ones in artifacts/caa_vectors_raw
        custom_vector_path: Optional custom vector path. Can contain {layer} and/or {behavior}
                          placeholders that will be substituted with the current layer index and
                          behavior name (useful for sweeping multiple behaviors in one run).
                          If None, uses default path based on behavior and layer.
    """
    if custom_vector_path is not None:
        # Use str.replace (not str.format) so unrelated braces in the path do not
        # raise KeyError, and so a path can contain only one of the placeholders.
        # A relative path is taken relative to the repository root; an absolute one
        # is used as given.
        vector_path = REPO_ROOT / custom_vector_path.replace("{layer}", str(layer)).replace(
            "{behavior}", behavior
        )
        return torch.load(require(vector_path, "Custom steering vector"), weights_only=True)
    else:
        vector_path = get_vector_path(behavior, layer, model_name_path, normalized=normalized)
        what = "normalized CAA steering vector" if normalized else "raw CAA steering vector"
        return torch.load(require(vector_path, what), weights_only=True)

def load_test_data(behavior: str, data_type: str) -> List[Dict[str, str]]:
    """Load the test split for one behaviour from `data/behaviors/<behavior>/`."""
    if data_type == "ab":
        filename = "test_dataset_ab.json"
    elif data_type == "open_ended":
        filename = "test_dataset_open_ended.json"
    else:
        raise ValueError(f"Unsupported data type: {data_type}. Supported: 'ab', 'open_ended'.")

    test_data_path = require(
        behavior_dir(behavior) / filename,
        f"{data_type} test split for behaviour '{behavior}'",
    )
    with open(test_data_path, "r") as f:
        return json.load(f)

def process_item_ab(
    item: Dict[str, str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase
) -> Dict[str, str]:
    """Process A/B test item (matches CAA implementation pattern)"""
    question: str =  item["question"]
    answer_matching_behavior = item["answer_matching_behavior"]
    answer_not_matching_behavior = item["answer_not_matching_behavior"]
    
    # Create prompt ending with "(" to get logits for A/B choice
    prompt = create_chat_prompt(
        tokenizer=tokenizer,
        question=question,
        answer="Choice: (",
        use_chat_format=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        outputs = model(**inputs.to(model.device))
        logits = outputs.logits
    
    # Use robust approach that includes both space-prefixed and non-space variants
    a_prob, b_prob = get_robust_a_b_probs(logits, tokenizer)
    
    result: Dict[str, object] = {
        "question": question,
        "answer_matching_behavior": answer_matching_behavior,
        "answer_not_matching_behavior": answer_not_matching_behavior,
        "a_prob": a_prob,
        "b_prob": b_prob,
    }
    
    return result


def process_item_open_ended(
    item: Dict[str, str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    max_new_tokens: int = 100,
) -> Dict[str, str]:
    """Process a single open-ended test item by generating free-form text."""
    question: str = item["question"]
    prompt = create_chat_prompt(
        tokenizer=tokenizer,
        question=question,
        answer=None,
        use_chat_format=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    input_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0, input_len:]
    model_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return {
        "question": question,
        "model_output": model_output,
    }


def process_batch_open_ended(
    items: List[Dict[str, str]],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    max_new_tokens: int = 100,
) -> List[Dict[str, str]]:
    """Process a batch of open-ended items, using left-padding for generation."""
    prompts = [
        create_chat_prompt(
            tokenizer=tokenizer,
            question=item["question"],
            answer=None,
            use_chat_format=True,
        )
        for item in items
    ]

    padding_side_original = tokenizer.padding_side
    tokenizer.padding_side = "left"  # required for batched generation
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(model.device)
    tokenizer.padding_side = padding_side_original

    input_seq_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    results = []
    for i, item in enumerate(items):
        generated_ids = output_ids[i, input_seq_len:]
        model_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        results.append({
            "question": item["question"],
            "model_output": model_output,
        })
    return results


def test_steering(
    layers: List[int],
    multipliers: List[float],
    behavior: str,
    data_type: str,
    model_size: str = "7b",
    model_family: str = "llama",
    overwrite: bool = False,
    custom_vector_path: Optional[str] = None,
    batch_size: int = 1,
    results_root: str = "results",
    run_tag: Optional[str] = None,
):
    """
    Test steering with different layers and multipliers using pre-trained vectors.

    Args:
        layers: List of layers to test steering on.
        multipliers: List of multipliers to test steering with.
        behavior: The behavior to test.
        data_type: Type of test data ("ab" or "open_ended").
        model_size: Model size ("7b" or "14b" for qwen, "8b" for llama).
        model_family: Model family ("llama" or "qwen").
        overwrite: Whether to overwrite existing results.
        custom_vector_path: Optional custom vector path. Can contain {layer} and/or
                          {behavior} placeholders. Results saved with the
                          "results_single_" prefix.
        results_root: Directory the results tree is written under. A relative path is
                          taken relative to the repository root, so the caller need
                          not cd anywhere.
        run_tag: Optional label inserted into the result filename prefix. Use it to
                          keep the runs of an ablation apart; without it filenames are
                          unchanged.
    """
    results_dir = os.path.join(str(REPO_ROOT / results_root), behavior)
    os.makedirs(results_dir, exist_ok=True)

    # Load model and tokenizer
    model_name = get_model_path(model_size, model_family)
    model, tokenizer = get_model_and_tokenizer(model_name)
    # Load test data
    test_data = load_test_data(behavior, data_type)

    process_methods = {
        "ab": process_item_ab,
        "open_ended": process_item_open_ended,
    }
    if data_type not in process_methods:
        raise ValueError(f"Unsupported data type: {data_type}. Supported: {list(process_methods.keys())}")
    process_method = process_methods[data_type]

    for layer in layers:
        if custom_vector_path:
            print(f"Loading custom steering vector for layer {layer} from {custom_vector_path}...")
        else:
            print(f"Loading steering vector for layer {layer}...")

        try:
            vector = get_steering_vector(behavior, layer, model_name, normalized=True, custom_vector_path=custom_vector_path)
            if model_size != "7b":
                model_dtype = _get_model_dtype(model_name)
                vector = vector.to(model_dtype)
            vector = vector.to(model.device)
        except Exception as e:
            raise RuntimeError(
                f"Cannot load the steering vector for {behavior}, layer {layer}, {model_name}"
            ) from e

        # Build SteeringVector once per layer
        steering_vector = SteeringVector(
            layer_activations={layer: vector},
            layer_type="decoder_block",
        )

        for multiplier in multipliers:
            result_save_suffix = f"{behavior}_{data_type}_{layer}_{multiplier}_{model_size}_{model_family}"
            filename_prefix = "results_single_" if custom_vector_path else "results_"
            if run_tag:
                filename_prefix = f"{filename_prefix}{run_tag}_"
            save_filename = os.path.join(
                results_dir, f"{filename_prefix}{result_save_suffix}.json"
            )

            if os.path.exists(save_filename) and not overwrite:
                print(f"Found existing {save_filename} - skipping")
                continue

            print(f"Testing layer {layer}, multiplier {multiplier}")
            results = []

            if data_type == "open_ended" and batch_size > 1:
                # Batched open-ended generation
                for batch_start in tqdm(
                    range(0, len(test_data), batch_size),
                    desc=f"Layer {layer}, multiplier {multiplier}",
                ):
                    batch = test_data[batch_start : batch_start + batch_size]
                    with steering_vector.apply(model, multiplier=multiplier, min_token_index=0):
                        batch_results = process_batch_open_ended(
                            items=batch,
                            model=model,
                            tokenizer=tokenizer,
                        )
                    results.extend(batch_results)
            else:
                # Single-item processing (ab always uses this path)
                for item in tqdm(test_data, desc=f"Layer {layer}, multiplier {multiplier}"):
                    # Section 3.1 applies steering at every token position.
                    min_idx = 0

                    with steering_vector.apply(model, multiplier=multiplier, min_token_index=min_idx):
                        result = process_method(
                            item=item,
                            model=model,
                            tokenizer=tokenizer,
                        )
                    results.append(result)

            # Save results
            with open(save_filename, "w") as f:
                json.dump(results, f, indent=4)

            print(f"Saved {len(results)} results to {save_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    parser.add_argument(
        "--behaviors",
        type=str,
        nargs="+",
        default=DEFAULT_BEHAVIORS,
        choices=PAPER_BEHAVIORS,
        help=(
            "Behaviours to sweep. Defaults to the three carrying the main results; "
            "the three Appendix B.7 behaviours must be named explicitly."
        ),
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["ab", "open_ended"],
        default="ab",
    )
    parser.add_argument("--model_size", type=str, choices=["7b", "8b", "14b"], default="7b")
    parser.add_argument("--model_family", type=str, choices=["llama", "qwen"], default="llama")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument(
        "--custom_vector",
        type=str,
        default=None,
        help=(
            "Path to a custom steering vector (e.g. a CAST vector). A relative path is "
            "taken relative to the repository root. The path may contain the placeholders "
            "'{layer}' and '{behavior}', which are substituted with the current layer and "
            "behaviour so that one invocation can sweep several of them. When provided, "
            "results are saved with the 'results_single_' prefix."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for open-ended generation. Default is 1 (no batching). "
             "Increase for faster processing if GPU memory allows. Has no effect for --type ab.",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="results",
        help=(
            "Directory the results tree is written under; a relative path is taken "
            "relative to the repository root. Give each ablation its own root "
            "(e.g. results_dim_2, results_no_frr) instead of relying on the vector path."
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
    # no debug flags

    args = parser.parse_args()

    for behavior in args.behaviors:
        test_steering(
            layers=args.layers,
            multipliers=args.multipliers,
            behavior=behavior,
            data_type=args.type,
            model_size=args.model_size,
            model_family=args.model_family,
            overwrite=args.overwrite,
            custom_vector_path=args.custom_vector,
            batch_size=args.batch_size,
            results_root=args.results_root,
            run_tag=args.run_tag,
        )
