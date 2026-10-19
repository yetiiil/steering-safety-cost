"""General-capability benchmarks under steering (Table 11).

Runs MMLU, TriviaQA, GSM8K and IFEval with a CAA or CAST steering vector applied,
and once without steering as the unsteered reference row.

This module monkeypatches the steering hook of the `steering-vectors` package (see
`_cache_aware_additive_hook` below), so it is coupled to that package's internals:
the repository pins `steering-vectors==0.12.2`, and the patch must be re-checked
against `steering_vectors.steering_vector` before that pin is moved.
"""

import argparse
import os
import json
import re
import string
import random
from typing import List
import numpy as np
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering_vectors import SteeringVector
from run_open_ended import (
    get_model_and_tokenizer,
    get_model_path,
    _compute_min_token_index_for_prompt,
    get_steering_vector,
)

from cast.paths import REPO_ROOT
# steering_settings.py is the single source of truth for the behaviour surface.
from steering_settings import PAPER_BEHAVIORS

# ---------- Cache-aware steering hook (monkeypatch) ----------
# The stock steering_vectors hook builds its patch mask from the *current forward's*
# sequence length: `mask = zeros(seq_len); mask[slice(min_token_index, None)] = 1`.
# With the KV cache, decode steps feed a length-1 tensor (only the new token), so a
# slice like [min_token_index:] (min_token_index = input_len-1) is empty -> the mask
# is all-zeros and steering is silently NOT applied to generated tokens. Disabling
# the cache fixes correctness but is ~50-100x slower for long generations (GSM8K).
#
# Fix: when seq_len <= min_token_index we are in a cached continuation past the
# instruction, so every token in this forward is an answer token -> steer all of them.
# This keeps the cache on and is correct for both min_token_index=0 (all positions)
# and after_instruction. token_indices mask tensors are left untouched.
import steering_vectors.steering_vector as _sv_mod
from steering_vectors.torch_utils import untuple_tensor as _untuple_tensor


def _cache_aware_additive_hook(target_activation, token_indices, operator=None):
    def hook_fn(_m, _inputs, outputs):
        original_tensor = _untuple_tensor(outputs)
        target_act = target_activation.to(original_tensor.device)
        delta = target_act if operator is None else operator(original_tensor, target_act)
        seq_len = original_tensor.shape[1]
        if isinstance(token_indices, torch.Tensor):
            mask = token_indices
        elif (
            isinstance(token_indices, slice)
            and token_indices.start
            and seq_len <= token_indices.start
        ):
            # cached decode: all tokens here are past min_token_index -> steer all
            mask = torch.ones(seq_len)
        else:
            mask = torch.zeros(seq_len)
            mask[token_indices] = 1
        mask = (
            mask.reshape(1, -1, 1)
            if len(mask.shape) == 1
            else mask.reshape(mask.shape[0], -1, 1)
        )
        mask = mask.to(original_tensor.device)
        original_tensor[None] = original_tensor + (mask * delta)
        return outputs

    return hook_fn


_sv_mod._create_additive_hook = _cache_aware_additive_hook

# ---------- Determinism ----------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------- MMLU Evaluation ----------

def evaluate_mmlu(
    model,
    tokenizer,
    dataset,
    steering_vector=None,
    multiplier: float = 1.0,
    layer: int = None,
    system_prompt: str = None,
    model_family: str = "qwen",
    max_new_tokens: int = 0,
    debug: bool = False,
    batch_size: int = 1,
):
    model.eval()
    correct = 0
    total = 0

    choice_labels = ["A", "B", "C", "D"]
    choice_token_ids = [
        tokenizer.encode(" " + label, add_special_tokens=False)[-1]
        for label in choice_labels
    ]

    if batch_size <= 1:
        for i, ex in enumerate(tqdm(dataset, desc="MMLU eval")):
            question = ex["question"]
            choices = ex["choices"]
            gold = ex["answer"]

            prompt = f"Question: {question}\n"
            labels = choice_labels[:len(choices)]
            for j, choice in enumerate(choices):
                prompt += f"{labels[j]}. {choice}\n"
            prompt += "Answer:"

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                if steering_vector:
                    min_idx = _compute_min_token_index_for_prompt(
                        tokenizer,
                        model.config._name_or_path,
                        model_family,
                        question,
                        system_prompt,
                    )
                    with steering_vector.apply(
                        model, multiplier=multiplier, min_token_index=min_idx
                    ):
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)

            last_logits = outputs.logits[:, -1, :]
            log_probs = torch.nn.functional.log_softmax(last_logits, dim=-1)
            loglikelihoods = [log_probs[0, choice_token_ids[j]].item() for j in range(len(choices))]

            pred = int(np.argmax(loglikelihoods))
            if pred == gold:
                correct += 1
            total += 1

            if debug:
                print("\n[MMLU Debug]")
                print("Q:", question)
                for j, choice in enumerate(choices):
                    print(f" {labels[j]}. {choice}  (logp={loglikelihoods[j]:.2f})")
                print("Gold:", labels[gold], "Pred:", labels[pred])
    else:
        padding_side_original = tokenizer.padding_side
        tokenizer.padding_side = "left"

        for batch_start in tqdm(range(0, len(dataset), batch_size), desc="MMLU eval"):
            batch_end = min(batch_start + batch_size, len(dataset))
            batch = [dataset[i] for i in range(batch_start, batch_end)]

            prompts = []
            golds = []
            for ex in batch:
                question = ex["question"]
                choices = ex["choices"]
                labels = choice_labels[:len(choices)]
                prompt = f"Question: {question}\n"
                for j, choice in enumerate(choices):
                    prompt += f"{labels[j]}. {choice}\n"
                prompt += "Answer:"
                prompts.append(prompt)
                golds.append(ex["answer"])

            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

            with torch.no_grad():
                if steering_vector:
                    with steering_vector.apply(model, multiplier=multiplier, min_token_index=0):
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)

            last_logits = outputs.logits[:, -1, :]
            log_probs = torch.nn.functional.log_softmax(last_logits, dim=-1)

            for i, ex in enumerate(batch):
                num_choices = len(ex["choices"])
                loglikelihoods = [log_probs[i, choice_token_ids[j]].item() for j in range(num_choices)]
                pred = int(np.argmax(loglikelihoods))
                if pred == golds[i]:
                    correct += 1
                total += 1

                if debug:
                    labels = choice_labels[:num_choices]
                    print("\n[MMLU Debug]")
                    print("Q:", ex["question"])
                    for j, choice in enumerate(ex["choices"]):
                        print(f" {labels[j]}. {choice}  (logp={loglikelihoods[j]:.2f})")
                    print("Gold:", labels[golds[i]], "Pred:", labels[pred])

        tokenizer.padding_side = padding_side_original

    return correct / max(total, 1)

# ---------- TriviaQA Evaluation ----------

def simple_evaluation_score(prediction, ground_truth):
    """Simple evaluation: check if gold answer (lowercased) is contained in model answer (lowercased)."""
    pred_lower = prediction.lower().strip()
    gold_lower = ground_truth.lower().strip()
    return gold_lower in pred_lower

def evaluate_triviaqa(
    model,
    tokenizer,
    dataset,
    steering_vector=None,
    multiplier: float = 1.0,
    layer: int = None,
    model_family: str = "qwen",
    system_prompt: str = None,
    debug: bool = False,
    batch_size: int = 1,
    steering_position: str = "after_instruction",
    max_new_tokens: int = 32,
):
    """Evaluate TriviaQA using the model's chat template.

    Prompts are built with create_chat_prompt() so instruct/chat models see
    their native turn structure (Qwen <|im_start|>..., Llama-3
    <|start_header_id|>..., etc.) instead of a raw `Question:/Answer:` base-
    model continuation. The assistant turn is opened by the template; we decode
    everything generated after `input_len` as the model's answer.

    Steering convention matches IFEval / GSM8K:
      * steering_position='after_instruction' -> min_token_index = input_len-1
        (the left-padded assistant-header position).
      * steering_position='all'              -> min_token_index = 0.
    The cache-aware additive hook installed at import time keeps the KV cache on
    in both modes (no more use_cache=False workaround).
    """
    from tokenizer import create_chat_prompt

    model.eval()
    if steering_position not in {"all", "after_instruction"}:
        raise ValueError(f"Unknown steering_position: {steering_position}")

    examples = [dataset[i] for i in range(len(dataset))]
    chat_prompts = [
        create_chat_prompt(
            tokenizer=tokenizer,
            question=f"Question: {ex['question']}\nAnswer:",
            answer=None,
            use_chat_format=True,
        )
        for ex in examples
    ]

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    padding_side_original = tokenizer.padding_side
    tokenizer.padding_side = "left"  # required for batched generation

    correct_total = 0
    total = 0
    responses = []

    effective_batch_size = max(batch_size, 1)
    for batch_start in tqdm(
        range(0, len(examples), effective_batch_size), desc="Evaluating TriviaQA"
    ):
        batch_end = min(batch_start + effective_batch_size, len(examples))
        batch = examples[batch_start:batch_end]
        batch_chat = chat_prompts[batch_start:batch_end]

        enc = tokenizer(
            batch_chat,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,  # template already inserts BOS / specials
        ).to(model.device)
        input_len = enc.input_ids.shape[1]

        steering_min_idx = None
        if steering_vector is not None:
            steering_min_idx = (
                input_len - 1 if steering_position == "after_instruction" else 0
            )

        with torch.no_grad():
            if steering_vector is not None:
                with steering_vector.apply(
                    model,
                    multiplier=multiplier,
                    min_token_index=steering_min_idx,
                ):
                    outputs = model.generate(**enc, **gen_kwargs)
            else:
                outputs = model.generate(**enc, **gen_kwargs)

        for i, ex in enumerate(batch):
            question = ex["question"]
            gold_answer = ex["answer"]["value"]
            pred_full = tokenizer.decode(
                outputs[i, input_len:], skip_special_tokens=True
            ).strip()
            pred_text = pred_full.split("\n")[0]

            is_correct = bool(gold_answer) and simple_evaluation_score(
                pred_text, gold_answer
            )
            correct_total += int(is_correct)
            total += 1

            responses.append({
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": pred_text,
                "predicted_answer_full": pred_full,
                "is_correct": is_correct,
                "prompt": batch_chat[i],
                "steering_position": steering_position if steering_vector else None,
                "steering_min_token_index": steering_min_idx,
            })

            if debug:
                print("\n[TriviaQA Debug]")
                print("Q:", question)
                print("Gold:", gold_answer)
                print("Pred:", pred_text)
                print("Correct:", is_correct)

    tokenizer.padding_side = padding_side_original

    return {"accuracy": 100.0 * correct_total / max(total, 1)}, responses

# ---------- IFEval Evaluation ----------

def _details_path_for(save_path, tag=None):
    root, _ext = os.path.splitext(save_path)
    suffix = f"_{tag}_details.jsonl" if tag else "_details.jsonl"
    return f"{root}{suffix}"


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def save_detail_rows(details_path, rows):
    if not details_path or not rows:
        return
    detail_dir = os.path.dirname(details_path)
    if detail_dir:
        os.makedirs(detail_dir, exist_ok=True)
    with open(details_path, "a") as f:
        for row in rows:
            f.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")


def _strip_none_kwargs(kwargs_list):
    """The HF `google/IFEval` dataset pads every example's kwargs with the union
    of all instruction args, filling unused ones with None. The Google verifiers'
    build_description(**kwargs) chokes on those Nones, so drop None-valued keys
    (same fix lm-eval-harness applies)."""
    return [{k: v for k, v in kw.items() if v is not None} for kw in kwargs_list]

def evaluate_ifeval(
    model,
    tokenizer,
    dataset,
    steering_vector=None,
    multiplier: float = 1.0,
    layer: int = None,
    model_family: str = "qwen",
    max_new_tokens: int = 1024,
    batch_size: int = 1,
    debug: bool = False,
    steering_position: str = "all",
):
    """IFEval (Zhou et al. 2023): instruction-following with deterministic verifiers.

    Generates a free-form response per prompt (chat-templated, multi-line preserved)
    and scores it with Google's verifier registry. Returns the four standard numbers:
    prompt-level and instruction-level accuracy, each strict and loose.

    By default, steering is applied across all positions (min_token_index=0),
    matching the generation convention used by run_benign.py / evaluate_triviaqa.
    With steering_position="after_instruction", steering starts at the assistant
    header position. In a left-padded batch this is input_len - 1 for every row.
    """
    # Imported lazily so the heavy verifier deps (nltk/langdetect/...) are only
    # needed when IFEval is actually requested.
    from third_party.instruction_following_eval import evaluation_lib
    from tokenizer import create_chat_prompt

    model.eval()

    if steering_position not in {"all", "after_instruction"}:
        raise ValueError(f"Unknown steering_position: {steering_position}")

    inputs = []
    raw_prompts = []
    chat_prompts = []
    for ex in dataset:
        inputs.append(
            evaluation_lib.InputExample(
                key=ex["key"],
                instruction_id_list=list(ex["instruction_id_list"]),
                prompt=ex["prompt"],
                kwargs=_strip_none_kwargs(ex["kwargs"]),
            )
        )
        raw_prompts.append(ex["prompt"])
        chat_prompts.append(
            create_chat_prompt(
                tokenizer=tokenizer,
                question=ex["prompt"],
                answer=None,
                use_chat_format=True,
            )
        )

    prompt_to_response = {}
    prompt_to_steering_min_idx = {}
    padding_side_original = tokenizer.padding_side
    tokenizer.padding_side = "left"  # required for batched generation

    def _run_generation():
        for batch_start in tqdm(range(0, len(chat_prompts), batch_size), desc="IFEval gen"):
            batch_end = min(batch_start + batch_size, len(chat_prompts))
            batch_chat = chat_prompts[batch_start:batch_end]
            batch_raw = raw_prompts[batch_start:batch_end]

            # create_chat_prompt already inserts the model's special tokens, so do
            # not let the tokenizer add them again.
            enc = tokenizer(
                batch_chat,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(model.device)
            input_len = enc.input_ids.shape[1]

            steering_min_idx = None
            if steering_vector is not None:
                steering_min_idx = input_len - 1 if steering_position == "after_instruction" else 0

            with torch.no_grad():
                if steering_vector is not None:
                    with steering_vector.apply(
                        model,
                        multiplier=multiplier,
                        min_token_index=steering_min_idx,
                    ):
                        out = model.generate(
                            **enc,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                            top_k=None,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        )
                else:
                    out = model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )

            for raw_prompt in batch_raw:
                prompt_to_steering_min_idx[raw_prompt] = steering_min_idx

            for i in range(len(batch_chat)):
                # Keep the full multi-line response (unlike TriviaQA's first line).
                resp = tokenizer.decode(
                    out[i, input_len:], skip_special_tokens=True
                ).strip()
                prompt_to_response[batch_raw[i]] = resp
                if debug:
                    print("\n[IFEval Debug]")
                    print("Prompt:", batch_raw[i][:120])
                    print("Response:", resp[:200])

    _run_generation()

    tokenizer.padding_side = padding_side_original

    # ---- Score with the verifiers (no LLM judge needed) ----
    n_prompt = 0
    prompt_strict = prompt_loose = 0
    inst_total = 0
    inst_strict = inst_loose = 0
    details = []
    for inp in inputs:
        s = evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
        l = evaluation_lib.test_instruction_following_loose(inp, prompt_to_response)
        response = prompt_to_response[inp.prompt]
        strict_list = [bool(x) for x in s.follow_instruction_list]
        loose_list = [bool(x) for x in l.follow_instruction_list]
        n_prompt += 1
        prompt_strict += int(s.follow_all_instructions)
        prompt_loose += int(l.follow_all_instructions)
        inst_total += len(strict_list)
        inst_strict += sum(strict_list)
        inst_loose += sum(loose_list)
        details.append({
            "key": inp.key,
            "prompt": inp.prompt,
            "response": response,
            "instruction_id_list": inp.instruction_id_list,
            "kwargs": inp.kwargs,
            "strict_follow_all": bool(s.follow_all_instructions),
            "loose_follow_all": bool(l.follow_all_instructions),
            "strict_follow_instruction_list": strict_list,
            "loose_follow_instruction_list": loose_list,
            "strict_failed_instruction_ids": [
                instruction_id
                for instruction_id, ok in zip(inp.instruction_id_list, strict_list)
                if not ok
            ],
            "steering_position": steering_position,
            "steering_min_token_index": prompt_to_steering_min_idx.get(inp.prompt),
            "loose_failed_instruction_ids": [
                instruction_id
                for instruction_id, ok in zip(inp.instruction_id_list, loose_list)
                if not ok
            ],
            "response_num_chars": len(response),
            "response_num_words": len(response.split()),
        })

    metrics = {
        "ifeval_prompt_strict_acc": prompt_strict / max(n_prompt, 1),
        "ifeval_prompt_loose_acc": prompt_loose / max(n_prompt, 1),
        "ifeval_inst_strict_acc": inst_strict / max(inst_total, 1),
        "ifeval_inst_loose_acc": inst_loose / max(inst_total, 1),
    }
    return metrics, details

# ---------- GSM8K Evaluation ----------

_GSM8K_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")

# Canonical 8-shot chain-of-thought exemplars for GSM8K (Wei et al. 2022,
# "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"). This is
# the fixed, hand-written exemplar set most GSM8K few-shot numbers are reported
# against. Answers end with "The answer is <number>.", which _gsm8k_extract_pred
# parses; --gsm8k_num_shots N uses the first N of these.
_GSM8K_COT_EXEMPLARS = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove "
     "today. After they are done, there will be 21 trees. How many trees did the "
     "grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were "
     "planted. So there must have been 21 - 15 = 6. The answer is 6."),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars "
     "are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces "
     "do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had "
     "32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 "
     "lollipops. How many lollipops did Jason give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
     "So he gave Denny 20 - 12 = 8. The answer is 8."),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
     "How many toys does he have now?",
     "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then "
     "that is 4 more toys. 5 + 4 = 9. The answer is 9."),
    ("There were nine computers in the server room. Five more computers were "
     "installed each day, from monday to thursday. How many computers are now in "
     "the server room?",
     "There were originally 9 computers. For each of 4 days, 5 more computers were "
     "added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. The answer is 29."),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, "
     "he lost 2 more. How many golf balls did he have at the end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he had "
     "58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer "
     "is 33."),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she "
     "have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
     "So she has 23 - 15 = 8 dollars left. The answer is 8."),
]


def _build_gsm8k_fewshot_prefix(num_shots):
    """Few-shot block from the canonical Wei et al. CoT exemplars (first num_shots).

    There are 8 exemplars; num_shots is clamped to that.
    """
    if num_shots <= 0:
        return ""
    shots = _GSM8K_COT_EXEMPLARS[:num_shots]
    blocks = [f"Q: {q}\nA: {a}" for q, a in shots]
    return "\n\n".join(blocks) + "\n\n"


def _gsm8k_user_message(question, fewshot_prefix):
    if fewshot_prefix:
        # Faithful Wei et al. format: Q/A demonstrations, then the target question.
        return f"{fewshot_prefix}Q: {question}\nA:"
    # Zero-shot fallback: explicit instruction so the answer is still parseable.
    return (
        "Solve the following grade-school math problem. Reason step by step, then "
        f"end with 'The answer is <number>.'\n\nQ: {question}\nA:"
    )


def _gsm8k_normalize_number(num_str):
    if num_str is None:
        return None
    cleaned = num_str.replace("$", "").replace(",", "").rstrip(".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _gsm8k_gold_answer(answer_text):
    """GSM8K gold rationales end with '#### <number>'."""
    tail = answer_text.split("####")[-1]
    nums = _GSM8K_NUMBER_RE.findall(tail)
    return _gsm8k_normalize_number(nums[-1]) if nums else None


def _gsm8k_extract_pred(response):
    """Prefer the last 'The answer is N' (the Wei-et-al. exemplar format), then a
    '#### N' marker, else fall back to the last number in the response
    (lm-eval-harness flexible-extract)."""
    ms = re.findall(
        r"answer is\s*:?\s*(-?\$?\d[\d,]*(?:\.\d+)?)", response, re.IGNORECASE
    )
    if ms:
        return _gsm8k_normalize_number(ms[-1])
    if "####" in response:
        nums = _GSM8K_NUMBER_RE.findall(response.split("####")[-1])
        if nums:
            return _gsm8k_normalize_number(nums[0])
    nums = _GSM8K_NUMBER_RE.findall(response)
    return _gsm8k_normalize_number(nums[-1]) if nums else None


def evaluate_gsm8k(
    model,
    tokenizer,
    dataset,
    steering_vector=None,
    multiplier: float = 1.0,
    layer: int = None,
    model_family: str = "qwen",
    max_new_tokens: int = 512,
    batch_size: int = 1,
    debug: bool = False,
    steering_position: str = "all",
    num_shots: int = 0,
):
    """GSM8K (Cobbe et al. 2021): grade-school math via chat-templated CoT.

    With num_shots>0, few-shot exemplars are prepended to the chat turn. Generates a
    step-by-step solution, extracts the final number, and scores exact numeric match
    against the gold answer. Steering convention matches evaluate_ifeval:
    'all' => min_token_index=0; 'after_instruction' => the left-padded header position.
    """
    from tokenizer import create_chat_prompt

    model.eval()
    if steering_position not in {"all", "after_instruction"}:
        raise ValueError(f"Unknown steering_position: {steering_position}")

    fewshot_prefix = _build_gsm8k_fewshot_prefix(num_shots)

    questions = []
    golds = []
    chat_prompts = []
    for ex in dataset:
        questions.append(ex["question"])
        golds.append(_gsm8k_gold_answer(ex["answer"]))
        chat_prompts.append(
            create_chat_prompt(
                tokenizer=tokenizer,
                question=_gsm8k_user_message(ex["question"], fewshot_prefix),
                answer=None,
                use_chat_format=True,
            )
        )

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    # Steering correctly with the KV cache (incl. after_instruction) relies on the
    # cache-aware steering hook installed at import (see _cache_aware_additive_hook).

    padding_side_original = tokenizer.padding_side
    tokenizer.padding_side = "left"  # required for batched generation

    correct = 0
    total = 0
    details = []
    for batch_start in tqdm(range(0, len(chat_prompts), batch_size), desc="GSM8K gen"):
        batch_end = min(batch_start + batch_size, len(chat_prompts))
        batch_chat = chat_prompts[batch_start:batch_end]

        # create_chat_prompt already inserts special tokens; don't double them.
        enc = tokenizer(
            batch_chat, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        input_len = enc.input_ids.shape[1]

        steering_min_idx = None
        if steering_vector is not None:
            steering_min_idx = input_len - 1 if steering_position == "after_instruction" else 0

        with torch.no_grad():
            if steering_vector is not None:
                with steering_vector.apply(
                    model, multiplier=multiplier, min_token_index=steering_min_idx
                ):
                    out = model.generate(**enc, **gen_kwargs)
            else:
                out = model.generate(**enc, **gen_kwargs)

        for i in range(len(batch_chat)):
            idx = batch_start + i
            resp = tokenizer.decode(
                out[i, input_len:], skip_special_tokens=True
            ).strip()
            pred = _gsm8k_extract_pred(resp)
            gold = golds[idx]
            is_correct = (
                pred is not None and gold is not None and abs(pred - gold) < 1e-4
            )
            correct += int(is_correct)
            total += 1
            details.append({
                "question": questions[idx],
                "response": resp,
                "pred": pred,
                "gold": gold,
                "is_correct": is_correct,
                "steering_position": steering_position,
                "steering_min_token_index": steering_min_idx,
            })
            if debug:
                print("\n[GSM8K Debug]")
                print("Q:", questions[idx][:120])
                print("Pred:", pred, "| Gold:", gold, "| Correct:", is_correct)

    tokenizer.padding_side = padding_side_original

    metrics = {"gsm8k_acc": correct / max(total, 1)}
    return metrics, details

# ---------- Main Script ----------

def main(args):
    set_seed(args.seed)

    # A relative --save_path is taken relative to the repository root, so results
    # land in the same place no matter where the script was launched from.
    if args.save_path:
        args.save_path = str(REPO_ROOT / args.save_path)

    detail_path = _details_path_for(args.save_path) if args.save_path and "ifeval" in args.benchmarks else None
    gsm8k_detail_path = _details_path_for(args.save_path, "gsm8k") if args.save_path and "gsm8k" in args.benchmarks else None
    triviaqa_detail_path = _details_path_for(args.save_path, "triviaqa") if args.save_path and "triviaqa" in args.benchmarks else None
    for _p in (detail_path, gsm8k_detail_path, triviaqa_detail_path):
        if _p and os.path.exists(_p):
            os.remove(_p)

    # get_model_path(size, model_family) from run_open_ended; Instruct models only,
    # matching the other evaluation scripts.
    model_name = get_model_path(args.model_size, args.model_family)
    model, tokenizer = get_model_and_tokenizer(model_name)

    results = {}
    if args.layer == None and args.behaviors == None and args.multipliers == None:
        bench_results = {}
        if "mmlu" in args.benchmarks:
            ds = load_dataset("cais/mmlu", "all")["test"]
            ds = ds.select(range(min(args.num_instances, len(ds))))
            acc = evaluate_mmlu(
                model,
                tokenizer,
                ds,
                model_family=args.model_family,
                debug=args.debug,
                batch_size=args.batch_size,
            )
            bench_results["mmlu_acc"] = acc
            print(f"MMLU acc: {acc:.4f}")

        if "triviaqa" in args.benchmarks:
            ds = load_dataset("trivia_qa", "rc.wikipedia")["validation"]
            ds = ds.select(range(min(args.num_instances, len(ds))))
            metrics, responses = evaluate_triviaqa(
                model,
                tokenizer,
                ds,
                model_family=args.model_family,
                debug=args.debug,
                batch_size=args.batch_size,
                steering_position=args.triviaqa_steering_position,
            )
            save_detail_rows(
                triviaqa_detail_path,
                [{"run": "base_0.0", **row} for row in responses],
            )
            bench_results["triviaqa_accuracy"] = metrics["accuracy"]
            print(f"TriviaQA Accuracy: {metrics['accuracy']:.2f}")

        if "ifeval" in args.benchmarks:
            ds = load_dataset("google/IFEval")["train"]
            ds = ds.select(range(min(args.num_instances, len(ds))))
            metrics, details = evaluate_ifeval(
                model,
                tokenizer,
                ds,
                model_family=args.model_family,
                max_new_tokens=args.ifeval_max_new_tokens,
                batch_size=args.batch_size,
                debug=args.debug,
                steering_position=args.ifeval_steering_position,
            )
            save_detail_rows(
                detail_path,
                [{"run": "base_0.0", **row} for row in details],
            )
            bench_results.update(metrics)
            print(f"IFEval: {metrics}")

        if "gsm8k" in args.benchmarks:
            ds = load_dataset("gsm8k", "main")["test"]
            ds = ds.select(range(min(args.num_instances, len(ds))))
            metrics, details = evaluate_gsm8k(
                model,
                tokenizer,
                ds,
                model_family=args.model_family,
                max_new_tokens=args.gsm8k_max_new_tokens,
                batch_size=args.batch_size,
                debug=args.debug,
                steering_position=args.gsm8k_steering_position,
                num_shots=args.gsm8k_num_shots,
            )
            save_detail_rows(
                gsm8k_detail_path,
                [{"run": "base_0.0", **row} for row in details],
            )
            bench_results.update(metrics)
            print(f"GSM8K acc: {metrics['gsm8k_acc']:.4f}")

        results[("base", 0.0)] = bench_results
    else:
        for behavior in args.behaviors:
            for multiplier in args.multipliers:
                print(f"\n=== Running with behavior={behavior}, multiplier={multiplier} ===")

                # Optional system prompt flag is currently a no-op in this script.
                system_prompt = None

                steering_vector = None
                if args.layer is not None:
                    # Allow either precomputed vectors by behavior/layer or a single custom vector path.
                    # If custom_vector_path is provided, it is passed through to get_steering_vector,
                    # which supports optional `{layer}` substitution in the path.
                    vec = get_steering_vector(
                        behavior,
                        args.layer,
                        model_name,
                        normalized=True,
                        custom_vector_path=args.custom_vector_path,
                    ).to(model.device)
                    steering_vector = SteeringVector({args.layer: vec}, layer_type="decoder_block")

                bench_results = {}

                if "mmlu" in args.benchmarks:
                    ds = load_dataset("cais/mmlu", "all")["test"]
                    ds = ds.select(range(min(args.num_instances, len(ds))))
                    acc = evaluate_mmlu(
                        model,
                        tokenizer,
                        ds,
                        steering_vector=steering_vector,
                        multiplier=multiplier,
                        layer=args.layer,
                        system_prompt=system_prompt,
                        model_family=args.model_family,
                        debug=args.debug,
                        batch_size=args.batch_size,
                    )
                    bench_results["mmlu_acc"] = acc
                    print(f"MMLU acc: {acc:.4f}")

                if "triviaqa" in args.benchmarks:
                    ds = load_dataset("trivia_qa", "rc.wikipedia")["validation"]
                    ds = ds.select(range(min(args.num_instances, len(ds))))
                    metrics, responses = evaluate_triviaqa(
                        model,
                        tokenizer,
                        ds,
                        steering_vector=steering_vector,
                        multiplier=multiplier,
                        layer=args.layer,
                        model_family=args.model_family,
                        system_prompt=system_prompt,
                        debug=args.debug,
                        batch_size=args.batch_size,
                        steering_position=args.triviaqa_steering_position,
                    )
                    save_detail_rows(
                        triviaqa_detail_path,
                        [
                            {
                                "run": f"{behavior}_{multiplier}",
                                "behavior": behavior,
                                "multiplier": multiplier,
                                **row,
                            }
                            for row in responses
                        ],
                    )
                    bench_results["triviaqa_accuracy"] = metrics["accuracy"]
                    print(f"TriviaQA Accuracy: {metrics['accuracy']:.2f}")

                if "ifeval" in args.benchmarks:
                    ds = load_dataset("google/IFEval")["train"]
                    ds = ds.select(range(min(args.num_instances, len(ds))))
                    metrics, details = evaluate_ifeval(
                        model,
                        tokenizer,
                        ds,
                        steering_vector=steering_vector,
                        multiplier=multiplier,
                        layer=args.layer,
                        model_family=args.model_family,
                        max_new_tokens=args.ifeval_max_new_tokens,
                        batch_size=args.batch_size,
                        debug=args.debug,
                        steering_position=args.ifeval_steering_position,
                    )
                    save_detail_rows(
                        detail_path,
                        [
                            {
                                "run": f"{behavior}_{multiplier}",
                                "behavior": behavior,
                                "multiplier": multiplier,
                                **row,
                            }
                            for row in details
                        ],
                    )
                    bench_results.update(metrics)
                    print(f"IFEval: {metrics}")

                if "gsm8k" in args.benchmarks:
                    ds = load_dataset("gsm8k", "main")["test"]
                    ds = ds.select(range(min(args.num_instances, len(ds))))
                    metrics, details = evaluate_gsm8k(
                        model,
                        tokenizer,
                        ds,
                        steering_vector=steering_vector,
                        multiplier=multiplier,
                        layer=args.layer,
                        model_family=args.model_family,
                        max_new_tokens=args.gsm8k_max_new_tokens,
                        batch_size=args.batch_size,
                        debug=args.debug,
                        steering_position=args.gsm8k_steering_position,
                        num_shots=args.gsm8k_num_shots,
                    )
                    save_detail_rows(
                        gsm8k_detail_path,
                        [
                            {
                                "run": f"{behavior}_{multiplier}",
                                "behavior": behavior,
                                "multiplier": multiplier,
                                **row,
                            }
                            for row in details
                        ],
                    )
                    bench_results.update(metrics)
                    print(f"GSM8K acc: {metrics['gsm8k_acc']:.4f}")

                results[(behavior, multiplier)] = bench_results

    print("\nFinal Results:")
    for (behavior, multiplier), res in results.items():
        print(f"Behavior={behavior}, Multiplier={multiplier} -> {res}")

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        with open(args.save_path, "w") as f:
            json.dump({f"{b}_{m}": r for (b, m), r in results.items()}, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_family", type=str, choices=["llama", "qwen"], default="qwen")
    parser.add_argument("--model_size", type=str, choices=["7b", "8b", "14b"], default="7b")

    parser.add_argument("--benchmarks", nargs="+", choices=["mmlu", "triviaqa", "ifeval", "gsm8k"], required=True)
    parser.add_argument(
        "--triviaqa_steering_position",
        type=str,
        choices=["all", "after_instruction"],
        default="after_instruction",
        help=(
            "Where to apply TriviaQA steering. 'all' uses min_token_index=0; "
            "'after_instruction' starts at the final prompt token and disables "
            "KV cache so generated answer tokens are steered too."
        ),
    )
    parser.add_argument(
        "--gsm8k_max_new_tokens",
        type=int,
        default=512,
        help="Max generated tokens per GSM8K problem (room for step-by-step reasoning).",
    )
    parser.add_argument(
        "--gsm8k_num_shots",
        type=int,
        default=0,
        help=(
            "Number of canonical Wei et al. CoT exemplars to prepend (0 = zero-shot; "
            "max 8). Use 4 for 4-shot."
        ),
    )
    parser.add_argument(
        "--gsm8k_steering_position",
        type=str,
        choices=["all", "after_instruction"],
        default="all",
        help=(
            "Where to apply GSM8K steering. 'all' uses min_token_index=0; "
            "'after_instruction' starts at the left-padded assistant-header position."
        ),
    )
    parser.add_argument(
        "--ifeval_max_new_tokens",
        type=int,
        default=1280,
        help="Max generated tokens per IFEval prompt (lm-eval-harness default; responses are multi-line).",
    )
    parser.add_argument(
        "--ifeval_steering_position",
        type=str,
        choices=["all", "after_instruction"],
        default="all",
        help=(
            "Where to apply IFEval steering. 'all' uses min_token_index=0; "
            "'after_instruction' starts at the left-padded assistant-header position."
        ),
    )
    parser.add_argument("--layer", type=int, default=None, help="If no layer is specified, steering vectors are not applied")
    parser.add_argument("--behaviors", nargs="+", type=str, choices=PAPER_BEHAVIORS, default=None, help="If no behavior is specified, steering vectors are not applied")
    parser.add_argument("--multipliers", nargs="+", type=float, default=None)
    parser.add_argument(
        "--custom_vector_path",
        type=str,
        default=None,
        help=(
            "Path to a custom steering vector .pt file (e.g. a CAST vector); a relative "
            "path is taken relative to the repository root. May contain the placeholders "
            "'{layer}' and '{behavior}', substituted with the layer index and the current "
            "behaviour."
        ),
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help=(
            "Where to write the results JSON; a relative path is taken relative to the "
            "repository root. Per-example rows for TriviaQA/GSM8K/IFEval are written "
            "alongside it as <save_path>_<benchmark>_details.jsonl."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_instances", type=int, default=1000, help="Number of instances to evaluate per dataset")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help=(
            "Batch size for benchmark evaluation. "
            "Supports both base and steering runs."
        ),
    )
    parser.add_argument("--debug", action="store_true", help="Print detailed evaluation logs")
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        choices=["pos", "neg"],
        required=False,
        help=(
            "Optional behavior-specific system prompt polarity "
            "(pos/neg) to match CAA-style steering configurations."
        ),
    )
    args = parser.parse_args()
    main(args)
