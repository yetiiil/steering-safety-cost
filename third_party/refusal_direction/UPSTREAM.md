# refusal_direction (vendored)

This directory is a partial, modified copy of the DIM ("difference-in-means")
refusal-direction pipeline released with

> Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery,
> Wes Gurnee, Neel Nanda. *Refusal in Language Models Is Mediated by a Single
> Direction.* NeurIPS 2024.

* Upstream: <https://github.com/andyrdt/refusal_direction>
* Upstream revision this copy tracks: `9d852fae1a9121c78b29142de733cb1340770cc3` (2024-10-01, `main`)
* Licence: **Apache-2.0** — the upstream repository declares Apache-2.0 and ships
  the stock licence text with no copyright-holder line filled in; `LICENSE` here is
  byte-identical to upstream's at the revision above (verified against a fresh copy
  of that revision). Upstream ships no `NOTICE` file, so section 4(d) has nothing to
  propagate.

CAST uses this pipeline for one thing: producing the refusal direction
`artifacts/DIM/<model>/direction.pt` (plus its `direction_metadata.json`, which
carries the selected layer and token position). Everything downstream —
`train_cast.py`, `ablate_refusal_direction.py` — reads those two files.

Run it with:

```bash
bash third_party/refusal_direction/bash_script/run.sh <hf-model-id>
```

## What was copied

```
dataset/__init__.py
dataset/load_dataset.py
pipeline/__init__.py
pipeline/config.py
pipeline/run_pipeline.py
pipeline/model_utils/{__init__,model_base,model_factory,llama3_model,qwen_model}.py
pipeline/submodules/{generate_directions,select_direction,evaluate_jailbreak,evaluate_loss}.py
pipeline/utils/{__init__,hook_utils,utils}.py
```

`bash_script/run.sh` is deliberately absent from that list: upstream has no
`bash_script/` directory (it ships a top-level `setup.sh` instead). The script
here is this repository's own launcher and contains no upstream code.

## What was deliberately not copied

| Path | Reason |
| --- | --- |
| `dataset/raw/`, `dataset/processed/`, `dataset/splits/` | ~8 MB of prompt data. This repository ships code only, and `processed/alpaca.json` is CC-BY-NC-4.0, which cannot be redistributed inside an Apache-2.0 tree. Re-fetch them from upstream if you need the evaluation sets. |
| `pipeline/model_utils/*_model.py` for families other than Llama-3 and Qwen | Upstream registers wrappers for several model families. Only the Llama-3 and Qwen ones are vendored, because the paper extracts refusal directions for three models and they are all covered by those two. `model_factory.py` is pruned to match (see B), so any other model path raises `ValueError` rather than importing a wrapper that is not here. |
| `pipeline/run_rdo_pipeline.py`, `pipeline/run_rdo_samples.py` | Evaluators for a different paper's method; they resolve their vectors out of a private Weights & Biases project (`WANDB_ENTITY`). |
| `logs/`, `wandb/`, `__pycache__/` | Run output. |

## Local modifications

Two layers of modification are recorded separately: changes that were already in
the working copy when the paper's numbers were produced, and changes made while
vendoring the code into this repository.

### Which files carry a modification notice

Section 4(b) of the Apache Licence requires a changed file to state that it was
changed. Every file whose bytes differ from upstream `9d852fa` carries this notice
at the top, above the imports; the files that do not differ carry nothing, so that
the presence of the notice is itself the record of what was touched.

```python
# Modified from the original: https://github.com/andyrdt/refusal_direction (Apache-2.0).
# See third_party/refusal_direction/UPSTREAM.md for the full list of changes.
```

| File | Notice | State relative to upstream `9d852fa` |
| --- | --- | --- |
| `dataset/load_dataset.py` | yes | modified (A and B below) |
| `pipeline/config.py` | yes | modified (A and B) |
| `pipeline/run_pipeline.py` | yes | modified (A and B) |
| `pipeline/model_utils/model_base.py` | yes | modified (A) |
| `pipeline/model_utils/llama3_model.py` | yes | modified (A) |
| `pipeline/model_utils/qwen_model.py` | yes | modified (A) |
| `pipeline/model_utils/model_factory.py` | yes | modified (B) |
| `pipeline/submodules/evaluate_jailbreak.py` | yes | modified (A) |
| `pipeline/submodules/select_direction.py` | no | byte-identical (restored to upstream, see B) |
| `pipeline/submodules/{generate_directions,evaluate_loss}.py` | no | byte-identical |
| `pipeline/utils/{hook_utils,utils}.py` | no | byte-identical |
| `dataset/__init__.py`, `pipeline/__init__.py`, `pipeline/model_utils/__init__.py`, `pipeline/utils/__init__.py` | no | byte-identical (all empty) |
| `bash_script/run.sh` | no | not an upstream file (see above) |

The eight annotated files are exactly the ones that differ. Seven of them were
established by diffing this tree against a fresh copy of revision `9d852fa` rather
than inferred from the list below; `model_factory.py` matched upstream until it was
pruned here (B).

### A. Pre-existing, kept as-is (these are what the paper ran)

* `pipeline/config.py`
  * `n_test` 100 → 128; upstream's `n_train`/`n_val` fields dropped (see the
    sampling change in `run_pipeline.py` below).
  * `splits = "saladbench"` records that the harmful/harmless splits are the
    SALAD-Bench derived ones, not upstream's `dataset/splits/`.
  * `jailbreak_eval_methodologies` is `("substring_matching", "strongreject")`
    instead of upstream's `("substring_matching", "llamaguard2")`: the
    StrongREJECT fine-tuned judge replaced the LlamaGuard-2 API judge.
  * `completions_batch_size = 128` added and passed through to
    `generate_completions`.
  * `sample`, `subspace_n_samples` and the four `evaluate_*` switches are read
    only by the RDO runners that were not vendored; they are inert here.
* `pipeline/run_pipeline.py`
  * Train/val prompts are no longer subsampled with `random.sample(..., n_train)`;
    the full harmful split is used and the harmless split is truncated to the
    same length (before and after refusal-score filtering).
  * Progress printing in `filter_data`.
  * The three `evaluate_loss_for_datasets` calls are commented out.
* `pipeline/model_utils/model_base.py`
  * `generate_completions` gained a `temperature` argument (`do_sample=temperature > 0`;
    the default `temperature=0` reproduces upstream's `do_sample=False`) and an
    `accelerate.utils.find_executable_batch_size` probe that picks the largest
    batch that fits, caching it in `self.max_batch_size`. The requested
    `batch_size` argument is consequently only used as a starting point.
* `pipeline/model_utils/qwen_model.py`
  * Retargeted from Qwen-1 to Qwen-2/2.5: `_get_model_block_modules` and
    `_get_attn_modules` read `model.model.layers` / `block.self_attn` instead of
    `model.transformer.h` / `block.attn`, plus plain `AutoModelForCausalLM`
    loading in bfloat16, no `use_fast=False`, no `<|extra_0|>` pad token.
  * **Known defect, left alone:** the retarget stopped at the class. The
    module-level `orthogonalize_qwen_weights` and `act_add_qwen_weights` still
    address `model.transformer.wte` / `model.transformer.h` / `c_proj`, which a
    Qwen-2/2.5 checkpoint does not have, and would raise `AttributeError`.
    Nothing reaches them: `run_pipeline.py` ablates through the hooks in
    `hook_utils.py`, and `_get_orthogonalization_mod_fn` / `_get_act_add_mod_fn`
    are called nowhere in this tree — every wrapper defines them only because
    `ModelBase` declares them abstract.
  * The Qwen system prompt ("You are Qwen, created by Alibaba Cloud. You are a
    helpful assistant.") is now passed to the tokenisation function; upstream
    passed `system=None`. This changes the prompt the activations are read from,
    so it is load-bearing for the Qwen directions in the paper.
* `pipeline/model_utils/{llama3,qwen}_model.py`
  * `cache_dir=os.getenv("HUGGINGFACE_CACHE_DIR")` on `from_pretrained`
    (`None` when unset, i.e. the default HF cache).
* `pipeline/model_utils/llama3_model.py`
  * **Known defect, left alone:** in `LLAMA3_CHAT_TEMPLATE_WITH_SYSTEM` the
    placeholder was changed to `{{instruction}}`, so `.format()` emits a literal
    `{instruction}` instead of the prompt. `Llama3Model` always tokenises with
    `system=None`, so this branch is never taken by the pipeline; it is recorded
    rather than patched because patching it would change what a future
    system-prompt run produces.
* `pipeline/submodules/evaluate_jailbreak.py`
  * The LlamaGuard-2 (`litellm` + `TOGETHER_API_KEY`) and HarmBench (`vllm`)
    judges were deleted and replaced by the StrongREJECT fine-tuned evaluator
    (`strong_reject.evaluate.evaluate_dataset`) plus an XSTest refusal-prefix
    heuristic, with a `wandb.summary` mirror guarded by `if wandb.run is not None`.
  * **Known hazard, left alone:** the `"llamaguard2"` and `"harmbench"` branches
    still call the deleted judge functions and would raise `NameError`. Neither
    is in the shipped `jailbreak_eval_methodologies`.
* `dataset/load_dataset.py`
  * `strongreject` is loaded from the `strong_reject` package rather than from
    `dataset/processed/`; `xstest` and `sorrybench` added to
    `PROCESSED_DATASET_NAMES`.

### B. Changes made while vendoring (this repository)

* `pipeline/config.py` — `artifact_path()` resolved the repository root through
  a hard-coded `"../../"`, which pointed one level too high once the tree moved
  under `third_party/`, and dereferenced `os.getenv("SAVE_DIR")` unconditionally
  (a `TypeError` when it was unset). It now resolves the repo root from the file
  location and mirrors `cast.paths.dim_direction_dir`: `SAVE_DIR/DIM_DIR` when
  `SAVE_DIR` is set, otherwise `<repo>/artifacts/DIM/<model_alias>`.
* `dataset/load_dataset.py` — same one-level path correction for the splits
  (`<repo>/data/saladbench_splits/`), and both loaders now raise a
  `FileNotFoundError` naming the missing file and pointing at the README instead
  of failing inside `open()`.
* `pipeline/run_pipeline.py`
  * **Fixed:** `torch.save(direction, ...)` saved a *view* into the candidate
    grid, and `torch.save` serialises the whole storage behind a view — a single
    `d_model` vector was written as a 5–10 MB file. It is now
    `torch.save(direction.detach().clone().contiguous(), ...)`. Value-identical;
    only the file size changes.
  * Removed `from dotenv import load_dotenv` and the `load_dotenv("..", override=True)`
    call. `".."` is a directory, so `python-dotenv` never read anything from it;
    dropping it removes a dependency and the invitation to keep credentials in a
    `.env` file next to the repository. Credentials for this repository come from
    the environment.
* `pipeline/model_utils/model_factory.py` — `construct_model_base` now
  dispatches only to the two wrappers this tree ships. The branches for the other
  families upstream registers were removed, so an unrecognised model path raises
  the function's own `ValueError` instead of a `ModuleNotFoundError` from an
  import that cannot resolve. The `'qwen'` and `'llama-3'` tests, and the order
  they are tried in, are untouched, so all three of the paper's models dispatch
  exactly as before.
* `pipeline/submodules/select_direction.py` — the vendored file is now
  **byte-identical to upstream**. Removed were `plot_density_plot`,
  `select_rdo_direction`, `sample_prob_vectors`, `sample_hypersphere_vectors`
  and `select_cone_basis` (~510 lines) together with the `wandb` and `seaborn`
  imports they needed. Those functions implement a different paper's direction
  search, are called only from the two RDO runners that were not vendored, and
  are unreachable from `run_pipeline.py`. `select_direction`, `filter_fn`,
  `plot_refusal_scores`, `refusal_score`, `get_refusal_scores`,
  `get_last_position_logits`, `masked_mean` and `kl_div_fn` — everything the
  pipeline actually calls — are untouched.
* `bash_script/run.sh` — rewritten. Not an upstream file: what was rewritten is
  the launcher used for the paper's own runs, every line of which set a
  machine-specific value (`CUDA_VISIBLE_DEVICES`, a local `HF_HOME`, a hard-coded
  model id). It now takes the model id as an argument, `cd`s to the root of this
  tree so the `pipeline.*` / `dataset.*` imports resolve, and leaves `SAVE_DIR`
  and `DIM_DIR` to the environment.

## Expected inputs

* `<repo>/data/saladbench_splits/{harmful,harmless}_{train,val,test}.json` —
  prompt splits, `[{"instruction": ..., "category": ...}, ...]`.
* `third_party/refusal_direction/dataset/processed/<name>.json` — only for the
  jailbreak evaluation stage (`jailbreakbench` by default); fetch from upstream.

## Python packages this tree needs

`torch`, `transformers`, `accelerate`, `einops`, `jaxtyping`, `matplotlib`,
`tqdm`, `numpy` and `datasets`, plus `strong_reject` and `wandb`, which
`evaluate_jailbreak.py` imports. `wandb` only has to be importable — the pipeline
never calls `wandb.init`, and the one W&B write is guarded by
`if wandb.run is not None`. `evaluate_loss.py` is imported but never called (its
call sites are commented out upstream of it in `run_pipeline.py`).
