# Data Card

## Summary

This repository provides fixed inputs for the main experiments in
[Could Inference-Time Interventions Preserve Alignment? Safety Cost of Steering
Vectors Is Separable and Reducible](https://arxiv.org/abs/2608.08383).
The package supports CAST training and safety and behavior evaluation for
Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, and Qwen2.5-14B-Instruct.

The package contains 26 files in [`data/paper/inputs.zip`](data/paper/inputs.zip).
It includes input prompts and dataset splits. Model weights, learned vectors,
and generated responses are produced or obtained separately.

## Access and versioning

Restore the inputs from the repository root:

```bash
python scripts/00_download_data.py
```

Restoration is offline. The archive and individual files are verified against
[`data/paper/manifest.json`](data/paper/manifest.json). Each manifest entry records
the destination path, record count, SHA-256 hash, source identity, and processing
applied to the packaged file. Existing matching files are reused; `--force`
restores the packaged version of a modified input.

## Behavior datasets

The main behavior datasets are derived from
[Anthropic's model-written evaluations](https://github.com/anthropics/evals),
using `advanced-ai-risk/lm_generated_evals/<behavior>.jsonl`.

| Behavior | Vector generation | A/B evaluation | Open-ended evaluation |
| --- | ---: | ---: | ---: |
| `corrigible-more-HHH` | 900 | 100 | 100 |
| `power-seeking-inclination` | 900 | 100 | 100 |
| `self-awareness-good-text-model` | 800 | 200 | 100 |

Each behavior directory contains:

| File | Contents | Use |
| --- | --- | --- |
| `generate_dataset.json` | Contrastive questions and matching/non-matching answers | CAA extraction and CAST effect preservation |
| `test_dataset_ab.json` | A/B evaluation questions | MCQ behavior matching |
| `test_dataset_open_ended.json` | Questions without answer options | Open-ended behavior evaluation and training monitoring |

**Processing.** Generation-provenance markers (`<META_START>…<META_END>`) are
removed and answer fields are trimmed. Corrigibility and power-seeking use a
seed-42 shuffle followed by a 900/100 split. Self-awareness uses an 800/200 split
in source order, with trailing whitespace removed from questions. Open-ended
sets use the first 100 A/B evaluation questions with their options removed.

**Split characteristics.** The preserved self-awareness split contains two exact
records shared between vector generation and A/B evaluation; one also appears
in the open-ended set. A/B and open-ended evaluations share questions by design.
The input package retains these splits for consistency with the source files.

Source: Perez et al., *Discovering Language Model Behaviors with Model-Written
Evaluations* (2022). These derived inputs retain the upstream **CC-BY-4.0** terms;
the processing above constitutes the statement of changes.

## Safety training inputs

CAST and DIM use separate prompt collections:

| Collection | Path | Records | Main fields |
| --- | --- | ---: | --- |
| CAST harmful training | `data/train/harmful_train.json` | 1,844 | `query` |
| DIM harmful training | `data/saladbench_splits/harmful_train.json` | 1,184 | `instruction`, `source` |
| DIM harmful validation | `data/saladbench_splits/harmful_val.json` | 128 | `instruction`, `source` |
| DIM harmless training | `data/saladbench_splits/harmless_train.json` | 25,058 | `instruction`, `category` |
| DIM harmless validation | `data/saladbench_splits/harmless_val.json` | 3,132 | `instruction`, `category` |

The CAST training pool combines harmful prompts from
[AlphaSteer](https://github.com/AlphaLab-USTC/AlphaSteer) and the non-AdvBench
portion of SALAD-Bench, with deduplication and JailbreakBench exclusions.
The DIM harmful splits derive from
[SALAD-Bench](https://github.com/OpenSafetyLab/SALAD-BENCH); its harmless splits
derive from [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca).
The DIM pipeline selects a matched harmless prefix before refusal filtering.
Packaged splits preserve the recorded rows and order.

## Benign and attack evaluation inputs

| Input | Records | Purpose |
| --- | ---: | --- |
| OR-Bench hard-1k | 1,000 | CAST false-refusal constraint during training |
| XSTest safe subset | 250 | False-refusal evaluation |
| Alpaca subset | 100 | Benign-response evaluation |
| JailbreakBench | 100 | Attack evaluation |
| AutoDAN prompts | 100 | Shared attack inputs |
| PAIR prompts | 100 per model | Target-model-specific attack inputs |
| GCG suffixes | 100 per model | Target-model-specific attack inputs |

The OR-Bench subset uses the first 1,000 rows of the recorded source CSV, in
order, following the training-set size specified in Appendix A.1. Its source and
subset hashes are included in the manifest. The Alpaca subset retains input
fields only. JailbreakBench retains the recorded goals, targets, and row order;
the DIM pipeline receives a JSON copy of the same evaluation prompts.

PAIR files retain `original_prompt`, `target`, `attack_prompt`, and `data_id`.
Generated responses and optimization logs are excluded from the package.
AutoDAN and GCG files retain their recorded inputs. The `none`, `prefix`,
`refusal_suppression`, and `aim` scenarios are implemented as prompt templates
in `run_steering.py`.

Sources:
[XSTest](https://github.com/paul-rottger/xstest),
[OR-Bench](https://huggingface.co/datasets/bench-llm/or-bench),
[JailbreakBench](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors),
[AutoDAN](https://github.com/SheltonLiu-N/AutoDAN),
[PAIR](https://github.com/patrickrchao/JailbreakingLLMs), and
[nanoGCG](https://github.com/GraySwanAI/nanoGCG).

## Evaluation protocol

- **Attack success:** mean continuous StrongREJECT score in [0, 1], multiplied by
  100. Responses are truncated to 512 characters before scoring. The binary
  score-threshold label is an auxiliary output.
- **False refusals:** GPT labels full compliance, partial refusal, and full
  refusal as 0, 0.5, and 1. The rate is the mean of valid scores; unclassified
  responses are reported separately.
- **Behavioral effect:** MCQ behavior matching and a separate 0–10 GPT score on
  open-ended completions, using the behavior-specific rubrics in
  `evaluate_results.py`.

Both GPT-based evaluators default to `gpt-4o` with temperature 0. These scores
are automated measurements and depend on the judge model and rubric.

## Generated artifacts

| Location | Contents |
| --- | --- |
| `artifacts/caa_vectors_raw/`, `artifacts/caa_vectors/` | Raw and normalized CAA vectors |
| `artifacts/DIM/` | DIM directions and selection metadata |
| `artifacts/cast_vectors/` | Learned CAST bases and sanitized steering vectors |
| `jailbreak_results/` | Responses to attack prompts |
| `benign_results/` | Responses to benign prompts |
| `results/` | MCQ and open-ended behavior outputs |
| `utility/` | General-capability evaluation results |

Judge scores are written to sibling `<behavior>_<judge>_evaluation/` directories.
Generated artifacts and extracted input files are gitignored; the fixed archive
and manifest are versioned.

## Intended use and terms

These inputs support research on the safety and behavioral effects of
inference-time interventions. Attack prompts and resulting completions may
contain harmful content. The evaluations measure behavior on the selected
benchmarks; they do not establish a general safety guarantee.

Dataset licenses apply independently of the repository's code license:
Anthropic evaluations, XSTest, and OR-Bench use **CC-BY-4.0**; Alpaca-derived
inputs use **CC-BY-NC-4.0**; SALAD-Bench uses **Apache-2.0**. AlphaSteer-derived
inputs retain their upstream dataset terms. Additional attribution and terms for
attack inputs are listed in [THIRD_PARTY.md](THIRD_PARTY.md).

Model weights are downloaded separately under their respective model licenses.
See [LICENSE](LICENSE) for the repository's code license scope and exceptions.
