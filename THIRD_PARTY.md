# Third-Party Notices

This document records upstream code, evaluation resources, and datasets used by
CAST. The repository's code license and file-specific exceptions are defined in
[LICENSE](LICENSE). Third-party components, datasets, and model weights retain
their respective upstream terms.

## Vendored code

### DIM refusal-direction pipeline

- **Source:** [andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction)
- **Reference:** Arditi et al., *Refusal in Language Models Is Mediated by a Single
  Direction*, NeurIPS 2024.
- **Upstream revision:** `9d852fae1a9121c78b29142de733cb1340770cc3`
- **Location:** `third_party/refusal_direction/`
- **License:** Apache-2.0; the license text is retained in
  [third_party/refusal_direction/LICENSE](third_party/refusal_direction/LICENSE).

This pipeline extracts DIM directions and selection metadata used by the CAST
training workflow and the fixed-direction baseline. Local adaptations include:

- Qwen2/2.5 support and the Qwen system prompt.
- Recorded training/validation splits, matched harmful/harmless sampling, and
  StrongREJECT evaluation.
- Adaptive generation batch sizes and model-cache configuration.
- Repository-relative data and artifact paths, configurable output directories,
  and a model-argument launcher.
- Compact serialization of the selected direction and model dispatch for the
  Llama-3 and Qwen families.

The file-level modification record is provided in
[UPSTREAM.md](third_party/refusal_direction/UPSTREAM.md). Changed source files
retain modification notices, and upstream license text is preserved.

### IFEval evaluator

- **Source:** [google-research/instruction_following_eval](https://github.com/google-research/google-research/tree/master/instruction_following_eval)
- **Reference:** Zhou et al., *Instruction-Following Evaluation for Large Language
  Models*, 2023.
- **Location:** `third_party/instruction_following_eval/`
- **License:** Apache-2.0; see the retained
  [LICENSE](third_party/instruction_following_eval/LICENSE) and
  [NOTICE](third_party/instruction_following_eval/NOTICE).

The evaluator is used by `run_utility.py`. Three sibling imports were made
relative for package integration, and an empty `__init__.py` was added.
Evaluation logic and upstream copyright headers are preserved.

## Adapted training code

### RDO

- **Source:** [wollschlager/geometry-of-refusal](https://github.com/wollschlager/geometry-of-refusal)
- **Reference:** Wollschläger et al., *The Geometry of Refusal in Large Language
  Models: Concept Cones and Representational Independence*, arXiv:2502.17420.
- **Affected files:** `train_cast.py` and `cast/projection.py`.

The training infrastructure in `train_cast.py` and the `projection_einops`
helper in `cast/projection.py` are adapted from this codebase. CAST introduces
its safety objective, effect and false-refusal constraints, primal–dual updates,
and steering-vector sanitization.

**License status:** no explicit license was identified in the upstream repository
at revision `43a86288f0e364ecb950ab88d6268f082a5fb8d1`. Permission for the
upstream-derived portions remains to be clarified. These two files are excluded
from the repository's MIT grant as specified in [LICENSE](LICENSE); this notice
does not grant additional rights to upstream code.

## Research libraries and evaluation rubrics

These libraries are installed through the requirement files and retain their
upstream licenses.

| Component | Upstream | License | Use |
| --- | --- | --- | --- |
| steering-vectors | [steering-vectors/steering-vectors](https://github.com/steering-vectors/steering-vectors) | MIT | CAA extraction and activation steering; version 0.12.2 |
| StrongREJECT | [dsbowen/strong_reject](https://github.com/dsbowen/strong_reject) | MIT | Fine-tuned safety scoring |
| nanoGCG | [GraySwanAI/nanoGCG](https://github.com/GraySwanAI/nanoGCG) | MIT | Optional GCG suffix generation |

CAA follows Rimsky et al., *Steering Llama 2 via Contrastive Activation Addition*
(ACL 2024), with the reference implementation at
[nrimsky/CAA](https://github.com/nrimsky/CAA) (MIT).
The behavior-scoring rubrics for `corrigible-more-HHH`, `myopic-reward`, and
`sycophancy` in `evaluate_results.py` reproduce the rater prompts from that work.
The power-seeking and self-awareness rubrics are provided by the CAST paper.

StrongREJECT follows Souly et al., *A StrongREJECT for Empty Jailbreaks*
(NeurIPS 2024). GCG follows Zou et al., *Universal and Transferable Adversarial
Attacks on Aligned Language Models* (2023).

## Packaged datasets and attack inputs

Fixed main-experiment inputs are included in
[`data/paper/inputs.zip`](data/paper/inputs.zip). The
[manifest](data/paper/manifest.json) records source identities, hashes, record
counts, and transformations. Detailed split definitions and processing are given
in [DATA_CARD.md](DATA_CARD.md).

| Resource | Upstream | Terms | Use |
| --- | --- | --- | --- |
| Anthropic model-written evaluations | [anthropics/evals](https://github.com/anthropics/evals) | CC-BY-4.0 | Three main behavior datasets |
| AlphaSteer prompt files | [AlphaLab-USTC/AlphaSteer](https://github.com/AlphaLab-USTC/AlphaSteer) | Repository: Apache-2.0; underlying dataset terms also apply | Source of part of the CAST harmful training pool |
| SALAD-Bench | [OpenSafetyLab/SALAD-BENCH](https://github.com/OpenSafetyLab/SALAD-BENCH) | Apache-2.0 | Harmful prompts for CAST and DIM |
| Alpaca | [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) | Data: CC-BY-NC-4.0 | DIM harmless splits and benign evaluation inputs |
| JailbreakBench | [JailbreakBench/JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) | MIT | Attack evaluation prompts and training exclusions |
| XSTest | [paul-rottger/xstest](https://github.com/paul-rottger/xstest) | CC-BY-4.0 | Safe prompts for false-refusal evaluation |
| OR-Bench | [bench-llm/or-bench](https://huggingface.co/datasets/bench-llm/or-bench) | Data: CC-BY-4.0 | Benign training prompts for the false-refusal constraint |
| AutoDAN | [SheltonLiu-N/AutoDAN](https://github.com/SheltonLiu-N/AutoDAN) | Upstream repository: MIT | Recorded attack inputs shared across models |
| PAIR | [patrickrchao/JailbreakingLLMs](https://github.com/patrickrchao/JailbreakingLLMs) | Upstream repository: MIT | Recorded attack inputs per target model |
| GCG | [GraySwanAI/nanoGCG](https://github.com/GraySwanAI/nanoGCG) | Upstream implementation: MIT | Recorded suffixes per target model |

**Anthropic evaluations — statement of changes.** Inputs derive from
`advanced-ai-risk/lm_generated_evals/<behavior>.jsonl`. Generation-provenance
markers are removed, answer fields are trimmed, and records are partitioned into
vector-generation and evaluation sets. Corrigibility and power-seeking use a
seed-42 shuffle and a 900/100 split; self-awareness uses an 800/200 split in source
order. Open-ended inputs are derived from A/B evaluation questions. See Perez
et al., *Discovering Language Model Behaviors with Model-Written Evaluations*
(2022), and the [Data Card](DATA_CARD.md) for split characteristics.

**Other processing.** The CAST harmful pool is deduplicated and excludes
JailbreakBench goals. The OR-Bench package uses the first 1,000 recorded source
rows. PAIR and Alpaca evaluation files retain input fields and exclude generated
responses. Remaining file-specific transformations are recorded in the manifest.

The Alpaca data license is separate from the Apache-2.0 license of the
[Stanford Alpaca code](https://github.com/tatsu-lab/stanford_alpaca).
Similarly, OR-Bench dataset terms are separate from its code repository's license.

## Optional runtime datasets

The downloader and utility runner also support the following sources, obtained
separately from the fixed input package:

| Resource | Upstream | Terms | Use |
| --- | --- | --- | --- |
| CAA behavior datasets | [nrimsky/CAA](https://github.com/nrimsky/CAA) | MIT | Additional `myopic-reward` and `sycophancy` inputs |
| AdvBench | [llm-attacks/llm-attacks](https://github.com/llm-attacks/llm-attacks) | MIT | Prompt source and optional attack evaluation |
| HarmBench | [centerforaisafety/HarmBench](https://github.com/centerforaisafety/HarmBench) | MIT | DIM data exclusions and optional attack evaluation |
| StrongREJECT prompt set | [alexandrasouly/strongreject](https://github.com/alexandrasouly/strongreject) | MIT | DIM data exclusions |
| MaliciousInstruct | [Princeton-SysML/Jailbreak_LLM](https://github.com/Princeton-SysML/Jailbreak_LLM) | No explicit license identified in the source repository | Optional source download |
| TDC 2023 Red Teaming | [centerforaisafety/tdc2023-starter-kit](https://github.com/centerforaisafety/tdc2023-starter-kit) | MIT | Optional source download |
| MMLU | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | MIT | General-capability evaluation |
| TriviaQA | [trivia_qa](https://huggingface.co/datasets/mandarjoshi/trivia_qa) | Data: [source terms](https://nlp.cs.washington.edu/triviaqa/); code: Apache-2.0 | General-capability evaluation |
| GSM8K | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | Dataset card: MIT | General-capability evaluation |
| IFEval | [google/IFEval](https://huggingface.co/datasets/google/IFEval) | Apache-2.0 | Instruction-following evaluation |

Runtime downloads retain the terms of the selected upstream dataset and revision.

## Models and external services

| Model or service | Terms | Use |
| --- | --- | --- |
| [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Llama 3.1 Community License | Target model |
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Apache-2.0 | Target model |
| [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) | Apache-2.0 | Target model |
| StrongREJECT fine-tuned judge | Model and base-model terms | Local safety scoring |
| OpenAI API (`gpt-4o`) | Applicable API terms | Behavior and false-refusal scoring |

Model weights are obtained separately and are not included in the fixed input
archive. The requirement files list the software dependencies used to load and
evaluate them.
