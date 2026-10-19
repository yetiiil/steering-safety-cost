#!/bin/bash
# Stage 9 -- score the completions from stages 6-8.
#
# Reads:   ${JAILBREAK_RESULTS}/, ${BENIGN_RESULTS}/, ${OPEN_ENDED_RESULTS}/
# Writes:  <behavior>_<evaluator>_evaluation/ beside each input directory
# Feeds:   every number in Tables 1-3, 5, 9, 12-16 and Figures 1, 2, 6.
#
# Three judges, one per result tree:
#   strongreject_finetuned  attack success; a local fine-tuned classifier
#   gpt_false_refusal       over-refusal on benign prompts; OpenAI API
#   gpt_behavior_scorer     0-10 behaviour score on open-ended text; OpenAI API
#
# Both GPT judges run on gpt-4o, the model the paper's numbers were produced
# with; it is passed explicitly rather than left to a default, because both
# scores are judge-model dependent.  Override with GPT_JUDGE_MODEL for
# exploratory runs, knowing they will not match the paper.
#
# The two GPT judges cost real money and need OPENAI_API_KEY in the environment
# -- there is deliberately no flag to pass a key on the command line.  Scoring is
# resumable: a file whose *_evaluation.json already exists is skipped unless you
# re-score it, which for the GPT judges means paying for it again:
#   scripts/09_judge.sh --overwrite
# Any other argument is forwarded to evaluate_results.py the same way, as is
# EXTRA_JUDGE_ARGS for callers that can only set the environment.
#
# EVALUATORS selects which of the three to run, e.g.
#   EVALUATORS=strongreject_finetuned scripts/09_judge.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh

EVALUATORS="${EVALUATORS:-strongreject_finetuned gpt_false_refusal gpt_behavior_scorer}"
EXTRA_JUDGE_ARGS="${EXTRA_JUDGE_ARGS:-}"

for evaluator in $EVALUATORS; do
    case "$evaluator" in
        strongreject_finetuned) base_dir="$JAILBREAK_RESULTS" ;;
        gpt_false_refusal)      base_dir="$BENIGN_RESULTS" ;;
        gpt_behavior_scorer)    base_dir="$OPEN_ENDED_RESULTS" ;;
        *) echo "$0: unknown evaluator '$evaluator'" >&2; exit 1 ;;
    esac

    case "$evaluator" in
        gpt_*) require_env OPENAI_API_KEY ;;
    esac

    if [ ! -d "$base_dir" ]; then
        echo "Skipping ${evaluator}: no ${base_dir}/ -- run the matching stage first."
        continue
    fi

    banner "Stage 9: ${evaluator} over ${base_dir}"
    python evaluate_results.py \
        --evaluator "$evaluator" \
        --base-dir "$base_dir" \
        --behaviors $BEHAVIORS \
        --model-family "$MODEL_FAMILY" \
        --model-type "$MODEL_SIZE" \
        --gpt-model "$GPT_JUDGE_MODEL" \
        --batch-size "$JUDGE_BATCH_SIZE" \
        $EXTRA_JUDGE_ARGS "$@"
done

echo "Scored. Each *_evaluation/ directory holds one *_evaluation.json per input"
echo "file, with the per-prompt verdicts; the per-file summary is printed above."
echo "Aggregating those into the paper's tables -- means over attacks, deltas"
echo "against the multiplier-0 row -- is not part of this release."
