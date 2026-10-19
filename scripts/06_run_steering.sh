#!/bin/bash
# Stage 6 -- generate completions under attack, with and without steering.
#
# Reads:   artifacts/caa_vectors/, artifacts/cast_vectors/, data/attacks/
# Writes:  ${JAILBREAK_RESULTS}/<family>/<size>/<behavior>/*.json
# Feeds:   Table 1, Table 2, Figure 1, Figure 6 (after stage 9 scores them).
#
# Three conditions land in the same tree, told apart by filename prefix:
#   results_          the unsteered / CAA runs   (multipliers include 0)
#   results_ablation_ the refusal-ablated baseline from stage 4
#   results_single_   CAST                        (0 omitted: it is the same row)
#
# Set RUN_BASELINES=0 to generate only the CAST condition -- what the ablation
# scripts do, since their baseline is the shared m=0 row in the main tree.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh

RUN_BASELINES="${RUN_BASELINES:-1}"
RUN_CAST="${RUN_CAST:-1}"

# AutoDAN prompts are thousands of tokens long, so they get their own pass at a
# smaller batch size rather than dragging the whole sweep down.
steer() {
    local batch="$1"; shift
    local attacks="$1"; shift
    if [ -z "$attacks" ]; then return 0; fi
    python run_steering.py \
        --layers "$LAYER" \
        --behaviors $BEHAVIORS \
        --model_family "$MODEL_FAMILY" \
        --model_size "$MODEL_SIZE" \
        --dataset "$DATASET" \
        --attack_method $attacks \
        --batch_size "$batch" \
        --results_root "$JAILBREAK_RESULTS" \
        "$@"
}

if [ "$RUN_BASELINES" = "1" ]; then
    banner "Stage 6: unsteered and CAA, ${DATASET}"
    steer "$BATCH_SIZE"             "$ATTACKS"            --multipliers $MULTIPLIERS
    steer "$BATCH_SIZE_LONG_PROMPT" "$ATTACKS_LONG_PROMPT" --multipliers $MULTIPLIERS

    if [ -d "artifacts/refusal_ablated_vectors" ]; then
        banner "Stage 6: refusal-direction-ablation baseline"
        steer "$BATCH_SIZE"             "$ATTACKS"            --multipliers $MULTIPLIERS --use_ablated_vectors
        steer "$BATCH_SIZE_LONG_PROMPT" "$ATTACKS_LONG_PROMPT" --multipliers $MULTIPLIERS --use_ablated_vectors
    else
        echo "No artifacts/refusal_ablated_vectors/; skipping the Table 1 baseline."
        echo "Run scripts/04_ablate_refusal_direction.sh to produce it."
    fi
fi

if [ "$RUN_CAST" = "1" ]; then
    banner "Stage 6: CAST (${CAST_VARIANT})"
    steer "$BATCH_SIZE"             "$ATTACKS"            --multipliers $CAST_MULTIPLIERS --custom_vector_path "$CAST_VECTOR"
    steer "$BATCH_SIZE_LONG_PROMPT" "$ATTACKS_LONG_PROMPT" --multipliers $CAST_MULTIPLIERS --custom_vector_path "$CAST_VECTOR"
fi

echo "Completions are in ${JAILBREAK_RESULTS}/. Score them with scripts/09_judge.sh."
