#!/bin/bash
# Stage 8 -- measure the behavioural effect itself, with no attack in sight.
#
# Reads:   artifacts/caa_vectors/, artifacts/cast_vectors/,
#          data/behaviors/<behavior>/test_dataset_{ab,open_ended}.json
# Writes:  ${OPEN_ENDED_RESULTS}/<behavior>/*.json
# Feeds:   Figure 2 and the effect columns of Table 3 (open_ended, scored in
#          stage 9), Figure 3 (ab), Table 10 and Table 15's length rows.
#
# --type ab is the forced two-choice probability and needs no judge; --type
# open_ended generates free text that stage 9 scores with a GPT judge.
# For Figure 3's layer sweep, ask for the whole stack:
#   AB_LAYERS=all scripts/08_run_open_ended.sh
# It is spelled `all` rather than a literal `seq` because the depth is
# per-model -- 28 layers for Qwen-7B, 32 for Llama-8B, 48 for Qwen-14B -- and a
# hardcoded sweep would cover half of Qwen-14B and stop four layers short of
# Llama-8B.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh

RUN_BASELINES="${RUN_BASELINES:-1}"
RUN_CAST="${RUN_CAST:-1}"
RUN_AB="${RUN_AB:-1}"
RUN_OPEN_ENDED="${RUN_OPEN_ENDED:-1}"
AB_LAYERS="${AB_LAYERS:-$LAYER}"
# Expanded here, not in the caller's shell: N_LAYERS comes from scripts/config.sh.
if [ "$AB_LAYERS" = "all" ]; then
    AB_LAYERS="$(seq 0 $((N_LAYERS - 1)))"
fi

open_ended() {
    python run_open_ended.py \
        --behaviors $BEHAVIORS \
        --model_family "$MODEL_FAMILY" \
        --model_size "$MODEL_SIZE" \
        --results_root "$OPEN_ENDED_RESULTS" \
        "$@"
}

if [ "$RUN_AB" = "1" ]; then
    if [ "$RUN_BASELINES" = "1" ]; then
        banner "Stage 8: A/B, CAA"
        open_ended --type ab --layers $AB_LAYERS --multipliers $EFFECT_MULTIPLIERS
    fi
    if [ "$RUN_CAST" = "1" ]; then
        banner "Stage 8: A/B, CAST"
        open_ended --type ab --layers $AB_LAYERS --multipliers $CAST_EFFECT_MULTIPLIERS \
            --custom_vector "$CAST_VECTOR"
    fi
fi

if [ "$RUN_OPEN_ENDED" = "1" ]; then
    if [ "$RUN_BASELINES" = "1" ]; then
        banner "Stage 8: open-ended, CAA"
        open_ended --type open_ended --layers "$LAYER" --multipliers $EFFECT_MULTIPLIERS \
            --batch_size "$BATCH_SIZE_OPEN_ENDED"
    fi
    if [ "$RUN_CAST" = "1" ]; then
        banner "Stage 8: open-ended, CAST"
        open_ended --type open_ended --layers "$LAYER" --multipliers $CAST_EFFECT_MULTIPLIERS \
            --batch_size "$BATCH_SIZE_OPEN_ENDED" --custom_vector "$CAST_VECTOR"
    fi
fi

echo "Completions are in ${OPEN_ENDED_RESULTS}/."
