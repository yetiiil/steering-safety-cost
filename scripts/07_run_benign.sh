#!/bin/bash
# Stage 7 -- generate completions on benign prompts, to measure over-refusal.
#
# Reads:   artifacts/caa_vectors/, artifacts/cast_vectors/, data/benign/
# Writes:  ${BENIGN_RESULTS}/<family>/<size>/<behavior>/*.json
# Feeds:   the false-refusal-rate columns of Table 3 and Tables 5, 9, 15.
#
# No attack template is applied here: the question is whether steering makes the
# model refuse things it should answer.  The paper reports two prompt sets,
# XSTest-safe and Alpaca.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh

RUN_BASELINES="${RUN_BASELINES:-1}"
RUN_CAST="${RUN_CAST:-1}"

benign() {
    local source="$1"; shift
    python run_benign.py \
        --layers "$LAYER" \
        --behaviors $BEHAVIORS \
        --model_family "$MODEL_FAMILY" \
        --model_size "$MODEL_SIZE" \
        --benign_source "$source" \
        --batch_size "$BATCH_SIZE" \
        --results_root "$BENIGN_RESULTS" \
        "$@"
}

for source in $BENIGN_SOURCES; do
    if [ "$RUN_BASELINES" = "1" ]; then
        banner "Stage 7: CAA on ${source}"
        benign "$source" --multipliers $EFFECT_MULTIPLIERS
    fi
    if [ "$RUN_CAST" = "1" ]; then
        banner "Stage 7: CAST on ${source}"
        benign "$source" --multipliers $CAST_EFFECT_MULTIPLIERS --custom_vector_path "$CAST_VECTOR"
    fi
done

echo "Completions are in ${BENIGN_RESULTS}/. Score them with scripts/09_judge.sh."
