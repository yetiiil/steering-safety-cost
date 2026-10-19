#!/bin/bash
# Stage 10 -- general capability under steering.
#
# Reads:   artifacts/caa_vectors/, artifacts/cast_vectors/ (benchmarks come
#          from the HuggingFace hub)
# Writes:  ${UTILITY_RESULTS}/<model_subdir>/{original,cleaned}_*.json
# Feeds:   Table 11.
#
# Two runs per model, both at multipliers -1/0/+1: the original CAA vector and
# the CAST vector.  The `original_` and `cleaned_` prefixes are what tell the two
# apart afterwards -- do not rename them.  Unlike stages 7 and 8, the CAST run
# here cannot borrow the CAA run's 0 row, because the two land in separate files
# and every Table 11 number is a delta against the unsteered score of the same
# run.  See UTILITY_CAST_MULTIPLIERS in scripts/config.sh.
#
# Resumable: a run whose output file already exists is skipped, so an
# interrupted sweep can simply be restarted.  A `cleaned_` file left over from a
# sweep that ran without the 0 multiplier is skipped too -- delete it to pick
# the extra row up.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh

OUT_DIR="${UTILITY_RESULTS}/${MODEL_SUBDIR}"
mkdir -p "$OUT_DIR"

# "mmlu triviaqa gsm8k ifeval" -> "mmlu-triviaqa-gsm8k-ifeval", so one filename
# names the whole run.
BENCH_TAG="$(echo "$UTILITY_BENCHMARKS" | tr ' ' '-')"

utility() {
    local save_path="$1"; shift
    if [ -f "$save_path" ]; then
        echo "[have] ${save_path}"
        return 0
    fi
    python run_utility.py \
        --model_family "$MODEL_FAMILY" \
        --model_size "$MODEL_SIZE" \
        --benchmarks $UTILITY_BENCHMARKS \
        --layer "$LAYER" \
        --behaviors $BEHAVIORS \
        --seed "$UTILITY_SEED" \
        --num_instances "$UTILITY_NUM_INSTANCES" \
        --batch_size "$BATCH_SIZE_UTILITY" \
        --save_path "$save_path" \
        "$@"
}

banner "Stage 10: ${UTILITY_BENCHMARKS} with the original CAA vector"
utility "${OUT_DIR}/original_${BENCH_TAG}_${MODEL_FAMILY}_${MODEL_SIZE}_L${LAYER}.json" \
    --multipliers $EFFECT_MULTIPLIERS

banner "Stage 10: ${UTILITY_BENCHMARKS} with the CAST vector"
utility "${OUT_DIR}/cleaned_${BENCH_TAG}_${MODEL_FAMILY}_${MODEL_SIZE}_L${LAYER}.json" \
    --multipliers $UTILITY_CAST_MULTIPLIERS \
    --custom_vector_path "$CAST_VECTOR"

echo "Results are in ${OUT_DIR}/."
