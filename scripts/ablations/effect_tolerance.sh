#!/bin/bash
# Table 9 -- effect-tolerance sweep: loosen epsilon_e in L_effect <= epsilon_e.
#
# Reads:   artifacts/caa_vectors/, artifacts/DIM/
# Writes:  artifacts/cast_vectors/<model_id>/dim_1_effecttol_<tag>_<loss>/
#          {jailbreak,benign}_results_effecttol_<tag>/ and results_effecttol_<tag>/
# Feeds:   Table 9.
#
#   scripts/ablations/effect_tolerance.sh              # 0.001 0.006 0.01
#   scripts/ablations/effect_tolerance.sh 0.001 0.005 0.01
#
# A larger epsilon_e lets the optimiser keep more of the original steering
# effect, at the cost of safety.  The paper sweeps one behaviour,
# power-seeking-inclination, which is the default here; set BEHAVIORS to change it.
#
# The dual variable for the effect constraint was initialised differently at
# each tolerance in the published runs -- a loose constraint starts from a
# smaller multiplier.  Those values are reproduced below; an unlisted tolerance
# falls back to the model's config and says so.
set -euo pipefail
cd "$(dirname "$0")/../.."

# The published sweep is one behaviour deep; an explicit BEHAVIORS wins.
: "${BEHAVIORS:=power-seeking-inclination}"

source scripts/config.sh

TOLERANCES=("$@")
if [ ${#TOLERANCES[@]} -eq 0 ]; then
    TOLERANCES=(0.001 0.006 0.01)
fi

export MODEL_FAMILY MODEL_SIZE BEHAVIORS

# lambda_effect_init used for each published (model, tolerance) pair.
published_lambda_effect_init() {
    case "${MODEL_FAMILY}-${MODEL_SIZE}:$1" in
        qwen-7b:0.001)  echo 0.6 ;;
        qwen-7b:0.006)  echo 0.2 ;;
        qwen-7b:0.01)   echo 0.1 ;;
        qwen-14b:*)     echo 0.1 ;;
        *)              echo "" ;;
    esac
}

for eps in "${TOLERANCES[@]}"; do
    # Must match the directory tag train_cast.py --effect_tol_tag builds:
    # f"{effect_margin:.6g}" with '-' -> 'neg' and '.' -> 'p'.
    # LC_ALL=C so a locale with a comma decimal separator cannot produce a tag
    # that no longer matches the directory name Python wrote.
    tag="$(LC_ALL=C printf '%.6g' "$eps")"
    tag="${tag//-/neg}"
    tag="${tag//./p}"

    train_flags=(--effect_margin "$eps" --effect_tol_tag)
    lambda_init="$(published_lambda_effect_init "$eps")"
    if [ -n "$lambda_init" ]; then
        train_flags+=(--lambda_effect_init "$lambda_init")
    else
        echo "No published lambda_effect_init for ${MODEL_ID} at eps=${eps};"
        echo "using the value in ${CAST_CONFIG}."
    fi

    banner "Table 9: effect tolerance ${eps} (tag ${tag})"

    scripts/05_train_cast.sh "${train_flags[@]}"

    CAST_VARIANT="dim_1_effecttol_${tag}_${SAFETY_LOSS_TYPE}" \
    JAILBREAK_RESULTS="jailbreak_results_effecttol_${tag}" \
    BENIGN_RESULTS="benign_results_effecttol_${tag}" \
    OPEN_ENDED_RESULTS="results_effecttol_${tag}" \
    RUN_BASELINES=0 \
    bash -c 'scripts/06_run_steering.sh && scripts/07_run_benign.sh && scripts/08_run_open_ended.sh && scripts/09_judge.sh'
done

echo "Effect-tolerance sweep done: ${TOLERANCES[*]}"
