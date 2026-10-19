#!/bin/bash
# Table 5 -- constraint ablation: drop one side of the CAST objective at a time.
#
# Reads:   artifacts/caa_vectors/, artifacts/DIM/
# Writes:  artifacts/cast_vectors/<model_id>/dim_1_<variant>_<loss>/
#          {jailbreak,benign}_results_ablation_<variant>/ and results_ablation_<variant>/
# Feeds:   Table 5.
#
#   scripts/ablations/constraints.sh                  # all three variants
#   scripts/ablations/constraints.sh no_frr           # just one
#
# Variants:
#   no_frr            L_frr    dropped: safety and effect only
#   no_effect         L_effect dropped: safety and false refusal only
#   no_frr_no_effect  both dropped: the unconstrained safety loss
#
# The full objective is the default CAST run in the main result trees, and the
# m = 0 baseline is shared, so only the ablated CAST condition is generated.
# The published ablation covers one behaviour,
# self-awareness-good-text-model, which is the default; set BEHAVIORS to change it.
set -euo pipefail
cd "$(dirname "$0")/../.."

# The published ablation is one behaviour deep; an explicit BEHAVIORS wins.
: "${BEHAVIORS:=self-awareness-good-text-model}"

source scripts/config.sh

VARIANTS=("$@")
if [ ${#VARIANTS[@]} -eq 0 ]; then
    VARIANTS=(no_frr no_effect no_frr_no_effect)
fi

export MODEL_FAMILY MODEL_SIZE BEHAVIORS

for variant in "${VARIANTS[@]}"; do
    # train_cast.py names the directory after the constraints it dropped, in a
    # fixed order (effect before frr), which is not the order the result trees
    # and the paper's table use.  Both spellings are load-bearing; keep them.
    case "$variant" in
        no_frr)           train_flags=(--ablate_frr);                 vector_tag="no_frr" ;;
        no_effect)        train_flags=(--ablate_effect);              vector_tag="no_effect" ;;
        no_frr_no_effect) train_flags=(--ablate_effect --ablate_frr); vector_tag="no_effect_no_frr" ;;
        *)
            echo "$0: unknown variant '$variant'" >&2
            echo "Choose from: no_frr, no_effect, no_frr_no_effect" >&2
            exit 1
            ;;
    esac

    banner "Table 5: ${variant}"

    scripts/05_train_cast.sh "${train_flags[@]}"

    CAST_VARIANT="dim_1_${vector_tag}_${SAFETY_LOSS_TYPE}" \
    JAILBREAK_RESULTS="jailbreak_results_ablation_${variant}" \
    BENIGN_RESULTS="benign_results_ablation_${variant}" \
    OPEN_ENDED_RESULTS="results_ablation_${variant}" \
    RUN_BASELINES=0 \
    bash -c 'scripts/06_run_steering.sh && scripts/07_run_benign.sh && scripts/08_run_open_ended.sh && scripts/09_judge.sh'
done

echo "Constraint ablation done: ${VARIANTS[*]}"
