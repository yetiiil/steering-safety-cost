#!/bin/bash
# Table 4 -- rank sweep: remove an N-dimensional refusal subspace, not just the
# single direction of Eq. (1).
#
# Reads:   artifacts/caa_vectors/, artifacts/DIM/
# Writes:  artifacts/cast_vectors/<model_id>/dim_<N>_<loss>/
#          {jailbreak,benign}_results_dim_<N>/ and results_dim_<N>/
# Feeds:   Table 4.
#
#   scripts/ablations/rank.sh            # the published sweep, N = 2 4 8
#   scripts/ablations/rank.sh 2 4        # just two ranks
#
# Rank 1 is the default CAST run and lives in the main result trees, so it is
# not repeated here.  Table 4 sweeps one behaviour on Qwen-7B,
# power-seeking-inclination, which is the default; set BEHAVIORS to change it.
# Table 16 is the same sweep on the same model for length_conciseness:
#   BEHAVIORS=length_conciseness scripts/ablations/rank.sh
# Each rank gets its own result tree; the baselines are the ones already in the
# main trees, so only the CAST condition is generated.
set -euo pipefail
cd "$(dirname "$0")/../.."

# The published sweep is one behaviour deep; an explicit BEHAVIORS wins.
: "${BEHAVIORS:=power-seeking-inclination}"

source scripts/config.sh

RANKS=("$@")
if [ ${#RANKS[@]} -eq 0 ]; then
    RANKS=(2 4 8)
fi

export MODEL_FAMILY MODEL_SIZE BEHAVIORS

for n in "${RANKS[@]}"; do
    banner "Table 4: rank ${n}"

    scripts/05_train_cast.sh --rank "$n"

    CAST_VARIANT="dim_${n}_${SAFETY_LOSS_TYPE}" \
    JAILBREAK_RESULTS="jailbreak_results_dim_${n}" \
    BENIGN_RESULTS="benign_results_dim_${n}" \
    OPEN_ENDED_RESULTS="results_dim_${n}" \
    RUN_BASELINES=0 \
    bash -c 'scripts/06_run_steering.sh && scripts/07_run_benign.sh && scripts/08_run_open_ended.sh && scripts/09_judge.sh'
done

echo "Rank sweep done: ${RANKS[*]}"
