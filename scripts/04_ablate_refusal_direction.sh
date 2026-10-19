#!/bin/bash
# Stage 4 -- the "Refusal Direction Ablation" baseline.
#
# Reads:   artifacts/caa_vectors/, artifacts/DIM/<model_id>/direction.pt
# Writes:  artifacts/refusal_ablated_vectors/<behavior>/..._direction.pt
# Feeds:   the baseline row of Table 1 and the third bar group of Figure 6.
#
# This is the comparison CAST is measured against: the same projection of
# Eq. (1), but with a direction found once from harmful/harmless activations
# instead of one optimised against the vector it is removed from.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh

banner "Stage 4: refusal-direction ablation baseline"

python ablate_refusal_direction.py \
    --model "$MODEL_ID" \
    --ablation_type direction

echo "Baseline vectors are in artifacts/refusal_ablated_vectors/."
echo "run_steering.py --use_ablated_vectors picks them up."
