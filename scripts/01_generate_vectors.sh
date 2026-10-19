#!/bin/bash
# Stage 1 -- extract raw CAA steering vectors from the contrastive A/B pairs.
#
# Reads:   data/behaviors/<behavior>/generate_dataset.json
# Writes:  artifacts/caa_vectors_raw/<behavior>/vec_layer_<L>_<model>.pt
# Feeds:   stage 2 (normalisation), and through it every table and figure.
#
# One vector per layer is extracted, not just the injection layer: stage 2
# normalises layer by layer, and Figure 3's layer sweep needs the whole stack.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh

banner "Stage 1: raw CAA vectors for ${BEHAVIORS}"

python generate_vectors.py \
    --model_family "$MODEL_FAMILY" \
    --model_size "$MODEL_SIZE" \
    --behaviors $BEHAVIORS \
    --batch_size "$BATCH_SIZE"

echo "Raw vectors are in artifacts/caa_vectors_raw/. Next: scripts/02_normalize_vectors.sh"
