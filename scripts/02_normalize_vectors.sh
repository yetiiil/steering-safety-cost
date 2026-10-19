#!/bin/bash
# Stage 2 -- put every behaviour's CAA vector on the same scale.
#
# Reads:   artifacts/caa_vectors_raw/
# Writes:  artifacts/caa_vectors/<behavior>/vec_layer_<L>_<model>.pt
# Feeds:   every downstream stage; also Figure 5 / Table 8, whose cosine
#          similarities assume these vectors are the normalised ones.
#
# The target norm is read per model and per layer from
# configs/normalization_target_norms.yaml -- the scale the released vectors
# carry -- so which behaviours are normalised together no longer decides what a
# steering multiplier means.  Behaviours can therefore be normalised one at a
# time, or added later, without rescaling the ones already written.
#
# normalize_vectors.py refuses to write anything if a named behaviour is missing
# a raw vector for any layer, rather than normalising the subset it can find.
# Pass the literal `all` to rescale every behaviour that has raw vectors:
#   BEHAVIORS=all scripts/02_normalize_vectors.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh

banner "Stage 2: normalising ${BEHAVIORS}"

python normalize_vectors.py \
    --model_family "$MODEL_FAMILY" \
    --model_size "$MODEL_SIZE" \
    --n_layers "$N_LAYERS" \
    --behaviors $BEHAVIORS

echo "Normalised vectors are in artifacts/caa_vectors/."
