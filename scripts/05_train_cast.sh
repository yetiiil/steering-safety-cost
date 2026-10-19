#!/bin/bash
# Stage 5 -- learn the CAST direction and write the sanitized steering vector.
#
# Reads:   artifacts/caa_vectors/, artifacts/DIM/<model_id>/, data/train/,
#          data/benign/, data/behaviors/<behavior>/
# Writes:  artifacts/cast_vectors/<model_id>/<variant>/{final_vector_*,sanitized_*}.pt
# Feeds:   everything the paper reports for CAST.
#
# One optimisation per behaviour; the per-model epsilons, dual step size and
# lambda initialisations live in configs/<model_id>.yaml.  Extra arguments to
# this script are forwarded to train_cast.py, which is how the ablation scripts
# reuse it -- for example `scripts/05_train_cast.sh --ablate_frr`.
#
# Weights & Biases is optional: pass --no_wandb, or export WANDB_MODE=disabled.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh

if [ ! -f "$CAST_CONFIG" ]; then
    echo "$0: no config at $CAST_CONFIG" >&2
    echo "Set CAST_CONFIG, or add a YAML for this model under configs/." >&2
    exit 1
fi

for behavior in $BEHAVIORS; do
    banner "Stage 5: training CAST for ${behavior}"
    python train_cast.py \
        --config "$CAST_CONFIG" \
        --behavior "$behavior" \
        "$@"
done

echo "Vectors are in artifacts/cast_vectors/${MODEL_ID}/."
