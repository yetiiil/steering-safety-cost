#!/usr/bin/env bash
# Extract a DIM refusal direction (Arditi et al., 2024) for one model.
#
#   bash third_party/refusal_direction/bash_script/run.sh <hf-model-id>
#
# Writes <repo>/artifacts/DIM/<model-basename>/{direction.pt,direction_metadata.json},
# which is where train_cast.py expects to find them.  Export SAVE_DIR (and
# optionally DIM_DIR) beforehand to write somewhere else; cast.paths reads the
# same two variables.
#
# Requires the harmful/harmless splits under <repo>/data/saladbench_splits/ and,
# for the jailbreak evaluation step, <repo>/third_party/refusal_direction/dataset/
# processed/jailbreakbench.json.  See the README, "Obtaining data and artifacts".
#
# Not an upstream file -- the DIM repository has no bash_script/ -- so unlike the
# vendored Python around it this script carries no "modified from the original"
# notice.  UPSTREAM.md tabulates which files do.
set -euo pipefail

MODEL_PATH="${1:?usage: run.sh <hf-model-id>}"

# The pipeline imports `pipeline.*` and `dataset.*` as top-level packages, so it
# has to be launched from the root of this vendored tree.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python3 -m pipeline.run_pipeline --model_path "$MODEL_PATH"
