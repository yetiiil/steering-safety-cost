#!/bin/bash
# Stage 3 -- the DIM refusal direction (Arditi et al., 2024).
#
# Reads:   data/saladbench_splits/{harmful,harmless}_{train,val}.json
# Writes:  artifacts/DIM/<model_id>/{direction.pt,direction_metadata.json}
# Feeds:   stage 4 (the refusal-direction-ablation baseline of Table 1) and
#          stage 5, which reads the metadata for the layer and token position.
#
# This wraps the vendored pipeline in third_party/refusal_direction/.  Export
# SAVE_DIR (and optionally DIM_DIR) to write elsewhere; cast.paths reads the
# same two variables, so the rest of the pipeline follows automatically.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh

if [ ! -f "data/saladbench_splits/harmful_train.json" ]; then
    echo "data/saladbench_splits/ is missing." >&2
    echo "Run: python scripts/00_download_data.py --only dim-splits" >&2
    exit 1
fi
if [ ! -f "third_party/refusal_direction/dataset/processed/jailbreakbench.json" ]; then
    echo "The DIM pipeline's evaluation stage needs its own copy of the" >&2
    echo "JailbreakBench prompts. Run:" >&2
    echo "  python scripts/00_download_data.py --only dim-eval-set" >&2
    exit 1
fi

banner "Stage 3: DIM refusal direction"

bash third_party/refusal_direction/bash_script/run.sh "$MODEL_PATH"

echo "Direction written under artifacts/DIM/${MODEL_ID}/."
