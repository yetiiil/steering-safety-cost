#!/bin/bash
# Tables 15-16 -- three behaviours outside the main set.
#
# Reads:   artifacts/caa_vectors/<behavior>/, artifacts/DIM/
# Writes:  artifacts/cast_vectors/<model_id>/<variant>/ and the standard
#          jailbreak_results/, benign_results/, results/ trees
# Feeds:   Tables 15 and 16.
#
#   scripts/ablations/extra_behaviors.sh                       # the published three
#   scripts/ablations/extra_behaviors.sh sycophancy            # just one
#
# Appendix B.7 runs these on Qwen2.5-7B-Instruct only, which is the default
# model; MODEL_FAMILY/MODEL_SIZE still select another one.
#
# Unlike the other ablations these runs go into the MAIN result trees: the
# behaviour is part of every result filename, so they cannot collide with the
# main three, and the shared m = 0 baseline is reused.
#
# All three train with the model's standard config, configs/<model_id>.yaml --
# there is no per-behaviour config.
#
# `length_conciseness` is scored by output length rather than by a GPT judge, so
# length_effect_stats.py is run for it at the end.  Neither it nor `sycophancy`
# has a 0-10 rubric in evaluate_results.py, so stage 9 warns and produces no
# behaviour score for them; the attack-success and false-refusal columns are
# unaffected.
#
# `length_conciseness` also needs a generation set this repository cannot
# provide: scripts/00_download_data.py builds its open-ended test prompts but
# not the contrastive A/B pairs its steering vector was extracted from.  Run
# `python scripts/00_download_data.py --only behaviors-conciseness-generate` to
# see what is missing.  The other two behaviours are fetched in full.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/config.sh

EXTRA_BEHAVIORS=("$@")
if [ ${#EXTRA_BEHAVIORS[@]} -eq 0 ]; then
    EXTRA_BEHAVIORS=(sycophancy myopic-reward length_conciseness)
fi

# A behaviour needs its normalised CAA vector before anything else can run.
# Stage 2 reads its target norm from configs/normalization_target_norms.yaml, so
# normalising these three does not disturb the scale of the main three and they
# can be added at any time.
missing=()
for behavior in "${EXTRA_BEHAVIORS[@]}"; do
    vector="artifacts/caa_vectors/${behavior}/vec_layer_${LAYER}_${MODEL_PATH//[\/-]/_}.pt"
    if [ ! -f "$vector" ]; then
        missing+=("$vector")
    fi
done
if [ ${#missing[@]} -gt 0 ]; then
    echo "$0: no normalised CAA vector for:" >&2
    printf '  %s\n' "${missing[@]}" >&2
    echo >&2
    echo "Generate the raw vectors, then normalise them:" >&2
    echo "  BEHAVIORS=\"${EXTRA_BEHAVIORS[*]}\" scripts/01_generate_vectors.sh" >&2
    echo "  BEHAVIORS=\"${EXTRA_BEHAVIORS[*]}\" scripts/02_normalize_vectors.sh" >&2
    echo "The main three keep the scale they were released with." >&2
    exit 1
fi

export MODEL_FAMILY MODEL_SIZE
export BEHAVIORS="${EXTRA_BEHAVIORS[*]}"

banner "Table 15: ${BEHAVIORS}"

scripts/05_train_cast.sh
scripts/06_run_steering.sh
scripts/07_run_benign.sh
scripts/08_run_open_ended.sh
scripts/09_judge.sh

for behavior in "${EXTRA_BEHAVIORS[@]}"; do
    case "$behavior" in
        length_conciseness)
            banner "Table 15: output-length statistics for ${behavior}"
            python length_effect_stats.py \
                --behavior "$behavior" \
                --results_dir "$OPEN_ENDED_RESULTS" \
                --layer "$LAYER" \
                --model_family "$MODEL_FAMILY" \
                --model_size "$MODEL_SIZE" \
                --tokenizer "$MODEL_PATH"
            ;;
    esac
done

echo "Table 15 runs are in the main result trees, tagged by behaviour."
