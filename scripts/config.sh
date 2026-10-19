#!/bin/bash
# Shared settings for every stage of the CAST pipeline.
#
# Sourced by scripts/0*.sh, scripts/10_run_utility.sh and scripts/ablations/*.sh.
# It sets variables only -- it never changes directory and never runs anything,
# so it is safe to source from an interactive shell to see what a stage will do.
#
# Everything here can be overridden from the environment, so one checkout covers
# all three models of the paper:
#
#   MODEL_FAMILY=qwen  MODEL_SIZE=14b scripts/06_run_steering.sh
#   BEHAVIORS="myopic-reward"         scripts/05_train_cast.sh
#
# Credentials are never stored in this repository.  Export them yourself:
#   OPENAI_API_KEY   required by scripts/09_judge.sh (the two GPT judges)
#   HF_TOKEN         required for gated models (Llama)
#   WANDB_*          optional; train_cast.py also accepts --no_wandb

# ---------------------------------------------------------------- model ------
MODEL_FAMILY="${MODEL_FAMILY:-qwen}"
MODEL_SIZE="${MODEL_SIZE:-7b}"

# Per-model settings.  LAYER is the injection layer the paper reports for this
# model; N_LAYERS is the model's depth, needed by normalize_vectors.py and by
# stage 8's layer sweep.  The multipliers CAST *optimises* over are not set
# here: they belong to the training run, and live as `steering_multipliers` in
# configs/<model_id>.yaml (-0.5/+0.5, and half that for 14B -- deliberately
# smaller than the evaluation sweep below).
case "${MODEL_FAMILY}-${MODEL_SIZE}" in
  llama-8b)
    MODEL_PATH="meta-llama/llama-3.1-8b-instruct"
    LAYER="${LAYER:-13}"
    N_LAYERS="${N_LAYERS:-32}"
    ;;
  qwen-7b)
    MODEL_PATH="qwen/qwen2.5-7b-instruct"
    LAYER="${LAYER:-18}"
    N_LAYERS="${N_LAYERS:-28}"
    ;;
  qwen-14b)
    MODEL_PATH="qwen/qwen2.5-14b-instruct"
    LAYER="${LAYER:-30}"
    N_LAYERS="${N_LAYERS:-48}"
    ;;
  *)
    echo "config.sh: unknown model '${MODEL_FAMILY}-${MODEL_SIZE}'." >&2
    echo "Supported: llama-8b, qwen-7b, qwen-14b.  Add a case block here to" >&2
    echo "extend the pipeline, and a matching configs/<model-id>.yaml." >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

# Directory name used for this model throughout artifacts/ -- the HuggingFace id
# without its owner, exactly what train_cast.py derives.
MODEL_ID="${MODEL_PATH##*/}"

# Per-model CAST hyper-parameters (epsilons, dual step size, lambda inits).
CAST_CONFIG="${CAST_CONFIG:-configs/${MODEL_ID}.yaml}"

# Sub-directory stage 10 writes its utility results under, one per model.
MODEL_SUBDIR="${MODEL_SUBDIR:-${MODEL_FAMILY}_${MODEL_SIZE}_instruct}"

# ------------------------------------------------------------- behaviours ----
# The three behaviours the main tables report.  Tables 15-16 use a different set;
# see scripts/ablations/extra_behaviors.sh.
BEHAVIORS="${BEHAVIORS:-power-seeking-inclination self-awareness-good-text-model corrigible-more-HHH}"

# ------------------------------------------------------------ multipliers ----
# MULTIPLIERS is the evaluation sweep for the unsteered and CAA runs and
# includes 0, the no-steering reference every delta in the paper is taken
# against.  CAST runs reuse that same 0 row, so they omit it -- steering with
# multiplier 0 adds nothing and would only duplicate the reference.
MULTIPLIERS="${MULTIPLIERS:--1.5 -1 -0.5 0 0.5 1 1.5}"
CAST_MULTIPLIERS="${CAST_MULTIPLIERS:--1.5 -1 -0.5 0.5 1 1.5}"

# The false-refusal and open-ended stages are far more expensive per prompt
# (a GPT judge call each), so they run on the three headline multipliers only.
EFFECT_MULTIPLIERS="${EFFECT_MULTIPLIERS:--1 0 1}"
CAST_EFFECT_MULTIPLIERS="${CAST_EFFECT_MULTIPLIERS:--1 1}"

# Stage 10 is the exception.  Its two runs land in separate files, so the CAST
# run cannot borrow the CAA run's multiplier-0 row the way stages 7 and 8 do and
# needs one of its own -- Table 11 reports every number as a delta against the
# unsteered score.  Steering at 0 adds a zero vector, so the extra row costs one
# forward pass and reproduces the unsteered reference exactly.
UTILITY_CAST_MULTIPLIERS="${UTILITY_CAST_MULTIPLIERS:--1 0 1}"

# ----------------------------------------------------------------- attacks ---
DATASET="${DATASET:-jailbreakbench}"
# The paper's seven scenarios.  AutoDAN prompts are several thousand tokens
# long, so it is run separately at a smaller batch size.
ATTACKS="${ATTACKS:-none prefix refusal_suppression aim gcg pair}"
ATTACKS_LONG_PROMPT="${ATTACKS_LONG_PROMPT:-autodan}"

BENIGN_SOURCES="${BENIGN_SOURCES:-xtest alpaca}"

# ------------------------------------------------------------ batch sizes ----
# Sized for one 80 GB GPU at 7B.  Halve for 14B, halve again if you see OOM.
BATCH_SIZE="${BATCH_SIZE:-16}"
BATCH_SIZE_LONG_PROMPT="${BATCH_SIZE_LONG_PROMPT:-8}"
BATCH_SIZE_OPEN_ENDED="${BATCH_SIZE_OPEN_ENDED:-10}"
BATCH_SIZE_UTILITY="${BATCH_SIZE_UTILITY:-16}"

# --------------------------------------------------------------- vectors -----
SAFETY_LOSS_TYPE="${SAFETY_LOSS_TYPE:-masked_kl}"
# Directory train_cast.py writes a rank-1 run into: dim_<rank>_<loss>.
CAST_VARIANT="${CAST_VARIANT:-dim_1_${SAFETY_LOSS_TYPE}}"

# Passed to --custom_vector_path / --custom_vector, which substitute {behavior}
# themselves, so one invocation can sweep several behaviours with one model load.
# Built with an if rather than ${CAST_VECTOR:-...}: the closing brace of
# {behavior} would end the parameter expansion and truncate the path.
if [ -z "${CAST_VECTOR:-}" ]; then
    CAST_VECTOR="artifacts/cast_vectors/${MODEL_ID}/${CAST_VARIANT}/sanitized_{behavior}_${MODEL_ID}_${SAFETY_LOSS_TYPE}.pt"
fi

# ---------------------------------------------------------- result trees -----
# Defaults match the entry points' own defaults.
JAILBREAK_RESULTS="${JAILBREAK_RESULTS:-jailbreak_results}"
BENIGN_RESULTS="${BENIGN_RESULTS:-benign_results}"
OPEN_ENDED_RESULTS="${OPEN_ENDED_RESULTS:-results}"
UTILITY_RESULTS="${UTILITY_RESULTS:-utility}"

# --------------------------------------------------------------- utility -----
# The four benchmarks of Table 11.  Each is loaded and scored independently, so
# dropping one to save GPU time leaves the others' numbers unchanged.
UTILITY_BENCHMARKS="${UTILITY_BENCHMARKS:-mmlu triviaqa gsm8k ifeval}"
UTILITY_NUM_INSTANCES="${UTILITY_NUM_INSTANCES:-1000}"
UTILITY_SEED="${UTILITY_SEED:-42}"

# ----------------------------------------------------------------- judge -----
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-16}"
# gpt-4o is the judge behind every LLM-as-a-Judge number in the paper.  Both the
# false-refusal verdict and the 0-10 behaviour score depend on the judge, so a
# different model here does not reproduce Tables 3, 5, 9, 13-16.
GPT_JUDGE_MODEL="${GPT_JUDGE_MODEL:-gpt-4o}"

# ---------------------------------------------------------------- helpers ----
# Print a stage banner.  Stages are long; this is what makes a log readable.
banner() {
    echo
    echo "=============================================================="
    echo "== $*"
    echo "== model=${MODEL_PATH}  layer=${LAYER}"
    echo "=============================================================="
}

# Fail early with an actionable message instead of after an hour of generation.
require_env() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        echo "$0: \$${name} is not set." >&2
        echo "Export it in your shell; this repository never stores credentials." >&2
        exit 1
    fi
}
