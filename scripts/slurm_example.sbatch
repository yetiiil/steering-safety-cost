#!/bin/bash
# Example Slurm submission for one pipeline stage.  Not a working script for any
# particular cluster: every #SBATCH line below is a placeholder.  Copy it, fill
# in your site's partition and account, and submit:
#
#   sbatch --export=ALL,STAGE=scripts/06_run_steering.sh scripts/slurm_example.sbatch
#
# or edit STAGE below.  One stage per job keeps the walltime honest; stages 6-8
# are the long ones (hours of generation), stage 9 is API-bound rather than
# GPU-bound and is better run outside the allocation.

#SBATCH --job-name=cast
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# --- fill these in for your site -------------------------------------------
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
## #SBATCH --qos=<QOS>
## #SBATCH --nodelist=<NODE>          # only if you must pin to one machine
# ---------------------------------------------------------------------------

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set -euo pipefail

# Slurm already restricts the job to the GPUs it allocated, and it numbers them
# from 0 within the job.  Setting CUDA_VISIBLE_DEVICES here -- for instance to a
# device id read from nvidia-smi -- points the job at a GPU it was not given, or
# at nothing at all.  Leave it alone.

# Activate your environment however your site expects (a module, a venv, a
# container).  Nothing in this repository depends on a particular mechanism.
# Example:
#   source .venv/bin/activate

# Caches: point them at shared scratch rather than a home directory with a quota.
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"

# Credentials come from the environment and are never stored in the repository.
#   HF_TOKEN         gated models
#   OPENAI_API_KEY   stage 9's two GPT judges
#   WANDB_API_KEY    optional; train_cast.py also accepts --no_wandb
# Export them in your shell before sbatch and submit with --export=ALL, or read
# them from a file that is not in version control.

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

MODEL_FAMILY="${MODEL_FAMILY:-qwen}"
MODEL_SIZE="${MODEL_SIZE:-7b}"
export MODEL_FAMILY MODEL_SIZE

STAGE="${STAGE:-scripts/06_run_steering.sh}"

echo "host=$(hostname)  job=${SLURM_JOB_ID}  stage=${STAGE}"
echo "model=${MODEL_FAMILY}-${MODEL_SIZE}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

bash "$STAGE"
