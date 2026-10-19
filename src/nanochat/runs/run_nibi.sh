#!/bin/bash

#SBATCH --job-name=nanochat_train
#SBATCH --account=rrg-aamodt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=2:30:00
#SBATCH --gpus-per-node=h100:8
#SBATCH --output=nanochat_train_output.txt
#SBATCH --error=nanochat_train_error.txt
#SBATCH --mail-user=jlew256@gmail.com
#SBATCH --mail-type=ALL

export HF_TOKEN=$(cat /home/jonth112/.cache/huggingface/token)

module load gcc/12.3 arrow/24.0.0 python/3.11 scipy-stack/2026a cuda/13.2 opencv/4.13.0
module list
export PIP_CACHE_DIR=$SLURM_TMPDIR/pip_cache

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/home/jonth112/projects/def-aamodt/jonathan/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu --group dev
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

pip install --no-index wandb

### Save your wandb API key in your .bash_profile or replace $API_KEY with your actual API key. Uncomment the line below and comment out "wandb offline" if running in online mode ###
wandb login $API_KEY 

export HF_HOME="/home/jonth112/scratch/"

cd $SLURM_SUBMIT_DIR

mv nanochat_train_output.txt nanochat_train_output_profile_int4_$SLURM_JOB_ID.txt
mv nanochat_train_error.txt nanochat_train_error_profile_int4_$SLURM_JOB_ID.txt

cd $SLURM_SUBMIT_DIR

OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1