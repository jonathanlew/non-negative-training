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

export HF_HOME="/home/jonth112/scratch/"

cd $SLURM_SUBMIT_DIR

mv nanochat_train_output.txt nanochat_train_output_profile_int4_$SLURM_JOB_ID.txt
mv nanochat_train_error.txt nanochat_train_error_profile_int4_$SLURM_JOB_ID.txt


nvidia-smi --query-gpu=uuid --format=csv,noheader > my_current_GPU_UUID.txt
nvidia-smi > nvidia_smi.txt

python -c "import torch; print(torch.cuda.is_available()); print(torch.__version__);"

cd $SLURM_SUBMIT_DIR

bash /home/jonth112/gradient-reuse/src/nanochat/runs/speedrun.sh
