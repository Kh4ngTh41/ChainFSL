#!/bin/bash
#SBATCH --job-name=chainfsl_e4_50n
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:a100:20
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=/home/uitchain/datastore/khang/ChainFSL/logs/e4_%j.out
#SBATCH --error=/home/uitchain/datastore/khang/ChainFSL/logs/e4_%j.err

# =========================================================
# E4: Security - 50 nodes, 100 rounds, 20% lazy clients
# Test Sybil attack detection and lazy client mitigation
# =========================================================
REQUIRED_VRAM=32768
N_NODES=50
GLOBAL_ROUNDS=100
LAZY_FRACTION=0.2

module load python312
source /home/uitchain/datastore/khang/venv/bin/activate
unset CUDA_VISIBLE_DEVICES

CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
EXIT_CODE=$?
if [ $EXIT_CODE -eq 10 ]; then echo "$CHECK_OUT"; exit 0
elif [ $EXIT_CODE -eq 11 ]; then echo "$CHECK_OUT"; exit 1; fi
BEST_GPU=$CHECK_OUT
echo "✅ GPU: $BEST_GPU"

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
export CUDA_VISIBLE_DEVICES=$BEST_GPU

cd /home/uitchain/datastore/khang/ChainFSL/chainfsl

python experiments/run_experiment.py \
    --exp e4 \
    --n_nodes $N_NODES \
    --global_rounds $GLOBAL_ROUNDS \
    --lazy_fraction $LAZY_FRACTION \
    --log_dir ./logs \
    --checkpoint_dir ./checkpoints

echo "quit" | nvidia-cuda-mps-control