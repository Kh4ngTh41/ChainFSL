#!/bin/bash
#SBATCH --job-name=chainfsl_e1_50n
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:a100:20
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=/home/uitchain/datastore/khang/ChainFSL/logs/e1_%j.out
#SBATCH --error=/home/uitchain/datastore/khang/ChainFSL/logs/e1_%j.err

# =========================================================
# E1: HASO Effectiveness - 50 nodes, 100 rounds
# Hypothesis: HASO reduces training latency 40-60% vs baselines
# =========================================================
REQUIRED_VRAM=32768
EXPERIMENT_NAME="e1"
N_NODES=50
GLOBAL_ROUNDS=100
SKIP_BASELINES=""

# =========================================================
# CHUẨN BỊ MÔI TRƯỜNG
# =========================================================
module load python312
source /home/uitchain/datastore/khang/venv/bin/activate

unset CUDA_VISIBLE_DEVICES

# === GỌI HELPER TÌM GPU (vRAM >= REQUIRED_VRAM) ===
CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
EXIT_CODE=$?
if [ $EXIT_CODE -eq 10 ]; then
    echo "$CHECK_OUT"
    exit 0
elif [ $EXIT_CODE -eq 11 ]; then
    echo "$CHECK_OUT"
    exit 1
fi
BEST_GPU=$CHECK_OUT
echo "✅ Job $SLURM_JOB_ID bắt đầu trên GPU: $BEST_GPU"

# =========================================================
# KHỞI TẠO PRIVATE MPS SERVER
# =========================================================
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
export CUDA_VISIBLE_DEVICES=$BEST_GPU

# =========================================================
# CHẠY EXPERIMENT
# =========================================================
cd /home/uitchain/datastore/khang/ChainFSL/chainfsl

python experiments/run_experiment.py \
    --exp $EXPERIMENT_NAME \
    --n_nodes $N_NODES \
    --global_rounds $GLOBAL_ROUNDS \
    --log_dir ./logs \
    --checkpoint_dir ./checkpoints

echo "quit" | nvidia-cuda-mps-control