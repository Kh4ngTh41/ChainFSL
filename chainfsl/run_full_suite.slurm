#!/bin/bash
#SBATCH --job-name=chainfsl_full_suite
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:a100:30
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --output=/home/uitchain/datastore/khang/ChainFSL/logs/full_%j.out
#SBATCH --error=/home/uitchain/datastore/khang/ChainFSL/logs/full_%j.err

# =========================================================
# FULL EXPERIMENT SUITE - All 7 experiments
# 50 nodes, 100 rounds per experiment
# Sequential: E1 → E2 → E3 → E4 → E5 → E6 → E7
# =========================================================
REQUIRED_VRAM=49152
N_NODES=50
GLOBAL_ROUNDS=100

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

# =========================================================
# E1: HASO Effectiveness
# =========================================================
echo "=========================================="
echo "Running E1: HASO Effectiveness"
echo "=========================================="
python experiments/run_experiment.py \
    --exp e1 \
    --n_nodes $N_NODES \
    --global_rounds $GLOBAL_ROUNDS \
    --log_dir ./logs \
    --checkpoint_dir ./checkpoints

# =========================================================
# E2: Scalability
# =========================================================
echo "=========================================="
echo "Running E2: Scalability"
echo "=========================================="
python experiments/run_experiment.py \
    --exp e2 \
    --n_nodes $N_NODES \
    --global_rounds $GLOBAL_ROUNDS \
    --log_dir ./logs \
    --checkpoint_dir ./checkpoints

# =========================================================
# E3: Non-IID
# =========================================================
echo "=========================================="
echo "Running E3: Non-IID Data"
echo "=========================================="
python experiments/run_experiment.py \
    --exp e3 \
    --n_nodes $N_NODES \
    --global_rounds $GLOBAL_ROUNDS \
    --alpha 0.1 \
    --log_dir ./logs \
    --checkpoint_dir ./checkpoints

# =========================================================
# E4: Security
# =========================================================
echo "=========================================="
echo "Running E4: Security"
echo "=========================================="
python experiments/run_experiment.py \
    --exp e4 \
    --n_nodes $N_NODES \
    --global_rounds $GLOBAL_ROUNDS \
    --lazy_fraction 0.2 \
    --log_dir ./logs \
    --checkpoint_dir ./checkpoints

# =========================================================
# E5: Incentive
# =========================================================
echo "=========================================="
echo "Running E5: Incentive Mechanism"
echo "=========================================="
python experiments/run_experiment.py \
    --exp e5 \
    --n_nodes $N_NODES \
    --global_rounds $GLOBAL_ROUNDS \
    --log_dir ./logs \
    --checkpoint_dir ./checkpoints

# =========================================================
# E6: Ablation
# =========================================================
echo "=========================================="
echo "Running E6: Ablation Study"
echo "=========================================="
python experiments/run_experiment.py \
    --exp e6 \
    --n_nodes $N_NODES \
    --global_rounds $GLOBAL_ROUNDS \
    --ablation full \
    --log_dir ./logs \
    --checkpoint_dir ./checkpoints

# =========================================================
# E7: Blockchain Overhead
# =========================================================
echo "=========================================="
echo "Running E7: Blockchain Overhead"
echo "=========================================="
python experiments/run_experiment.py \
    --exp e7 \
    --n_nodes $N_NODES \
    --global_rounds $GLOBAL_ROUNDS \
    --log_dir ./logs \
    --checkpoint_dir ./checkpoints

echo "quit" | nvidia-cuda-mps-control
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="