#!/bin/bash
#SBATCH --job-name=chainfsl_full_suite
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:2
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --output=/home/uitchain/datastore/khang/ChainFSL/logs/full_%j.out
#SBATCH --error=/home/uitchain/datastore/khang/ChainFSL/logs/full_%j.err

# =========================================================
# FULL EXPERIMENT SUITE - All 7 experiments
# Hierarchical HASO: k clusters, a=5 nodes/cluster
# Pretrain: auto-detect, skip if exists
# =========================================================
REQUIRED_VRAM=30000
N_NODES=20
GLOBAL_ROUNDS=50
CLUSTER_SIZE=5
PRETRAIN_ROUNDS=500
VENV_DIR="/home/uitchain/datastore/khang/ChainFSL/chainfsl/.venv"
REQ_FILE="/home/uitchain/datastore/khang/ChainFSL/chainfsl/requirements.txt"
PRETRAIN_DIR="/home/uitchain/datastore/khang/ChainFSL/chainfsl/pretrainppo"

module load python312

pip3 install -r $REQ_FILE
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
# PRETRAIN PPO - Lightweight, skip if exists
# =========================================================
echo "=========================================="
echo "PPO Pretrain Check"
echo "=========================================="
if [ ! -f "$PRETRAIN_DIR/$PRETRAIN_ROUNDS/orchestrator.zip" ]; then
    echo "No pretrained model found. Training PPO (lightweight) for $PRETRAIN_ROUNDS rounds..."
    python3 pretrain_simple.py \
        --rounds $PRETRAIN_ROUNDS \
        --n_nodes $N_NODES \
        --seed 42 \
        --save_dir $PRETRAIN_DIR
    echo "Pretrain complete!"
else
    echo "Pretrained model FOUND at $PRETRAIN_DIR/$PRETRAIN_ROUNDS/ - skipping pretrain"
fi

# =========================================================
# E1: HASO Effectiveness (skip baselines)
# =========================================================
echo "=========================================="
echo "Running E1: HASO Effectiveness"
echo "=========================================="
python experiments/run_experiment.py \
    --exp e1 \
    --n_nodes $N_NODES \
    --global_rounds $GLOBAL_ROUNDS \
    --cluster_size $CLUSTER_SIZE \
    --pretrain_rounds $PRETRAIN_ROUNDS \
    --pretrain_dir $PRETRAIN_DIR \
    --skip_baselines \
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
    --cluster_size $CLUSTER_SIZE \
    --pretrain_rounds $PRETRAIN_ROUNDS \
    --pretrain_dir $PRETRAIN_DIR \
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
    --cluster_size $CLUSTER_SIZE \
    --alpha 0.1 \
    --pretrain_rounds $PRETRAIN_ROUNDS \
    --pretrain_dir $PRETRAIN_DIR \
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
    --cluster_size $CLUSTER_SIZE \
    --lazy_fraction 0.2 \
    --pretrain_rounds $PRETRAIN_ROUNDS \
    --pretrain_dir $PRETRAIN_DIR \
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
    --cluster_size $CLUSTER_SIZE \
    --pretrain_rounds $PRETRAIN_ROUNDS \
    --pretrain_dir $PRETRAIN_DIR \
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
    --cluster_size $CLUSTER_SIZE \
    --ablation full \
    --pretrain_rounds $PRETRAIN_ROUNDS \
    --pretrain_dir $PRETRAIN_DIR \
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
    --cluster_size $CLUSTER_SIZE \
    --pretrain_rounds $PRETRAIN_ROUNDS \
    --pretrain_dir $PRETRAIN_DIR \
    --log_dir ./logs \
    --checkpoint_dir ./checkpoints

echo "quit" | nvidia-cuda-mps-control
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="