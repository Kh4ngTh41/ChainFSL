#!/bin/bash
#SBATCH --job-name=e1_baseline_comparison
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/uitchain/datastore/khang/ChainFSL/logs/e1_baseline_%j.out
#SBATCH --error=/home/uitchain/datastore/khang/ChainFSL/logs/e1_baseline_%j.err

set -euo pipefail

# =========================================================
# E1 Baseline Comparison — All 6 Methods × 3 Cluster Scenarios
# =========================================================
N_NODES=20
GLOBAL_ROUNDS=50
PRETRAIN_ROUNDS=50
OFFLINE_HASO=1
PPO_DEVICE=cpu
PRETRAIN_DIR="/home/uitchain/datastore/khang/ChainFSL/chainfsl/pretrainppo"
VENV_DIR="/home/uitchain/datastore/khang/ChainFSL/chainfsl/.venv"
REQ_FILE="/home/uitchain/datastore/khang/ChainFSL/chainfsl/requirements.txt"
LOG_DIR="/home/uitchain/datastore/khang/ChainFSL/chainfsl/logs"
CHECKPOINT_DIR="/home/uitchain/datastore/khang/ChainFSL/chainfsl/checkpoints"
PRETRAIN_LOAD_TIMEOUT=120
REQUIRE_CUDA=1
SETUP_ONCE=1

module load python312
echo "[SLURM] E1 Baseline Comparison v2026-05-01"

# Setup venv
if [ "$SETUP_ONCE" -eq 1 ] && [ -f "$VENV_DIR/.chainfsl_env_ready" ]; then
    echo "[ENV] Reusing existing .venv"
else
    echo "[ENV] Installing dependencies..."
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"
    export PYTHONNOUSERSITE=1
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r $REQ_FILE
    touch "$VENV_DIR/.chainfsl_env_ready"
fi

source "$VENV_DIR/bin/activate"
export PYTHONUNBUFFERED=1

# GPU probe
export CUDA_VISIBLE_DEVICES="-1"
HEALTHY_GPU=-1
for i in {0..7}; do
    echo -n "  Testing GPU $i..."
    if timeout 7s python -c "
import os; os.environ['CUDA_VISIBLE_DEVICES']=str($i)
import torch; torch.zeros(1).cuda()
" >/dev/null 2>&1; then
        HEALTHY_GPU=$i
        echo " OK!"
        break
    else
        echo " DEADLOCKED/FAILED"
    fi
done

if [ "$HEALTHY_GPU" -eq -1 ]; then
    echo "[WARN] No healthy GPU found — forcing CPU mode"
    export CUDA_VISIBLE_DEVICES=""
else
    echo "[STEP] Using GPU $HEALTHY_GPU"
    export CUDA_VISIBLE_DEVICES="$HEALTHY_GPU"
fi

cd /home/uitchain/datastore/khang/ChainFSL/chainfsl

# Cluster scenarios: cluster_size for N=20
# 10%: cluster_size=2 (10 clusters)
# 20%: cluster_size=4 (5 clusters)
# 30%: cluster_size=6 (3-4 clusters, auto-adjusted)

CLUSTER_SIZES=(2 4 6)
CLUSTER_NAMES=("10pct" "20pct" "30pct")

# =========================================================
# STEP 1: Pretrain PPO if not exists
# =========================================================
if [ ! -f "$PRETRAIN_DIR/$PRETRAIN_ROUNDS/orchestrator.zip" ]; then
    echo "Pretraining PPO for $PRETRAIN_ROUNDS rounds..."
    python -u pretrain_pipeline.py \
        --rounds $PRETRAIN_ROUNDS \
        --n_nodes $N_NODES \
        --seed 42 \
        --ppo_device $PPO_DEVICE \
        --log_dir ./logs
else
    echo "Pretrained model found — skipping pretrain"
fi

# =========================================================
# STEP 2: Run HASO architectures
# =========================================================
echo "=========================================="
echo "HASO-PerNode (per-node PPO agents)"
echo "=========================================="
python -u experiments/run_experiment.py \
    --exp e1 \
    --arch_mode per_node \
    --n_nodes $N_NODES \
    --global_rounds $GLOBAL_ROUNDS \
    --pretrain_rounds $PRETRAIN_ROUNDS \
    --pretrain_dir $PRETRAIN_DIR \
    --pretrain_load_timeout $PRETRAIN_LOAD_TIMEOUT \
    --ppo_device $PPO_DEVICE \
    --offline_haso \
    --log_dir $LOG_DIR \
    --checkpoint_dir $CHECKPOINT_DIR

echo "=========================================="
echo "HASO-Cluster (1 PPO per cluster)"
echo "=========================================="
for i in "${!CLUSTER_NAMES[@]}"; do
    NAME="${CLUSTER_NAMES[$i]}"
    CS="${CLUSTER_SIZES[$i]}"
    echo "  Cluster scenario: $NAME (cluster_size=$CS)"
    python -u experiments/run_experiment.py \
        --exp e1 \
        --arch_mode cluster \
        --cluster_size $CS \
        --n_nodes $N_NODES \
        --global_rounds $GLOBAL_ROUNDS \
        --pretrain_rounds $PRETRAIN_ROUNDS \
        --pretrain_dir $PRETRAIN_DIR \
        --pretrain_load_timeout $PRETRAIN_LOAD_TIMEOUT \
        --ppo_device $PPO_DEVICE \
        --offline_haso \
        --log_dir $LOG_DIR \
        --checkpoint_dir $CHECKPOINT_DIR
done

echo "=========================================="
echo "HASO-Centralized (1 PPO for all nodes)"
echo "=========================================="
python -u experiments/run_experiment.py \
    --exp e1 \
    --arch_mode centralized \
    --n_nodes $N_NODES \
    --global_rounds $GLOBAL_ROUNDS \
    --pretrain_rounds $PRETRAIN_ROUNDS \
    --pretrain_dir $PRETRAIN_DIR \
    --pretrain_load_timeout $PRETRAIN_LOAD_TIMEOUT \
    --ppo_device $PPO_DEVICE \
    --offline_haso \
    --log_dir $LOG_DIR \
    --checkpoint_dir $CHECKPOINT_DIR

# =========================================================
# STEP 3: Run baseline methods
# =========================================================
for BASELINE in fedavg splitfed_v1 splitfed_v2; do
    echo "=========================================="
    echo "Running baseline: $BASELINE"
    echo "=========================================="
    python -u experiments/run_experiment.py \
        --exp e1 \
        --baseline $BASELINE \
        --n_nodes $N_NODES \
        --global_rounds $GLOBAL_ROUNDS \
        --cluster_ratio 0.20 \
        --log_dir $LOG_DIR \
        --checkpoint_dir $CHECKPOINT_DIR
done

echo "=========================================="
echo "All E1 baseline experiments completed!"
echo "=========================================="