#!/bin/bash
#SBATCH --job-name=chainfsl_e1_fully_distributed
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/uitchain/datastore/khang/ChainFSL/logs/e1_fully_distributed_%j.out
#SBATCH --error=/home/uitchain/datastore/khang/ChainFSL/logs/e1_fully_distributed_%j.err

set -euo pipefail

# =========================================================
# PREVENT PYTORCH CPU THRASHING DEADLOCKS ON SLURM
# =========================================================
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export torch_num_threads=$SLURM_CPUS_PER_TASK

# =========================================================
# E1 EXPERIMENT: Fully Distributed P2P Mode
# Compare: Static Cut vs Cluster vs Fully Distributed
# =========================================================
N_NODES=20
GLOBAL_ROUNDS=50
CLUSTER_SIZE=5
PRETRAIN_ROUNDS=50
TIER_DIST=iot_heavy
OFFLINE_HASO=1
PPO_DEVICE=cuda
VENV_DIR="/home/uitchain/datastore/khang/ChainFSL/chainfsl/.venv"
REQ_FILE="/home/uitchain/datastore/khang/ChainFSL/chainfsl/requirements.txt"
PRETRAIN_DIR="/home/uitchain/datastore/khang/ChainFSL/chainfsl/pretrainppo"

if [ "$OFFLINE_HASO" -eq 1 ]; then
    OFFLINE_ARG="--offline_haso --ppo_device $PPO_DEVICE"
    echo "HASO mode: OFFLINE (pretrained PPO inference only)"
else
    OFFLINE_ARG="--ppo_device $PPO_DEVICE"
    echo "HASO mode: ONLINE (PPO updates during experiment)"
fi

module load python312
echo "[SLURM] E1 Fully Distributed P2P Experiment"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

if [ -f "$VENV_DIR/.chainfsl_env_ready" ]; then
    echo "[ENV] Reusing existing .venv dependencies"
else
    echo "[ENV] Installing dependencies..."
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r $REQ_FILE
    touch "$VENV_DIR/.chainfsl_env_ready"
fi

export REQUIRE_CUDA=1
cd /home/uitchain/datastore/khang/ChainFSL/chainfsl

# =========================================================
# PRETRAIN PPO (if not exists)
# =========================================================
echo "=========================================="
echo "PPO Pretrain Check"
echo "=========================================="
if [ ! -f "$PRETRAIN_DIR/$PRETRAIN_ROUNDS/orchestrator.zip" ]; then
    echo "Training PPO offline for $PRETRAIN_ROUNDS rounds..."
    python -u pretrain_pipeline.py \
        --rounds $PRETRAIN_ROUNDS \
        --n_nodes $N_NODES \
        --seed 42 \
        --ppo_device $PPO_DEVICE \
        --log_dir ./logs
else
    echo "Pretrained model FOUND - skipping pretrain"
fi

# =========================================================
# E1-1: Static Cut Baseline (no HASO)
# =========================================================
echo "=========================================="
echo "E1-1: Static Cut Baseline (no HASO)"
echo "=========================================="
python experiments/run_experiment.py \
    --exp e1 \
    --n_nodes $N_NODES \
    --global_rounds $GLOBAL_ROUNDS \
    --tier_dist $TIER_DIST \
    --log_dir ./logs \
    --checkpoint_dir ./checkpoints \
    --checkpoint 2>/dev/null || true

# =========================================================
# E1-2: Cluster Mode (baseline)
# =========================================================
echo "=========================================="
echo "E1-2: Cluster Mode"
echo "=========================================="
python experiments/run_experiment.py \
    --exp e1 \
    --n_nodes $N_NODES \
    --global_rounds $GLOBAL_ROUNDS \
    --tier_dist $TIER_DIST \
    --cluster_size $CLUSTER_SIZE \
    --arch_mode cluster \
    --pretrain_rounds $PRETRAIN_ROUNDS \
    --pretrain_dir $PRETRAIN_DIR \
    --log_dir ./logs \
    --checkpoint_dir ./checkpoints \
    $OFFLINE_ARG

# =========================================================
# E1-3: Fully Distributed P2P Mode (NEW)
# =========================================================
echo "=========================================="
echo "E1-3: Fully Distributed P2P Mode"
echo "=========================================="
python experiments/run_experiment.py \
    --exp e1 \
    --n_nodes $N_NODES \
    --global_rounds $GLOBAL_ROUNDS \
    --tier_dist $TIER_DIST \
    --cluster_size $CLUSTER_SIZE \
    --arch_mode fully_distributed \
    --pretrain_rounds $PRETRAIN_ROUNDS \
    --pretrain_dir $PRETRAIN_DIR \
    --log_dir ./logs \
    --checkpoint_dir ./checkpoints \
    $OFFLINE_ARG

echo "=========================================="
echo "E1 EXPERIMENTS COMPLETED"
echo "=========================================="
