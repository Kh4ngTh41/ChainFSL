#!/bin/bash
#SBATCH --job-name=chainfsl_full_suite
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/uitchain/datastore/khang/ChainFSL/logs/full_%j.out
#SBATCH --error=/home/uitchain/datastore/khang/ChainFSL/logs/full_%j.err

set -euo pipefail

# =========================================================
# PREVENT PYTORCH CPU THRASHING DEADLOCKS ON SLURM
# PyTorch os.cpu_count() sees 128 cores on DGX but cgroups only allows 8.
# This causes massive thread spinning and deadlocks during forward/backward.
# =========================================================
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export torch_num_threads=$SLURM_CPUS_PER_TASK

# =========================================================
# FULL EXPERIMENT SUITE - All 7 experiments
# Hierarchical HASO: k clusters, a=5 nodes/cluster
# Pretrain: auto-detect, skip if exists
# =========================================================
REQUIRED_VRAM=30000
N_NODES=20
GLOBAL_ROUNDS=50
CLUSTER_SIZE=5
PRETRAIN_ROUNDS=50
OFFLINE_HASO=1
PPO_DEVICE=cuda
SETUP_ONCE=1
VENV_DIR="/home/uitchain/datastore/khang/ChainFSL/chainfsl/.venv"
REQ_FILE="/home/uitchain/datastore/khang/ChainFSL/chainfsl/requirements.txt"
PRETRAIN_DIR="/home/uitchain/datastore/khang/ChainFSL/chainfsl/pretrainppo"

if [ "$OFFLINE_HASO" -eq 1 ]; then
    OFFLINE_ARG="--offline_haso --ppo_device $PPO_DEVICE"
    echo "HASO mode: OFFLINE (pretrained PPO inference only, PPO_DEVICE=$PPO_DEVICE)"
else
    OFFLINE_ARG="--ppo_device $PPO_DEVICE"
    echo "HASO mode: ONLINE (PPO updates during experiment, PPO_DEVICE=$PPO_DEVICE)"
fi

module load python312
echo "[SLURM SCRIPT] chainfsl run_all stable version"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

if [ "$SETUP_ONCE" -eq 1 ] && [ -f "$VENV_DIR/.chainfsl_env_ready" ]; then
    echo "[ENV] Reusing existing .venv dependencies (setup-once marker found)."
else
    echo "[ENV] Installing/updating dependencies in .venv..."
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r $REQ_FILE
    touch "$VENV_DIR/.chainfsl_env_ready"
fi

echo "[STEP] GPU mode selected. Sequential execution prevents CUDA context deadlocks."
# Use the GPU assigned by SLURM
export REQUIRE_CUDA=1

cd /home/uitchain/datastore/khang/ChainFSL/chainfsl

# =========================================================
# PRETRAIN PPO - Auto-detect: skip if exists
# =========================================================
echo "=========================================="
echo "PPO Pretrain Check"
echo "=========================================="
if [ ! -f "$PRETRAIN_DIR/$PRETRAIN_ROUNDS/orchestrator.zip" ]; then
    echo "No pretrained model found. Training PPO offline for $PRETRAIN_ROUNDS rounds..."
    python -u pretrain_pipeline.py \
        --rounds $PRETRAIN_ROUNDS \
        --n_nodes $N_NODES \
        --seed 42 \
        --ppo_device $PPO_DEVICE \
        --log_dir ./logs
else
    echo "Pretrained model FOUND at $PRETRAIN_DIR/$PRETRAIN_ROUNDS/ - skipping pretrain"
fi

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
    --cluster_size $CLUSTER_SIZE \
    --pretrain_rounds $PRETRAIN_ROUNDS \
    --pretrain_dir $PRETRAIN_DIR \
    --log_dir ./logs \
    --checkpoint_dir ./checkpoints \
    $OFFLINE_ARG

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
    --checkpoint_dir ./checkpoints \
    $OFFLINE_ARG

# =========================================================
# E3: Non-IID Robustness
# =========================================================
echo "=========================================="
echo "Running E3: Non-IID Robustness"
echo "=========================================="
python experiments/run_experiment.py \
    --exp e3 \
    --n_nodes $N_NODES \
    --global_rounds $GLOBAL_ROUNDS \
    --cluster_size $CLUSTER_SIZE \
    --pretrain_rounds $PRETRAIN_ROUNDS \
    --pretrain_dir $PRETRAIN_DIR \
    --log_dir ./logs \
    --checkpoint_dir ./checkpoints \
    $OFFLINE_ARG

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
    --checkpoint_dir ./checkpoints \
    $OFFLINE_ARG

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
    --checkpoint_dir ./checkpoints \
    $OFFLINE_ARG

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
    --ablation no_haso \
    --pretrain_rounds $PRETRAIN_ROUNDS \
    --pretrain_dir $PRETRAIN_DIR \
    --log_dir ./logs \
    --checkpoint_dir ./checkpoints \
    $OFFLINE_ARG

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
    --checkpoint_dir ./checkpoints \
    $OFFLINE_ARG

echo "=========================================="
echo "ALL EXPERIMENTS COMPLETED"
echo "=========================================="