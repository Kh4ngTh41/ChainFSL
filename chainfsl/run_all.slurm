#!/bin/bash
#SBATCH --job-name=chainfsl_full_suite
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/uitchain/datastore/khang/ChainFSL/logs/full_%j.out
#SBATCH --error=/home/uitchain/datastore/khang/ChainFSL/logs/full_%j.err

set -euo pipefail

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
PPO_DEVICE=cpu
REQUIRE_CUDA=1
SETUP_ONCE=1
VENV_DIR="/home/uitchain/datastore/khang/ChainFSL/chainfsl/.venv"
REQ_FILE="/home/uitchain/datastore/khang/ChainFSL/chainfsl/requirements.txt"
PRETRAIN_DIR="/home/uitchain/datastore/khang/ChainFSL/chainfsl/pretrainppo"

if [ "$OFFLINE_HASO" -eq 1 ]; then
    OFFLINE_ARG="--offline_haso"
    echo "HASO mode: OFFLINE (pretrained PPO inference only, PPO_DEVICE=$PPO_DEVICE)"
else
    OFFLINE_ARG=""
    echo "HASO mode: ONLINE (PPO updates during experiment, PPO_DEVICE=$PPO_DEVICE)"
fi

module load python312
echo "[SLURM SCRIPT] chainfsl run_all v2026-04-26-import-probe"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [ "$SETUP_ONCE" -eq 1 ] && [ -f "$VENV_DIR/.chainfsl_env_ready" ]; then
    echo "[ENV] Reusing existing .venv dependencies (setup-once marker found)."
else
    echo "[ENV] Installing/updating dependencies in .venv..."
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r $REQ_FILE
    touch "$VENV_DIR/.chainfsl_env_ready"
fi
echo "Requested 1 GPU via SLURM. PPO/ChainFSL will use CUDA if torch can initialize it."

echo "[STEP] GPU visibility check..."
nvidia-smi -L || true

if [ "$REQUIRE_CUDA" -eq 1 ]; then
    echo "[STEP] CUDA required; proceeding directly to experiment launch after visibility check."
fi

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
# E1: HASO Effectiveness (skip baselines - just ChainFSL vs ChainFSL-NoHASO)
# =========================================================
echo "=========================================="
echo "Running E1: HASO Effectiveness"
echo "=========================================="
echo "[STEP] Probing Python import path for experiments runner (timeout 120s)..."
if ! timeout 120s python -u -X faulthandler - << 'PY'
print("[PROBE] import experiments.run_experiment ...", flush=True)
import experiments.run_experiment
print("[PROBE] import experiments.run_experiment OK", flush=True)
PY
then
    echo "[FATAL] Python import probe failed or timed out before E1 launch."
    exit 3
fi

echo "[STEP] Launching E1 command..."
python -u experiments/run_experiment.py \
    --exp e1 \
    --n_nodes $N_NODES \
    --global_rounds $GLOBAL_ROUNDS \
    --cluster_size $CLUSTER_SIZE \
    --pretrain_rounds $PRETRAIN_ROUNDS \
    --pretrain_dir $PRETRAIN_DIR \
    --ppo_device $PPO_DEVICE \
    $OFFLINE_ARG \
    --skip_baselines \
    --log_dir ./logs \
    --checkpoint_dir ./checkpoints

# =========================================================
# E2: Scalability
# =========================================================
echo "=========================================="
echo "Running E2: Scalability"
echo "=========================================="
python -u experiments/run_experiment.py \
    --exp e2 \
    --n_nodes $N_NODES \
    --global_rounds $GLOBAL_ROUNDS \
    --cluster_size $CLUSTER_SIZE \
    --pretrain_rounds $PRETRAIN_ROUNDS \
    --pretrain_dir $PRETRAIN_DIR \
    --ppo_device $PPO_DEVICE \
    $OFFLINE_ARG \
    --log_dir ./logs \
    --checkpoint_dir ./checkpoints

# =========================================================
# E3: Non-IID
# =========================================================
echo "=========================================="
echo "Running E3: Non-IID Data"
echo "=========================================="
python -u experiments/run_experiment.py \
    --exp e3 \
    --n_nodes $N_NODES \
    --global_rounds $GLOBAL_ROUNDS \
    --cluster_size $CLUSTER_SIZE \
    --alpha 0.1 \
    --pretrain_rounds $PRETRAIN_ROUNDS \
    --pretrain_dir $PRETRAIN_DIR \
    --ppo_device $PPO_DEVICE \
    $OFFLINE_ARG \
    --log_dir ./logs \
    --checkpoint_dir ./checkpoints

# =========================================================
# E4: Security
# =========================================================
echo "=========================================="
echo "Running E4: Security"
echo "=========================================="
python -u experiments/run_experiment.py \
    --exp e4 \
    --n_nodes $N_NODES \
    --global_rounds $GLOBAL_ROUNDS \
    --cluster_size $CLUSTER_SIZE \
    --lazy_fraction 0.2 \
    --pretrain_rounds $PRETRAIN_ROUNDS \
    --pretrain_dir $PRETRAIN_DIR \
    --ppo_device $PPO_DEVICE \
    $OFFLINE_ARG \
    --log_dir ./logs \
    --checkpoint_dir ./checkpoints

# =========================================================
# E5: Incentive
# =========================================================
echo "=========================================="
echo "Running E5: Incentive Mechanism"
echo "=========================================="
python -u experiments/run_experiment.py \
    --exp e5 \
    --n_nodes $N_NODES \
    --global_rounds $GLOBAL_ROUNDS \
    --cluster_size $CLUSTER_SIZE \
    --pretrain_rounds $PRETRAIN_ROUNDS \
    --pretrain_dir $PRETRAIN_DIR \
    --ppo_device $PPO_DEVICE \
    $OFFLINE_ARG \
    --log_dir ./logs \
    --checkpoint_dir ./checkpoints

# =========================================================
# E6: Ablation
# =========================================================
echo "=========================================="
echo "Running E6: Ablation Study"
echo "=========================================="
python -u experiments/run_experiment.py \
    --exp e6 \
    --n_nodes $N_NODES \
    --global_rounds $GLOBAL_ROUNDS \
    --cluster_size $CLUSTER_SIZE \
    --ablation full \
    --pretrain_rounds $PRETRAIN_ROUNDS \
    --pretrain_dir $PRETRAIN_DIR \
    --ppo_device $PPO_DEVICE \
    $OFFLINE_ARG \
    --log_dir ./logs \
    --checkpoint_dir ./checkpoints

# =========================================================
# E7: Blockchain Overhead
# =========================================================
echo "=========================================="
echo "Running E7: Blockchain Overhead"
echo "=========================================="
python -u experiments/run_experiment.py \
    --exp e7 \
    --n_nodes $N_NODES \
    --global_rounds $GLOBAL_ROUNDS \
    --cluster_size $CLUSTER_SIZE \
    --pretrain_rounds $PRETRAIN_ROUNDS \
    --pretrain_dir $PRETRAIN_DIR \
    --ppo_device $PPO_DEVICE \
    $OFFLINE_ARG \
    --log_dir ./logs \
    --checkpoint_dir ./checkpoints
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="