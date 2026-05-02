#!/bin/bash
#SBATCH --job-name=e1_full_comparison
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/home/uitchain/datastore/khang/ChainFSL/logs/e1_full_%j.out
#SBATCH --error=/home/uitchain/datastore/khang/ChainFSL/logs/e1_full_%j.err

set -euo pipefail

# =========================================================
# E1 Full Baseline Comparison - All 6 Methods × 3 Scenarios
# 18 experiments total: 50 rounds each
# Expected runtime: ~18-24 hours
# =========================================================

N_NODES=20
GLOBAL_ROUNDS=50
PPO_DEVICE=cpu
VENV_DIR="/home/uitchain/datastore/khang/ChainFSL/chainfsl/.venv"
REQ_FILE="/home/uitchain/datastore/khang/ChainFSL/chainfsl/requirements.txt"
LOG_DIR="/home/uitchain/datastore/khang/ChainFSL/chainfsl/logs/e1_full_experiment"

echo "[SLURM] E1 Full Baseline Comparison - $(date)"
echo "=========================================="
echo "  Methods: haso_per_node, haso_cluster, haso_centralized, fedavg, splitfed_v1, splitfed_v2"
echo "  Scenarios: 10pct, 20pct, 30pct"
echo "  Total: 6 methods × 3 scenarios = 18 experiments"
echo "  Rounds per experiment: $GLOBAL_ROUNDS"
echo "=========================================="

module load python312

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

if [ -f "$VENV_DIR/.chainfsl_env_ready" ]; then
    echo "[ENV] Reusing existing .venv"
else
    echo "[ENV] Installing dependencies..."
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r $REQ_FILE
    touch "$VENV_DIR/.chainfsl_env_ready"
fi

# GPU setup - Use A100 directly without probe (probe fails on A100)
export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS:-0}"
echo "[GPU] Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[GPU] GPU count: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"

cd /home/uitchain/datastore/khang/ChainFSL/chainfsl

# Create log directory
mkdir -p "$LOG_DIR"

echo "[START] E1 Full Experiment - $(date)"

# Run the full experiment suite
python -u experiments/run_e1_full_logging.py

echo "[DONE] E1 Full Experiment - $(date)"
echo "Logs saved to: $LOG_DIR"
echo "=========================================="