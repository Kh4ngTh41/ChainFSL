"""
Shared utilities for experiments.

Provides common helpers for:
- Config building from YAML
- CSV logging
- Result loading/saving
- Experiment runner scaffolding
"""

import os
import sys
import json
import csv
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from multiprocessing import cpu_count

import numpy as np

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml

from src.utils.metrics import compute_metrics, jains_fairness, gini_coefficient


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def merge_config(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge override dict into base config."""
    result = base.copy()
    for key, value in overrides.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


def build_config(
    n_nodes: int = 10,
    global_rounds: int = 30,
    dataset: str = "cifar10",
    haso_enabled: bool = True,
    tve_enabled: bool = True,
    gtm_enabled: bool = True,
    lazy_fraction: float = 0.0,
    dirichlet_alpha: float = 0.5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Build a standard config dict for experiments.

    All experiment configs inherit from this base to ensure consistency.
    """
    return {
        "n_nodes": n_nodes,
        "tier_distribution": [0.1, 0.3, 0.4, 0.2],
        "dataset": dataset,
        "n_classes": 10,
        "global_rounds": global_rounds,
        "batch_size_default": 32,
        "label_smoothing": 0.1,
        "dirichlet_alpha": dirichlet_alpha,
        "local_epochs": 1,
        "sample_fraction": 1.0,
        "haso_enabled": haso_enabled,
        "haso_online_update": True,
        "ppo_device": "auto",
        "ppo_learning_rate": 3e-4,
        "ppo_n_steps": 256,
        "ppo_batch_size": 64,
        "ppo_update_timesteps": 256,  # timesteps per RL update per round (was 64, too little)
        "reward_alpha": 1.0,
        "reward_beta": 0.5,
        "reward_gamma": 0.1,
        "reward_latency_penalty_weight": 0.0,
        "reward_penalty_source": "train_time",  # train_time | round_latency
        "straggler_fraction": 0.0,
        "straggler_slowdown_factor": 3,
        "ema_beta": 0.9,
        "tve_enabled": tve_enabled,
        "committee_size": max(3, n_nodes // 10),
        "vrf_omega": 0.3,
        "stake_min": 10.0,
        "gtm_enabled": gtm_enabled,
        "shapley_M": max(20, n_nodes * 3),
        "reward_total_init": 1000.0,
        "reward_min": 10.0,
        "halving_rounds": 50,
        "staleness_decay": 0.9,
        "lazy_client_fraction": lazy_fraction,
        "sybil_fraction": 0.0,
        "poison_fraction": 0.0,
        "log_dir": "./logs",
        "use_wandb": False,
        "experiment_name": "chainfsl_default",
        "seed": seed,
    }


def ensure_dir(path: str) -> Path:
    """Ensure directory exists."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def csv_path(exp_name: str, log_dir: str = "./logs") -> Path:
    """Get CSV path for an experiment."""
    return ensure_dir(log_dir) / f"{exp_name}_results.csv"


def save_results_csv(
    exp_name: str,
    metrics: List[Dict[str, Any]],
    log_dir: str = "./logs",
) -> Path:
    """
    Save experiment results to CSV.

    Args:
        exp_name: Experiment name (e.g., "e1_haso").
        metrics: List of per-round metric dicts.
        log_dir: Directory to save CSV.

    Returns:
        Path to saved CSV file.
    """
    if not metrics:
        return csv_path(exp_name, log_dir)

    p = csv_path(exp_name, log_dir)
    keys = list(metrics[0].keys())

    with open(p, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metrics)

    return p


def load_results_csv(exp_name: str, log_dir: str = "./logs") -> List[Dict[str, Any]]:
    """Load results from CSV file."""
    p = csv_path(exp_name, log_dir)
    if not p.exists():
        return []

    with open(p) as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def summary_stats(metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute summary statistics from metrics history.

    Args:
        metrics: List of metric dicts (one per round).

    Returns:
        Summary dict with mean, std, min, max for numeric fields.
    """
    if not metrics:
        return {}

    # Convert to numpy arrays per key
    numeric_keys = [k for k in metrics[0].keys() if k != "round"]
    result = {}

    for key in numeric_keys:
        values = []
        for m in metrics:
            try:
                values.append(float(m[key]))
            except (ValueError, TypeError):
                continue

        if values:
            arr = np.array(values)
            result[f"{key}_mean"] = float(np.mean(arr))
            result[f"{key}_std"] = float(np.std(arr))
            result[f"{key}_min"] = float(np.min(arr))
            result[f"{key}_max"] = float(np.max(arr))

    # Explicit final metric snapshots for downstream consumers.
    for key in numeric_keys:
        try:
            result[f"final_{key}"] = float(metrics[-1][key])
        except (ValueError, TypeError, KeyError):
            continue

    return result


def print_summary(exp_name: str, metrics: List[Dict[str, Any]]) -> None:
    """Print a formatted summary of experiment results."""
    if not metrics:
        print(f"[{exp_name}] No results to summarize.")
        return

    stats = summary_stats(metrics)
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'=' * 60}")

    # Key metrics
    try:
        final_test_acc = float(metrics[-1].get("test_acc", 0.0))
        print(f"  Final Test Acc: {final_test_acc:.2f}%")
    except (ValueError, TypeError, KeyError, IndexError):
        pass
    if "test_acc_mean" in stats:
        print(f"  Mean Test Acc:  {stats['test_acc_mean']:.2f}% ± {stats['test_acc_std']:.2f}%")
    if "train_loss_mean" in stats:
        print(f"  Train Loss:    {stats['train_loss_mean']:.4f} ± {stats['train_loss_std']:.4f}")
    if "round_latency_mean" in stats:
        print(f"  Latency:       {stats['round_latency_mean']:.2f}s ± {stats['round_latency_std']:.2f}s")
    if "train_only_latency_mean" in stats:
        print(
            f"  Train-Only:    {stats['train_only_latency_mean']:.2f}s "
            f"± {stats['train_only_latency_std']:.2f}s"
        )
    if "ppo_update_time_mean" in stats:
        print(
            f"  PPO Update:    {stats['ppo_update_time_mean']:.2f}s "
            f"± {stats['ppo_update_time_std']:.2f}s"
        )
    if "fairness_index_mean" in stats:
        print(f"  Fairness:      {stats['fairness_index_mean']:.3f} ± {stats['fairness_index_std']:.3f}")
    if "total_reward_mean" in stats:
        print(f"  Total Reward:   {stats['total_reward_mean']:.2f}")
    if "attack_detection_rate_mean" in stats:
        print(f"  Detection:     {stats['attack_detection_rate_mean']:.2%}")

    print(f"  Rounds:        {len(metrics)}")
    print(f"{'=' * 60}\n")


def get_timestamp() -> str:
    """Get timestamp string for experiment naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def config_fingerprint(config: Dict[str, Any]) -> str:
    """Generate a short hash fingerprint of a config dict."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def run_experiment(
    exp_name: str,
    run_fn: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
    config: Dict[str, Any],
    log_dir: str = "./logs",
    save_csv: bool = True,
    print_summary_flag: bool = True,
) -> Dict[str, Any]:
    """
    Scaffold for running an experiment.

    Args:
        exp_name: Experiment name (e.g., "e1_haso").
        run_fn: Function that takes config and returns metrics list.
        config: Config dict.
        log_dir: Directory for logs/CSVs.
        save_csv: Whether to save CSV.
        print_summary_flag: Whether to print summary.

    Returns:
        Summary dict.
    """
    print(f"\n[{exp_name}] Starting experiment...")
    print(f"[{exp_name}] Config: n_nodes={config['n_nodes']}, "
          f"rounds={config['global_rounds']}, "
          f"alpha={config.get('dirichlet_alpha', 'N/A')}")

    start = time.perf_counter()

    # Run experiment
    metrics = run_fn(config)

    elapsed = time.perf_counter() - start

    # Save CSV
    if save_csv and metrics:
        csv_p = save_results_csv(exp_name, metrics, log_dir)
        print(f"[{exp_name}] Results saved to: {csv_p}")

    # Print summary
    if print_summary_flag:
        print_summary(exp_name, metrics)

    print(f"[{exp_name}] Completed in {elapsed:.1f}s")

    # Return summary
    summary = summary_stats(metrics)
    summary["elapsed_seconds"] = elapsed
    summary["n_rounds"] = len(metrics)

    return summary
