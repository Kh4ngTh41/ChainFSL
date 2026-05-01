"""
E1 Baseline Comparison Runner — All 6 Methods × 3 Cluster Scenarios.

This experiment compares all 6 methods:
  HASO methods (via ChainFSLProtocol):
    - haso_per_node: Per-node architecture, each node gets its own HASO agent
    - haso_cluster: Cluster architecture, k nodes form a cluster coordinated by cluster agent
    - haso_centralized: Centralized architecture, single orchestrator controls all nodes

  Baseline methods:
    - fedavg: Standard federated averaging (McMhan et al., 2017)
    - splitfed_v1: SplitFed with uniform cut_layer=2 (Singh et al., 2019)
    - splitfed_v2: SplitFed with tier-adaptive cut layer selection

Cluster scenarios (fraction of nodes per cluster):
    - 10pct: 10% of nodes per cluster (e.g., 2 nodes for N=20)
    - 20pct: 20% of nodes per cluster (e.g., 4 nodes for N=20)
    - 30pct: 30% of nodes per cluster (e.g., 6 nodes for N=20)

Metrics per method × scenario:
    - Final test accuracy
    - Mean round latency
    - Fairness index (Jain's)
    - Time-to-accuracy (target accuracy threshold)
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Add project root
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from src.protocol.chainfsl import ChainFSLProtocol
from src.emulator.tier_factory import TIER_DISTRIBUTIONS, TierDistribution
from baselines import FedAvgBaseline
from baselines.splitfed import SplitFedBaseline
from baselines.splitfed_v2 import SplitFedV2Baseline
from experiments.utils import (
    build_config,
    save_results_csv,
    print_summary,
    ensure_dir,
)


# =============================================================================
# Method Configurations
# =============================================================================

METHODS: Dict[str, Dict[str, Any]] = {
    # HASO methods — use ChainFSLProtocol with different arch_mode
    "haso_per_node": {
        "arch_mode": "per_node",
        "haso_enabled": True,
        "description": "HASO per-node architecture",
    },
    "haso_cluster": {
        "arch_mode": "cluster",
        "haso_enabled": True,
        "description": "HASO cluster architecture (k nodes per cluster)",
    },
    "haso_centralized": {
        "arch_mode": "centralized",
        "haso_enabled": True,
        "description": "HASO centralized architecture",
    },
    # Baseline methods — no HASO
    "fedavg": {
        "arch_mode": None,
        "haso_enabled": False,
        "description": "Standard FedAvg (no split learning)",
    },
    "splitfed_v1": {
        "arch_mode": None,
        "haso_enabled": False,
        "description": "SplitFed with uniform cut_layer=2",
    },
    "splitfed_v2": {
        "arch_mode": None,
        "haso_enabled": False,
        "description": "SplitFed with tier-adaptive cut layer",
    },
}


# =============================================================================
# Cluster Scenarios
# =============================================================================

CLUSTER_SCENARIOS: Dict[str, float] = {
    "10pct": 0.10,  # 10% of nodes per cluster
    "20pct": 0.20,  # 20% of nodes per cluster
    "30pct": 0.30,  # 30% of nodes per cluster
}


# =============================================================================
# Helper Functions
# =============================================================================

def compute_cluster_size(n_nodes: int, ratio: float) -> int:
    """
    Compute cluster size from number of nodes and ratio.

    For n_nodes=20:
        ratio=0.10 → cluster_size=2
        ratio=0.20 → cluster_size=4
        ratio=0.30 → cluster_size=6

    Handles divisibility by rounding to nearest integer.

    Args:
        n_nodes: Total number of nodes in the system.
        ratio: Fraction of nodes per cluster (0.0 to 1.0).

    Returns:
        Cluster size (at least 1, at most n_nodes).
    """
    raw = max(1.0, n_nodes * ratio)
    cluster_size = max(1, min(n_nodes, int(raw + 0.5)))  # Round to nearest
    return cluster_size


def _mean(values: List[float]) -> float:
    """Compute mean of a list, return 0.0 for empty list."""
    if not values:
        return 0.0
    return sum(values) / len(values)


# =============================================================================
# Per-Method Runners
# =============================================================================

def _run_chainfsl(
    config: Dict[str, Any],
    arch_mode: str,
    cluster_size: int,
    n_nodes: int,
) -> List[Dict[str, Any]]:
    """
    Run ChainFSL protocol with HASO enabled.

    Args:
        config: Base configuration dict.
        arch_mode: Architecture mode — "per_node", "cluster", or "centralized".
        cluster_size: Number of nodes per cluster (only used for cluster mode).
        n_nodes: Total number of nodes.

    Returns:
        List of per-round metric dicts.
    """
    log_dir = config.get("log_dir", "./logs")
    seed = config.get("seed", 42)
    os.makedirs(log_dir, exist_ok=True)

    db_path = os.path.join(log_dir, f"chainfsl_{arch_mode}_s{seed}.db")

    # Build config for ChainFSL
    chainfsl_config = {
        **config,
        "arch_mode": arch_mode,
        "haso_enabled": True,
        "tve_enabled": config.get("tve_enabled", True),
        "gtm_enabled": config.get("gtm_enabled", True),
    }

    # For cluster mode, add cluster_size
    if arch_mode == "cluster":
        chainfsl_config["cluster_size"] = cluster_size

    protocol = ChainFSLProtocol(
        config=chainfsl_config,
        device=None,
        db_path=db_path,
    )

    metrics = protocol.run(
        total_rounds=config["global_rounds"],
        eval_every=5,
    )

    return [m.to_dict() for m in metrics]


def _run_fedavg(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Run FedAvg baseline.

    Args:
        config: Configuration dict.

    Returns:
        List of per-round metric dicts.
    """
    baseline = FedAvgBaseline(config=config)
    metrics = baseline.run()
    return metrics


def _run_splitfed_v1(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Run SplitFedV1 baseline (uniform cut_layer=2).

    Args:
        config: Configuration dict.

    Returns:
        List of per-round metric dicts.
    """
    baseline = SplitFedBaseline(config=config, cut_layer=2)
    metrics = baseline.run()
    return metrics


def _run_splitfed_v2(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Run SplitFedV2 baseline (tier-adaptive cut layer).

    Args:
        config: Configuration dict.

    Returns:
        List of per-round metric dicts.
    """
    baseline = SplitFedV2Baseline(config=config)
    metrics = baseline.run()
    return metrics


# =============================================================================
# Main Run Function
# =============================================================================

def run(
    config: Dict[str, Any],
    method: str,
    cluster_ratio: float = 0.0,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run a single method with specified cluster_ratio.

    Args:
        config: Base configuration dict.
        method: Method name — one of METHODS keys.
        cluster_ratio: Ratio of nodes per cluster (for cluster mode).
                     Set to 0.0 for non-cluster modes.
        verbose: If True, print progress.

    Returns:
        List of per-round metric dicts.

    Raises:
        ValueError: If method is not a valid METHODS key.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Available: {list(METHODS.keys())}")

    method_cfg = METHODS[method]
    arch_mode = method_cfg.get("arch_mode")
    n_nodes = config["n_nodes"]

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Running method: {method}")
        print(f"  arch_mode: {arch_mode}")
        if arch_mode == "cluster":
            cluster_size = compute_cluster_size(n_nodes, cluster_ratio)
            print(f"  cluster_ratio: {cluster_ratio:.2f} → cluster_size: {cluster_size}")
        print(f"{'=' * 60}")

    # Dispatch to appropriate runner
    if method.startswith("haso_"):
        # HASO methods — use ChainFSL
        cluster_size = compute_cluster_size(n_nodes, cluster_ratio) if arch_mode == "cluster" else 0
        return _run_chainfsl(
            config=config,
            arch_mode=arch_mode,
            cluster_size=cluster_size,
            n_nodes=n_nodes,
        )
    elif method == "fedavg":
        return _run_fedavg(config)
    elif method == "splitfed_v1":
        return _run_splitfed_v1(config)
    elif method == "splitfed_v2":
        return _run_splitfed_v2(config)
    else:
        raise ValueError(f"Unknown method '{method}'")


# =============================================================================
# Run All Combinations
# =============================================================================

def run_all_combinations(
    config: Dict[str, Any],
    methods: List[str],
    cluster_ratios: List[float],
    seeds: List[int],
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Run all combinations of methods × cluster_ratios × seeds.

    Args:
        config: Base configuration dict.
        methods: List of method names to run.
        cluster_ratios: List of cluster ratios to test.
        seeds: List of random seeds for repetition.
        verbose: If True, print progress.

    Returns:
        Nested dict: results[method][scenario][seed] = metrics_list
    """
    results: Dict[str, Dict[str, Any]] = {}

    for method in methods:
        results[method] = {}

        for ratio_key, ratio_val in CLUSTER_SCENARIOS.items():
            if ratio_key not in cluster_ratios:
                continue

            results[method][ratio_key] = {}

            for seed in seeds:
                # Update config with seed
                exp_config = {**config, "seed": seed}

                # Build experiment name
                exp_name = f"e1_{method}_{ratio_key}_s{seed}"

                if verbose:
                    print(f"\n{'=' * 60}")
                    print(f"EXPERIMENT: {exp_name}")
                    print(f"  Method: {method}")
                    print(f"  Cluster scenario: {ratio_key} (ratio={ratio_val:.2f})")
                    print(f"  Seed: {seed}")
                    print(f"{'=' * 60}")

                # Run
                try:
                    metrics = run(
                        config=exp_config,
                        method=method,
                        cluster_ratio=ratio_val,
                        verbose=verbose,
                    )
                    results[method][ratio_key][str(seed)] = metrics

                    # Save CSV
                    log_dir = exp_config.get("log_dir", "./logs")
                    save_results_csv(exp_name, metrics, log_dir)

                except Exception as e:
                    print(f"[ERROR] {exp_name} failed: {e}")
                    results[method][ratio_key][str(seed)] = []

    return results


# =============================================================================
# Comparison Table
# =============================================================================

def print_comparison_table(
    all_results: Dict[str, Dict[str, Any]],
    target_acc: float = 60.0,
) -> None:
    """
    Print summary comparison table across all method × scenario combinations.

    Args:
        all_results: Nested dict from run_all_combinations.
        target_acc: Target accuracy for time-to-accuracy metric.
    """
    print("\n" + "=" * 100)
    print("E1 BASELINE COMPARISON — ALL METHODS × CLUSTER SCENARIOS")
    print("=" * 100)
    print(f"{'Method':<18} {'Scenario':<10} {'Final Acc':>10} {'Mean Lat':>10} {'Fairness':>10} {'Time-to-Acc':>12}")
    print("-" * 100)

    for method in METHODS.keys():
        if method not in all_results:
            continue

        for scenario in CLUSTER_SCENARIOS.keys():
            if scenario not in all_results[method]:
                continue

            # Aggregate across seeds
            seed_metrics = all_results[method][scenario]
            if not seed_metrics:
                continue

            # Compute mean across seeds
            final_accs = []
            mean_lats = []
            fairness_vals = []
            time_to_accs = []

            for seed_str, metrics in seed_metrics.items():
                if not metrics:
                    continue

                # Final accuracy (last round)
                final_acc = metrics[-1].get("test_acc", 0.0) if metrics else 0.0
                final_accs.append(final_acc)

                # Mean latency
                latencies = [m.get("round_latency", 0.0) for m in metrics]
                mean_lat = _mean(latencies)
                mean_lats.append(mean_lat)

                # Fairness (mean of per-round fairness_index)
                fairness = _mean([m.get("fairness_index", 0.0) for m in metrics])
                fairness_vals.append(fairness)

                # Time-to-accuracy (first round to reach target_acc)
                tta = _time_to_accuracy(metrics, target_acc)
                time_to_accs.append(tta)

            # Mean across seeds
            avg_final_acc = _mean(final_accs)
            avg_mean_lat = _mean(mean_lats)
            avg_fairness = _mean(fairness_vals)
            avg_tta = _mean(time_to_accs)

            tta_str = f"{avg_tta:.1f}r" if avg_tta > 0 else "N/A"

            print(f"{method:<18} {scenario:<10} {avg_final_acc:>9.2f}% {avg_mean_lat:>9.2f}s {avg_fairness:>10.3f} {tta_str:>12}")

    print("=" * 100)


def _time_to_accuracy(metrics: List[Dict[str, Any]], target_acc: float) -> float:
    """
    Compute time-to-accuracy: first round where test_acc >= target_acc.

    Args:
        metrics: List of per-round metrics.
        target_acc: Target accuracy threshold.

    Returns:
        Round number where target was first reached, or 0.0 if never reached.
    """
    for m in metrics:
        if m.get("test_acc", 0.0) >= target_acc:
            return float(m.get("round", 0))
    return 0.0  # Never reached


# =============================================================================
# E1 Suite Entry Point
# =============================================================================

def run_e1_suite(
    n_nodes: int = 20,
    global_rounds: int = 30,
    dataset: str = "cifar10",
    methods: Optional[List[str]] = None,
    cluster_ratios: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    tier_dist: str = "balanced",
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Run the full E1 baseline comparison suite.

    Args:
        n_nodes: Number of nodes (default 20).
        global_rounds: Number of federated rounds (default 30).
        dataset: Dataset name (default "cifar10").
        methods: List of method names to test. If None, test all 6.
        cluster_ratios: List of cluster scenario keys. If None, test all 3.
        seeds: List of random seeds. If None, use [42].
        tier_dist: Tier distribution name (default "balanced").
        verbose: If True, print progress.

    Returns:
        Nested dict of results: results[method][scenario][seed] = metrics_list
    """
    if methods is None:
        methods = list(METHODS.keys())

    if cluster_ratios is None:
        cluster_ratios = list(CLUSTER_SCENARIOS.keys())

    if seeds is None:
        seeds = [42]

    # Get tier distribution
    tier_dist_obj = TIER_DISTRIBUTIONS.get(tier_dist)
    if tier_dist_obj is None:
        raise ValueError(f"Unknown tier_dist '{tier_dist}'. Available: {list(TIER_DISTRIBUTIONS.keys())}")

    # Build base config
    config = build_config(
        n_nodes=n_nodes,
        global_rounds=global_rounds,
        dataset=dataset,
        haso_enabled=True,
        tve_enabled=True,
        gtm_enabled=True,
    )
    config["tier_distribution"] = tier_dist_obj.probabilities

    if verbose:
        print("=" * 80)
        print("E1 BASELINE COMPARISON SUITE")
        print(f"N_nodes: {n_nodes}")
        print(f"Global rounds: {global_rounds}")
        print(f"Dataset: {dataset}")
        print(f"Tier distribution: {tier_dist}")
        print(f"Methods: {methods}")
        print(f"Cluster scenarios: {cluster_ratios}")
        print(f"Seeds: {seeds}")
        print("=" * 80)

    # Run all combinations
    results = run_all_combinations(
        config=config,
        methods=methods,
        cluster_ratios=cluster_ratios,
        seeds=seeds,
        verbose=verbose,
    )

    # Print comparison table
    print_comparison_table(results, target_acc=60.0)

    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="E1 Baseline Comparison Runner")
    parser.add_argument("--n_nodes", type=int, default=20, help="Number of nodes")
    parser.add_argument("--global_rounds", type=int, default=30, help="Number of rounds")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name")
    parser.add_argument("--tier_dist", type=str, default="balanced", help="Tier distribution")
    parser.add_argument("--methods", type=str, nargs="+", default=None, help="Methods to run")
    parser.add_argument("--scenarios", type=str, nargs="+", default=None, help="Cluster scenarios")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Random seeds")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Log directory")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print progress")

    args = parser.parse_args()

    run_e1_suite(
        n_nodes=args.n_nodes,
        global_rounds=args.global_rounds,
        dataset=args.dataset,
        tier_dist=args.tier_dist,
        methods=args.methods,
        cluster_ratios=args.scenarios,
        seeds=args.seeds,
        verbose=args.verbose,
    )