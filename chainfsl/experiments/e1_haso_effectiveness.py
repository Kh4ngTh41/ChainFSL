"""
E1: HASO Effectiveness - Straggler Analysis Experiment.

Hypothesis: HASO reduces training latency by avoiding stragglers vs static cut baselines.

This experiment tests 4 tier distribution scenarios × 2 methods:
1. Static Cut (baseline): All nodes use cut_layer=2 (exposes straggler problem)
2. HASO (ours): Each node optimizes cut_layer based on tier + network conditions

Key insight: When all nodes use the same static cut_layer, weak devices (tier3/4)
must reduce batch_size or crash → become stragglers. HASO avoids this by assigning
shallow cuts to weak devices.

Metrics:
- Mean round latency
- Straggler ratio (% nodes with latency > 1.5x mean)
- Final accuracy
- Time-to-accuracy (target: 60% accuracy)

Tier Distribution Scenarios:
- iot_heavy: 5% tier1, 10% tier2, 35% tier3, 50% tier4 (most challenging)
- balanced: 10% tier1, 30% tier2, 40% tier3, 20% tier4 (default)
- gpu_heavy: 30% tier1, 35% tier2, 25% tier3, 10% tier4 (easiest)
- uniform: 25% each tier
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

# Add project root
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.protocol.chainfsl import ChainFSLProtocol
from src.emulator.tier_factory import TIER_DISTRIBUTIONS, TierDistribution
from baselines import FedAvgBaseline
from experiments.utils import (
    build_config,
    save_results_csv,
    print_summary,
    ensure_dir,
)


TARGET_ACCURACY = 60.0  # Target accuracy % for time-to-accuracy measurement


def run(
    config: Dict[str, Any],
    tier_dist: str = "balanced",
    skip_baselines: bool = False,
    resume: bool = False,
    checkpoint_dir: str = "./checkpoints",
    pretrained_orchestrator=None,
    cluster_agent_pool=None,
    pretrain_dir: str = "pretrainppo",
) -> Dict[str, Any]:
    """
    Run E1 experiment with specified tier distribution.

    Args:
        config: Base config dict.
        tier_dist: Tier distribution name ("iot_heavy", "balanced", "gpu_heavy", "uniform").
        skip_baselines: If True, skip baseline comparisons.
        resume: If True, resume from latest checkpoint.
        checkpoint_dir: Directory for checkpoint files.
        pretrained_orchestrator: Pre-trained HASOOrchestrator (if available).
        cluster_agent_pool: Pre-trained ClusterAgentPool (if available).
        pretrain_dir: Directory containing pretrained models.

    Returns:
        Dict of results for each method.
    """
    # Get tier distribution
    tier_dist_obj = TIER_DISTRIBUTIONS.get(tier_dist)
    if tier_dist_obj is None:
        raise ValueError(f"Unknown tier_dist '{tier_dist}'. Available: {list(TIER_DISTRIBUTIONS.keys())}")

    print("=" * 60)
    print("E1: HASO Effectiveness - Straggler Analysis")
    print(f"Tier Distribution: {tier_dist}")
    print(f"Probabilities: {dict(zip(tier_dist_obj.tiers, tier_dist_obj.probabilities))}")
    print("=" * 60)

    results = {}

    # Update config with tier distribution
    exp_config = {
        **config,
        "tier_distribution": tier_dist_obj.probabilities,
    }

    # --- Method 1: Static Cut (baseline) ---
    # All nodes use cut_layer=2 regardless of tier.
    # Tier4/Tier3 devices will be slow due to OOM/batch_reduction → stragglers
    print(f"\n--- Static Cut (cut_layer=2 for all nodes) ---")
    metrics_static = _run_static_cut(
        {**exp_config, "haso_enabled": False},
    )
    results["static_cut"] = metrics_static
    save_results_csv(f"e1_{tier_dist}_static", metrics_static, config["log_dir"])
    print_summary("static_cut", metrics_static)

    # --- Method 2: HASO (ours) ---
    # Each node optimizes cut_layer based on tier and conditions
    print(f"\n--- HASO (adaptive cut per node) ---")
    metrics_haso = _run_chainfsl(
        {**exp_config, "haso_enabled": True, "tve_enabled": True, "gtm_enabled": True},
        pretrained_orchestrator=pretrained_orchestrator,
        cluster_agent_pool=cluster_agent_pool,
    )
    results["haso"] = metrics_haso
    save_results_csv(f"e1_{tier_dist}_haso", metrics_haso, config["log_dir"])
    print_summary("haso", metrics_haso)

    # --- Print comparison table ---
    _print_comparison_table(results, tier_dist, TARGET_ACCURACY)

    # Save combined results
    combined_path = Path(config["log_dir"]) / f"e1_{tier_dist}_comparison.csv"
    _save_combined_results(results, combined_path)

    return results


def _run_chainfsl(
    config: Dict[str, Any],
    pretrained_orchestrator=None,
    cluster_agent_pool=None,
) -> List[Dict[str, Any]]:
    """Run ChainFSL protocol with HASO."""
    import os

    log_dir = config.get("log_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)
    db_path = os.path.join(log_dir, f"chainfsl_e1_{config['seed']}.db")

    protocol = ChainFSLProtocol(
        config=config,
        device=None,
        db_path=db_path,
    )

    # Attach pretrained agents if available
    if cluster_agent_pool is not None:
        print(f"  [ChainFSL] Using pretrained cluster agent pool")
        protocol.cluster_agent_pool = cluster_agent_pool
    elif pretrained_orchestrator is not None:
        print(f"  [ChainFSL] Using pretrained orchestrator")
        protocol._orchestrator = pretrained_orchestrator

    metrics = protocol.run(
        total_rounds=config["global_rounds"],
        eval_every=5,
    )

    return [m.to_dict() for m in metrics]


def _run_static_cut(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Run ChainFSL with static cut_layer=2 for all nodes (no HASO).

    This baseline exposes the straggler problem:
    - Tier4/Tier3 devices with cut_layer=2 will have OOM issues
    - They must reduce batch_size → slower training
    - Results in high latency variance across nodes
    """
    import os

    # Force static cut by using haso_enabled=False
    # The memory constraint in protocol will automatically select deepest valid cut
    # For Tier4/Tier3: deepest valid is L1 (not L2)
    # → This creates the straggler effect!
    log_dir = config.get("log_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)
    db_path = os.path.join(log_dir, f"static_e1_{config['seed']}.db")

    # Create protocol with HASO disabled - it will use default L2 for all that can fit
    protocol = ChainFSLProtocol(
        config={**config, "haso_enabled": False},
        device=None,
        db_path=db_path,
    )

    # For Tier4/Tier3 nodes that can't fit L2, protocol falls back to L1
    # This is correct behavior for showing straggler effect

    metrics = protocol.run(
        total_rounds=config["global_rounds"],
        eval_every=5,
    )

    return [m.to_dict() for m in metrics]


def _print_comparison_table(results: Dict[str, Any], tier_dist: str, target_acc: float) -> None:
    """Print comparison table for E1."""
    print("\n" + "=" * 80)
    print(f"E1 COMPARISON TABLE - {tier_dist}")
    print("=" * 80)
    print(f"{'Method':<15} {'Final Acc':>10} {'Mean Latency':>12} {'Straggler%':>12} {'Fairness':>10}")
    print("-" * 80)

    for name, metrics in results.items():
        if not metrics:
            continue

        final_acc = metrics[-1].get("test_acc", 0)
        mean_lat = _mean([m.get("round_latency", 0) for m in metrics])
        straggler_ratio = _compute_straggler_ratio(metrics)
        fairness = _mean([m.get("fairness_index", 0) for m in metrics])

        print(f"{name:<15} {final_acc:>9.2f}% {mean_lat:>11.2f}s {straggler_ratio:>11.1f}% {fairness:>10.3f}")

    print("=" * 80)

    # Highlight key insight
    if "static_cut" in results and "haso" in results:
        static_lat = _mean([m.get("round_latency", 0) for m in results["static_cut"]])
        haso_lat = _mean([m.get("round_latency", 0) for m in results["haso"]])
        static_strag = _compute_straggler_ratio(results["static_cut"])
        haso_strag = _compute_straggler_ratio(results["haso"])

        lat_improvement = (static_lat - haso_lat) / static_lat * 100 if static_lat > 0 else 0
        strag_improvement = (static_strag - haso_strag) / static_strag * 100 if static_strag > 0 else 0

        print(f"\n📊 HASO vs Static Cut Improvement:")
        print(f"   Latency: {lat_improvement:+.1f}% ({static_lat:.2f}s → {haso_lat:.2f}s)")
        print(f"   Straggler Ratio: {strag_improvement:+.1f}% ({static_strag:.1f}% → {haso_strag:.1f}%)")


def _compute_straggler_ratio(metrics: List[Dict[str, Any]], threshold: float = 1.5) -> float:
    """
    Compute straggler ratio: fraction of rounds where a node exceeds threshold × mean latency.

    A node is a "straggler" if its per-node latency > threshold × mean_latency.
    """
    if not metrics:
        return 0.0

    # Collect all per-node latencies
    node_latencies = []
    for m in metrics:
        # Each round may have per-node latency breakdown
        for node_id, lat in m.get("per_node_latency", {}).items():
            node_latencies.append(lat)

    if not node_latencies:
        return 0.0

    mean_lat = sum(node_latencies) / len(node_latencies)
    threshold_value = mean_lat * threshold
    stragglers = sum(1 for lat in node_latencies if lat > threshold_value)
    return (stragglers / len(node_latencies)) * 100


def _save_combined_results(results: Dict[str, Any], path: Path) -> None:
    """Save combined results from all methods."""
    ensure_dir(str(path.parent))

    import csv

    all_keys = {"method"}
    for metrics in results.values():
        for m in metrics:
            all_keys.update(m.keys())

    sorted_keys = sorted(all_keys, key=lambda k: (k == "method", k))

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted_keys)
        writer.writeheader()
        for name, metrics in results.items():
            for m in metrics:
                writer.writerow({**m, "method": name})

    print(f"[E1] Combined results saved to: {path}")


def _mean(values: list) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


# ============================================================================
# E1 Full Suite: Run all 4 tier distribution scenarios
# ============================================================================

def run_all_scenarios(
    config: Dict[str, Any],
    pretrained_orchestrator=None,
    cluster_agent_pool=None,
    pretrain_dir: str = "pretrainppo",
) -> Dict[str, Dict[str, Any]]:
    """
    Run E1 across all 4 tier distribution scenarios.

    This is the full E1 experiment suite as specified in the plan:
    - iot_heavy, balanced, gpu_heavy, uniform

    Args:
        config: Base config dict.
        pretrained_orchestrator: Pre-trained HASOOrchestrator.
        cluster_agent_pool: Pre-trained ClusterAgentPool.
        pretrain_dir: Directory containing pretrained models.

    Returns:
        Dict mapping scenario_name → results_dict.
    """
    all_results = {}

    for tier_dist in ["iot_heavy", "balanced", "gpu_heavy", "uniform"]:
        print("\n" + "=" * 80)
        print(f"E1 FULL SUITE: Testing {tier_dist}")
        print("=" * 80)

        results = run(
            config=config,
            tier_dist=tier_dist,
            skip_baselines=False,
            pretrained_orchestrator=pretrained_orchestrator,
            cluster_agent_pool=cluster_agent_pool,
            pretrain_dir=pretrain_dir,
        )
        all_results[tier_dist] = results

    # Print final summary across all scenarios
    _print_all_scenarios_summary(all_results)

    return all_results


def _print_all_scenarios_summary(all_results: Dict[str, Dict[str, Any]]) -> None:
    """Print summary table across all scenarios."""
    print("\n" + "=" * 100)
    print("E1 FULL SUITE SUMMARY - All 4 Tier Distribution Scenarios")
    print("=" * 100)
    print(f"{'Scenario':<12} {'Method':<12} {'Final Acc':>10} {'Mean Lat':>10} {'Strag%':>8} {'Fairness':>10}")
    print("-" * 100)

    for scenario, results in all_results.items():
        for method, metrics in results.items():
            if not metrics:
                continue
            final_acc = metrics[-1].get("test_acc", 0)
            mean_lat = _mean([m.get("round_latency", 0) for m in metrics])
            strag = _compute_straggler_ratio(metrics)
            fairness = _mean([m.get("fairness_index", 0) for m in metrics])
            print(f"{scenario:<12} {method:<12} {final_acc:>9.2f}% {mean_lat:>9.2f}s {strag:>7.1f}% {fairness:>10.3f}")

    print("=" * 100)