#!/usr/bin/env python3
"""
E1 Full Baseline Comparison - All 6 Methods × 3 Cluster Scenarios
20 nodes, 50 rounds

Usage:
    python experiments/run_e1_full_comparison.py
    python experiments/run_e1_full_comparison.py --method haso_per_node --scenario 20pct
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.e1_baseline_comparison import (
    METHODS, CLUSTER_SCENARIOS, run, run_e1_suite,
    compute_cluster_size, save_results_csv,
)

N_NODES = 20
GLOBAL_ROUNDS = 50
DATASET = "cifar10"
TIER_DIST = "balanced"
SEEDS = [42]


def run_single_method(method: str, scenario: str = None, verbose: bool = True):
    """Run a single method across all scenarios or a specific scenario."""
    from experiments.utils import build_config
    from src.emulator.tier_factory import TIER_DISTRIBUTIONS

    tier_dist_obj = TIER_DISTRIBUTIONS.get(TIER_DIST)
    if tier_dist_obj is None:
        raise ValueError(f"Unknown tier_dist '{TIER_DIST}'")

    config = build_config(
        n_nodes=N_NODES,
        global_rounds=GLOBAL_ROUNDS,
        dataset=DATASET,
        haso_enabled=True,
        tve_enabled=True,
        gtm_enabled=True,
    )
    config["tier_distribution"] = tier_dist_obj.probabilities
    config["log_dir"] = "./logs"
    config["seed"] = 42

    results = {}

    scenarios = {scenario: CLUSTER_SCENARIOS[scenario]} if scenario else CLUSTER_SCENARIOS

    for ratio_key, ratio_val in scenarios.items():
        exp_name = f"e1_{method}_{ratio_key}_s{42}"

        print(f"\n{'#' * 60}")
        print(f"# Running: {exp_name}")
        print(f"# Method: {method}, Scenario: {ratio_key}")
        print(f"{'#' * 60}")

        try:
            metrics = run(
                config={**config},
                method=method,
                cluster_ratio=ratio_val,
                verbose=verbose,
            )
            results[ratio_key] = metrics

            # Save CSV
            save_results_csv(exp_name, metrics, config["log_dir"])
            print(f"[OK] {exp_name} completed - {len(metrics)} rounds")

            # Print summary
            if metrics:
                final_acc = metrics[-1].get("test_acc", 0.0)
                mean_lat = sum(m.get("round_latency", 0) for m in metrics) / len(metrics)
                mean_fair = sum(m.get("fairness_index", 0) for m in metrics) / len(metrics)
                print(f"    Final Acc: {final_acc:.2f}%, Mean Lat: {mean_lat:.2f}s, Fairness: {mean_fair:.3f}")

        except Exception as e:
            print(f"[ERROR] {exp_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results[ratio_key] = []

    return results


def run_all_methods():
    """Run all 6 methods × 3 scenarios = 18 experiments."""
    from experiments.utils import build_config
    from src.emulator.tier_factory import TIER_DISTRIBUTIONS

    tier_dist_obj = TIER_DISTRIBUTIONS.get(TIER_DIST)
    config = build_config(
        n_nodes=N_NODES,
        global_rounds=GLOBAL_ROUNDS,
        dataset=DATASET,
        haso_enabled=True,
        tve_enabled=True,
        gtm_enabled=True,
    )
    config["tier_distribution"] = tier_dist_obj.probabilities
    config["log_dir"] = "./logs"

    all_results = {}

    for method in METHODS.keys():
        print(f"\n{'=' * 70}")
        print(f"METHOD: {method}")
        print(f"{'=' * 70}")

        method_results = {}
        for ratio_key, ratio_val in CLUSTER_SCENARIOS.items():
            exp_name = f"e1_{method}_{ratio_key}_s{42}"

            print(f"\n--- {exp_name} ---")

            try:
                metrics = run(
                    config={**config, "seed": 42},
                    method=method,
                    cluster_ratio=ratio_val,
                    verbose=True,
                )
                method_results[ratio_key] = metrics
                save_results_csv(exp_name, metrics, config["log_dir"])

                if metrics:
                    final_acc = metrics[-1].get("test_acc", 0.0)
                    mean_lat = sum(m.get("round_latency", 0) for m in metrics) / len(metrics)
                    print(f"  [OK] Acc={final_acc:.2f}%, Lat={mean_lat:.2f}s")

            except Exception as e:
                print(f"  [ERROR] {e}")
                method_results[ratio_key] = []

        all_results[method] = method_results

    # Print summary table
    print("\n" + "=" * 100)
    print("E1 FULL BASELINE COMPARISON SUMMARY")
    print(f"Config: n_nodes={N_NODES}, rounds={GLOBAL_ROUNDS}, dataset={DATASET}")
    print("=" * 100)
    print(f"{'Method':<18} {'Scenario':<10} {'Final Acc':>10} {'Mean Lat':>10} {'Fairness':>10}")
    print("-" * 100)

    for method in METHODS.keys():
        if method not in all_results:
            continue
        for scenario in CLUSTER_SCENARIOS.keys():
            metrics_list = all_results[method].get(scenario, [])
            if not metrics_list:
                print(f"{method:<18} {scenario:<10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
                continue

            final_acc = metrics_list[-1].get("test_acc", 0.0) if metrics_list else 0.0
            mean_lat = sum(m.get("round_latency", 0) for m in metrics_list) / len(metrics_list)
            mean_fair = sum(m.get("fairness_index", 0) for m in metrics_list) / len(metrics_list)
            print(f"{method:<18} {scenario:<10} {final_acc:>9.2f}% {mean_lat:>9.2f}s {mean_fair:>10.3f}")

    print("=" * 100)
    return all_results


def main():
    parser = argparse.ArgumentParser(description="E1 Full Baseline Comparison")
    parser.add_argument("--method", choices=list(METHODS.keys()), help="Run specific method only")
    parser.add_argument("--scenario", choices=list(CLUSTER_SCENARIOS.keys()), help="Run specific scenario only")
    parser.add_argument("--list", action="store_true", help="List all methods and scenarios")
    args = parser.parse_args()

    if args.list:
        print("Available methods:")
        for m, cfg in METHODS.items():
            print(f"  {m}: {cfg['description']}")
        print("\nAvailable scenarios:")
        for s, r in CLUSTER_SCENARIOS.items():
            cluster_size = compute_cluster_size(N_NODES, r)
            print(f"  {s}: ratio={r:.2f} → {cluster_size} nodes/cluster")
        return

    print(f"E1 Full Baseline Comparison")
    print(f"Config: n_nodes={N_NODES}, rounds={GLOBAL_ROUNDS}, dataset={DATASET}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    if args.method:
        results = run_single_method(args.method, args.scenario)
    else:
        results = run_all_methods()

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Done!")


if __name__ == "__main__":
    main()