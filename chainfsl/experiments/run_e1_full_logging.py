#!/usr/bin/env python3
"""
E1 Full Experiment Runner - All 6 Methods × 3 Cluster Scenarios
20 nodes, 50 rounds, CIFAR-10

Logs:
- Per-method results: logs/e1_full_experiment/{method}_{scenario}_s{seed}.csv
- Convergence tracking: logs/e1_full_experiment/convergence.csv
- Node cut decisions: logs/e1_full_experiment/node_cuts_{method}_{scenario}.csv

Usage:
    python3 experiments/run_e1_full_logging.py
"""

import os
import sys
import csv
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from experiments.e1_baseline_comparison import (
    METHODS, CLUSTER_SCENARIOS, run, compute_cluster_size,
)
from experiments.utils import build_config, save_results_csv


# ============================================================================
# Configuration
# ============================================================================

N_NODES = 20
GLOBAL_ROUNDS = 50
DATASET = "cifar10"
TIER_DIST = "balanced"
SEEDS = [42]
EVAL_EVERY = 5

LOG_DIR = "./logs/e1_full_experiment"
os.makedirs(LOG_DIR, exist_ok=True)


# ============================================================================
# Config Builder
# ============================================================================

def build_experiment_config():
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
    config["log_dir"] = LOG_DIR
    return config


# ============================================================================
# Convergence Tracker
# ============================================================================

class ConvergenceTracker:
    """Track convergence metrics across rounds."""

    def __init__(self, method: str, scenario: str, seed: int):
        self.method = method
        self.scenario = scenario
        self.seed = seed
        self.rounds_data: List[Dict[str, Any]] = []
        self.node_cuts: Dict[int, List[int]] = {}  # node_id -> [cut per round]
        self.node_h: Dict[int, List[int]] = {}  # node_id -> [H per round]

    def add_round(self, round_idx: int, metrics: Dict[str, Any],
                  node_configs: Dict[int, Dict[str, Any]]):
        """Add round data."""
        self.rounds_data.append({
            "round": round_idx,
            "test_acc": metrics.get("test_acc", 0.0),
            "train_loss": metrics.get("train_loss", 0.0),
            "f1_macro": metrics.get("f1_macro", 0.0),
            "precision_macro": metrics.get("precision_macro", 0.0),
            "recall_macro": metrics.get("recall_macro", 0.0),
            "round_latency": metrics.get("round_latency", 0.0),
            "fairness_index": metrics.get("fairness_index", 0.0),
            "n_valid": metrics.get("n_valid_updates", 0),
            "n_participants": metrics.get("n_participants", 0),
            "mean_shapley": metrics.get("mean_shapley", 0.0),
        })

        # Track node cuts
        for node_id, cfg in node_configs.items():
            if cfg is None:
                continue
            if node_id not in self.node_cuts:
                self.node_cuts[node_id] = []
                self.node_h[node_id] = []
            self.node_cuts[node_id].append(cfg.get("cut_layer", 0))
            self.node_h[node_id].append(cfg.get("H", 1))

    def save_convergence_csv(self):
        """Save convergence data to CSV."""
        filepath = os.path.join(
            LOG_DIR, f"convergence_{self.method}_{self.scenario}_s{self.seed}.csv"
        )
        if not self.rounds_data:
            return

        fieldnames = list(self.rounds_data[0].keys())
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rounds_data)

    def save_node_cuts_csv(self):
        """Save per-node cut decisions to CSV."""
        filepath = os.path.join(
            LOG_DIR, f"node_cuts_{self.method}_{self.scenario}_s{self.seed}.csv"
        )
        if not self.rounds_data:
            return

        n_rounds = len(self.rounds_data)
        rows = []
        for node_id in sorted(self.node_cuts.keys()):
            cuts = self.node_cuts.get(node_id, [])
            hs = self.node_h.get(node_id, [])
            row = {"node_id": node_id}
            for r in range(n_rounds):
                row[f"round_{r}_cut"] = cuts[r] if r < len(cuts) else ""
                row[f"round_{r}_H"] = hs[r] if r < len(hs) else ""
            rows.append(row)

        if rows:
            fieldnames = ["node_id"] + [f"round_{r}_cut" for r in range(n_rounds)] + \
                         [f"round_{r}_H" for r in range(n_rounds)]
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)


# ============================================================================
# Run Single Method
# ============================================================================

def run_single_experiment(
    method: str,
    scenario: str,
    seed: int,
    config: Dict[str, Any],
) -> ConvergenceTracker:
    """Run single experiment and track all metrics."""
    tracker = ConvergenceTracker(method, scenario, seed)

    print(f"\n{'='*70}")
    print(f"METHOD: {method} | SCENARIO: {scenario} | SEED: {seed}")
    print(f"{'='*70}")

    cluster_ratio = CLUSTER_SCENARIOS[scenario]
    cluster_size = compute_cluster_size(N_NODES, cluster_ratio)

    # Build method-specific config
    method_cfg = METHODS[method]
    if method.startswith("haso_"):
        run_config = {
            **config,
            "arch_mode": method_cfg.get("arch_mode"),
            "haso_enabled": True,
            "seed": seed,
        }
        if method_cfg.get("arch_mode") == "cluster":
            run_config["cluster_size"] = cluster_size
    else:
        run_config = {**config, "seed": seed}

    start_time = time.time()
    try:
        metrics_list = run(
            config=run_config,
            method=method,
            cluster_ratio=cluster_ratio,
            verbose=True,
        )

        # Extract per-round metrics and node configs
        for i, metrics in enumerate(metrics_list):
            round_idx = i + 1

            # Build node configs for this round
            node_configs = {}
            if hasattr(metrics, 'to_dict'):
                m_dict = metrics.to_dict()
            else:
                m_dict = metrics

            # For HASO methods, we need per-node cut decisions
            # This info should be in the protocol's node_progress or similar
            # For now, we record what we have from metrics
            node_configs = {}  # Would need protocol changes to capture full info

            tracker.add_round(round_idx, m_dict, node_configs)

            # Log progress
            if round_idx % 10 == 0 or round_idx == 1:
                acc = m_dict.get("test_acc", 0.0)
                loss = m_dict.get("train_loss", 0.0)
                lat = m_dict.get("round_latency", 0.0)
                f1 = m_dict.get("f1_macro", 0.0)
                print(
                    f"  Round {round_idx:3d}/{GLOBAL_ROUNDS} | "
                    f"Acc: {acc:.2f}% | Loss: {loss:.4f} | "
                    f"F1: {f1:.3f} | Lat: {lat:.2f}s"
                )

    except Exception as e:
        print(f"[ERROR] {method}/{scenario}/s{seed}: {e}")
        import traceback
        traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s")

    # Save tracking data
    tracker.save_convergence_csv()
    tracker.save_node_cuts_csv()

    return tracker


# ============================================================================
# Main
# ============================================================================

def main():
    print(f"""
╔════════════════════════════════════════════════════════════════════╗
║  E1 Full Experiment - All Methods × All Scenarios × All Seeds       ║
║  20 nodes, 50 rounds, CIFAR-10                                     ║
╚════════════════════════════════════════════════════════════════════╝
""")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log dir: {LOG_DIR}")

    config = build_experiment_config()

    # Track all results
    all_results: Dict[str, Dict[str, Any]] = {}

    for method in METHODS.keys():
        all_results[method] = {}

        for scenario in CLUSTER_SCENARIOS.keys():
            scenario_results = []

            for seed in SEEDS:
                tracker = run_single_experiment(
                    method=method,
                    scenario=scenario,
                    seed=seed,
                    config=config,
                )
                scenario_results.append(tracker.rounds_data)

            # Aggregate scenario results
            all_results[method][scenario] = scenario_results

    # Print summary table
    print_summary(all_results)

    # Save summary
    save_summary_json(all_results)

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Done! All logs saved to:", LOG_DIR)


def print_summary(all_results: Dict[str, Dict[str, Any]]):
    """Print summary table of all experiments."""
    print("\n" + "="*100)
    print("E1 FULL EXPERIMENT SUMMARY")
    print("="*100)
    print(f"{'Method':<18} {'Scenario':<10} {'Final Acc':>9} {'Avg Lat':>9} {'Fairness':>9} {'Convergence':>12}")
    print("-"*100)

    for method in METHODS.keys():
        for scenario in CLUSTER_SCENARIOS.keys():
            runs = all_results.get(method, {}).get(scenario, [])
            if not runs:
                print(f"{method:<18} {scenario:<10} {'N/A':>9} {'N/A':>9} {'N/A':>9} {'N/A':>12}")
                continue

            # Average across seeds
            final_accs = []
            avg_lats = []
            final_fairness = []

            for run_data in runs:
                if run_data:
                    final_accs.append(run_data[-1].get("test_acc", 0.0))
                    lats = [r.get("round_latency", 0.0) for r in run_data]
                    avg_lats.append(np.mean(lats) if lats else 0.0)
                    final_fairness.append(run_data[-1].get("fairness_index", 0.0))

            if final_accs:
                avg_acc = np.mean(final_accs)
                avg_lat = np.mean(avg_lats)
                avg_fair = np.mean(final_fairness)

                # Estimate convergence round (time to 70% acc)
                conv_rounds = "N/A"
                for r in runs[0] if runs else []:
                    if r.get("test_acc", 0) >= 70.0:
                        conv_rounds = str(r["round"])
                        break

                print(
                    f"{method:<18} {scenario:<10} "
                    f"{avg_acc:>8.2f}% {avg_lat:>8.2f}s {avg_fair:>9.3f} {conv_rounds:>12}"
                )

    print("="*100)


def save_summary_json(all_results: Dict[str, Dict[str, Any]]):
    """Save summary as JSON for programmatic analysis."""
    summary = {}
    for method in METHODS.keys():
        summary[method] = {}
        for scenario in CLUSTER_SCENARIOS.keys():
            runs = all_results.get(method, {}).get(scenario, [])
            if not runs:
                continue

            runs_data = []
            for run_data in runs:
                if not run_data:
                    continue
                run_summary = {
                    "final_acc": run_data[-1].get("test_acc", 0.0),
                    "avg_latency": np.mean([r.get("round_latency", 0.0) for r in run_data]),
                    "final_fairness": run_data[-1].get("fairness_index", 0.0),
                    "convergence_round": None,
                    "accuracy_curve": [r.get("test_acc", 0.0) for r in run_data],
                    "latency_curve": [r.get("round_latency", 0.0) for r in run_data],
                }
                # Find convergence round
                for r in run_data:
                    if r.get("test_acc", 0) >= 70.0:
                        run_summary["convergence_round"] = r["round"]
                        break
                runs_data.append(run_summary)

            if runs_data:
                summary[method][scenario] = {
                    "n_runs": len(runs_data),
                    "final_acc_mean": np.mean([r["final_acc"] for r in runs_data]),
                    "final_acc_std": np.std([r["final_acc"] for r in runs_data]),
                    "avg_latency_mean": np.mean([r["avg_latency"] for r in runs_data]),
                    "final_fairness_mean": np.mean([r["final_fairness"] for r in runs_data]),
                    "convergence_rounds": [r["convergence_round"] for r in runs_data],
                }

    with open(os.path.join(LOG_DIR, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()