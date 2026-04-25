"""
E1: HASO Effectiveness Experiment.

Hypothesis: HASO reduces training latency 40-60% vs uniform-split baselines.

This experiment compares ChainFSL (with HASO DRL) against three baselines:
1. ChainFSL-NoHASO (uniform cut_layer=2)
2. SplitFed (uniform fixed cut_layer=2)
3. FedAvg (full model on client, no split)

Metrics:
- Time-to-accuracy (target: 60% accuracy)
- Training latency per round
- Straggler ratio (fraction of nodes exceeding 1.5x mean latency)
- Final accuracy

The experiment uses 10 nodes, 50 rounds.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

# Add project root
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.protocol.chainfsl import ChainFSLProtocol
from baselines import FedAvgBaseline, SplitFedBaseline
from experiments.utils import (
    build_config,
    run_experiment,
    save_results_csv,
    print_summary,
    ensure_dir,
)


TARGET_ACCURACY = 60.0  # Target accuracy % for time-to-accuracy measurement


def run(
    config: Dict[str, Any],
    skip_baselines: bool = False,
    resume: bool = False,
    checkpoint_dir: str = "./checkpoints",
    pretrained_orchestrator=None,
    pretrain_dir: str = "pretrainppo",
) -> Dict[str, Any]:
    """
    Run E1 experiment.

    Args:
        config: Base config dict.
        skip_baselines: If True, skip baseline comparisons.
        resume: If True, resume from latest checkpoint.
        checkpoint_dir: Directory for checkpoint files.
        pretrained_orchestrator: Pre-trained HASOOrchestrator (if available).
        pretrain_dir: Directory containing pretrained models.

    Returns:
        Dict of results for each method.
    """
    print("=" * 60)
    print("E1: HASO Effectiveness")
    print("Hypothesis: HASO reduces latency 40-60% vs uniform splits")
    print("=" * 60)

    results = {}

    # --- Method 1: ChainFSL (full HASO) ---
    print("\n--- ChainFSL (HASO enabled) ---")
    chainfsl_cfg = {**config, "haso_enabled": True, "tve_enabled": True, "gtm_enabled": True}
    metrics_chainfsl = _run_chainfsl(
        chainfsl_cfg,
        pretrained_orchestrator=pretrained_orchestrator,
    )
    results["chainfsl"] = metrics_chainfsl
    save_results_csv("e1_chainfsl", metrics_chainfsl, config["log_dir"])
    print_summary("chainfsl", metrics_chainfsl)

    # --- Method 2: ChainFSL-NoHASO (required core comparison) ---
    print("\n--- ChainFSL-NoHASO ---")
    no_haso_cfg = {
        **config,
        "haso_enabled": False,
        "tve_enabled": True,
        "gtm_enabled": True,
    }
    metrics_nohaso = _run_chainfsl(no_haso_cfg)
    results["chainfsl_nohaso"] = metrics_nohaso
    save_results_csv("e1_chainfsl_nohaso", metrics_nohaso, config["log_dir"])
    print_summary("chainfsl_nohaso", metrics_nohaso)

    if not skip_baselines:

        # --- Method 3: SplitFed baseline ---
        print("\n--- SplitFed (uniform cut=2) ---")
        splitfed_cfg = {**config, "global_rounds": min(config["global_rounds"], 30)}
        metrics_splitfed = _run_splitfed(splitfed_cfg)
        results["splitfed"] = metrics_splitfed
        save_results_csv("e1_splitfed", metrics_splitfed, config["log_dir"])
        print_summary("splitfed", metrics_splitfed)

        # --- Method 4: FedAvg baseline ---
        print("\n--- FedAvg (no split) ---")
        metrics_fedavg = _run_fedavg(config)
        results["fedavg"] = metrics_fedavg
        save_results_csv("e1_fedavg", metrics_fedavg, config["log_dir"])
        print_summary("fedavg", metrics_fedavg)

    # --- Print comparison table ---
    _print_comparison_table(results, TARGET_ACCURACY)

    # Save combined results
    combined_path = Path(config["log_dir"]) / "e1_haso_comparison.csv"
    _save_combined_results(results, combined_path)

    return results


def _run_chainfsl(
    config: Dict[str, Any],
    pretrained_orchestrator=None,
) -> List[Dict[str, Any]]:
    """Run ChainFSL protocol."""
    import os
    # Use log_dir for ledger DB to ensure Windows compatibility
    log_dir = config.get("log_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)
    db_path = os.path.join(log_dir, f"chainfsl_e1_{config['seed']}.db")

    protocol = ChainFSLProtocol(
        config=config,
        device=None,  # auto
        db_path=db_path,
    )

    # Attach pretrained orchestrator if available
    if pretrained_orchestrator is not None:
        print(f"  [ChainFSL] Using pretrained orchestrator")
        protocol._orchestrator = pretrained_orchestrator

    metrics = protocol.run(
        total_rounds=config["global_rounds"],
        eval_every=5,
    )

    return [m.to_dict() for m in metrics]


def _run_splitfed(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run SplitFed baseline."""
    baseline = SplitFedBaseline(
        config={**config, "cut_layer": 2},
        cut_layer=2,
        device=None,
    )
    raw = baseline.run()
    # Add standard keys
    for m in raw:
        m["method"] = "splitfed"
    return raw


def _run_fedavg(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run FedAvg baseline."""
    baseline = FedAvgBaseline(
        config={**config, "global_rounds": min(config["global_rounds"], 30)},
        device=None,
    )
    raw = baseline.run()
    for m in raw:
        m["method"] = "fedavg"
    return raw


def _print_comparison_table(results: Dict[str, Any], target_acc: float) -> None:
    """Print a comparison table of all methods."""
    print("\n" + "=" * 70)
    print("E1 COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Method':<20} {'Final Acc':>10} {'Mean Latency':>12} {'Fairness':>10} {'Rounds':>8}")
    print("-" * 70)

    for name, metrics in results.items():
        if not metrics:
            continue

        final_acc = metrics[-1].get("test_acc", 0)
        mean_lat = _mean([m.get("round_latency", 0) for m in metrics])
        fairness = _mean([m.get("fairness_index", 0) for m in metrics])
        n_rounds = len(metrics)

        print(f"{name:<20} {final_acc:>9.2f}% {mean_lat:>11.2f}s {fairness:>10.3f} {n_rounds:>8}")

    print("=" * 70)


def _save_combined_results(results: Dict[str, Any], path: Path) -> None:
    """Save combined results from all methods."""
    ensure_dir(str(path.parent))

    import csv

    all_keys = {"method"}  # "method" is always present
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
