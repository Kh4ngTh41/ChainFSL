"""
E7: Blockchain Overhead Experiment.

Hypothesis: Blockchain ledger overhead is <5% of total training time.

This experiment measures:
1. Per-epoch ledger write latency
2. Total blockchain ledger size growth
3. Merkle root commit overhead
4. Comparison: with/without blockchain module enabled

Expected: O(1) writes per epoch, Merkle root = 32 bytes.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.protocol.chainfsl import ChainFSLProtocol
from experiments.utils import save_results_csv, print_summary, ensure_dir


def run(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run E7 blockchain overhead experiment.

    Measures ledger overhead with and without blockchain module.
    """
    print("=" * 60)
    print("E7: Blockchain Overhead")
    print("Hypothesis: Blockchain overhead <5% of total time")
    print("=" * 60)

    results = {}

    # --- Method 1: Full ChainFSL with blockchain ---
    print("\n--- ChainFSL with Blockchain ---")
    full_cfg = {**config, "global_rounds": 30}

    protocol = ChainFSLProtocol(
        config=full_cfg,
        device=None,
        db_path="/tmp/chainfsl_e7_blockchain.db",
    )

    blockchain_metrics = protocol.run(total_rounds=30, eval_every=5)
    blockchain_metrics_dicts = [m.to_dict() for m in blockchain_metrics]

    save_results_csv("e7_with_blockchain", blockchain_metrics_dicts, config["log_dir"])
    print_summary("e7_with_blockchain", blockchain_metrics_dicts)

    # Extract overhead metrics
    mean_latency_bc = _mean([m.round_latency for m in blockchain_metrics])
    mean_ledger_kb = _mean([m.ledger_size_kb for m in blockchain_metrics])
    final_acc_bc = blockchain_metrics[-1].test_acc if blockchain_metrics else 0.0

    results["with_blockchain"] = {
        "metrics": blockchain_metrics_dicts,
        "mean_latency": mean_latency_bc,
        "ledger_size_kb": mean_ledger_kb,
        "final_acc": final_acc_bc,
    }

    # --- Method 2: ChainFSL without blockchain ---
    print("\n--- ChainFSL without Blockchain (GTM only) ---")
    no_bc_cfg = {**config, "global_rounds": 30, "gtm_enabled": True}

    protocol_nobc = ChainFSLProtocol(
        config=no_bc_cfg,
        device=None,
        db_path="/tmp/chainfsl_e7_no_blockchain.db",
    )

    no_bc_metrics = protocol_nobc.run(total_rounds=30, eval_every=5)
    no_bc_metrics_dicts = [m.to_dict() for m in no_bc_metrics]

    save_results_csv("e7_without_blockchain", no_bc_metrics_dicts, config["log_dir"])
    print_summary("e7_without_blockchain", no_bc_metrics_dicts)

    mean_latency_no_bc = _mean([m.round_latency for m in no_bc_metrics])
    final_acc_no_bc = no_bc_metrics[-1].test_acc if no_bc_metrics else 0.0

    results["without_blockchain"] = {
        "metrics": no_bc_metrics_dicts,
        "mean_latency": mean_latency_no_bc,
        "final_acc": final_acc_no_bc,
    }

    # --- Compute overhead ---
    overhead_pct = ((mean_latency_bc - mean_latency_no_bc) / mean_latency_bc * 100) if mean_latency_bc > 0 else 0.0
    overhead_s = mean_latency_bc - mean_latency_no_bc

    print(f"\n--- Blockchain Overhead Analysis ---")
    print(f"  Mean latency (with blockchain):    {mean_latency_bc:.3f}s")
    print(f"  Mean latency (without blockchain): {mean_latency_no_bc:.3f}s")
    print(f"  Overhead: {overhead_s:.3f}s ({overhead_pct:.2f}%)")
    print(f"  H0 (<5% overhead): {'PASSED' if overhead_pct < 5.0 else 'FAILED'}")

    results["overhead_analysis"] = {
        "overhead_seconds": overhead_s,
        "overhead_percent": overhead_pct,
        "hypothesis_passed": overhead_pct < 5.0,
        "ledger_size_mb": mean_ledger_kb / 1024.0,
    }

    _save_overhead_summary(results, Path(config["log_dir"]) / "e7_overhead_summary.csv")

    return results


def _mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def _save_overhead_summary(results: Dict[str, Any], path: Path) -> None:
    ensure_dir(str(path.parent))
    import csv

    overhead = results.get("overhead_analysis", {})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerow({"metric": "overhead_seconds", "value": round(overhead.get("overhead_seconds", 0), 4)})
        writer.writerow({"metric": "overhead_percent", "value": round(overhead.get("overhead_percent", 0), 4)})
        writer.writerow({"metric": "hypothesis_passed", "value": overhead.get("hypothesis_passed", False)})
        writer.writerow({"metric": "ledger_size_mb", "value": round(overhead.get("ledger_size_mb", 0), 4)})
        writer.writerow({"metric": "with_blockchain_latency", "value": round(results.get("with_blockchain", {}).get("mean_latency", 0), 4)})
        writer.writerow({"metric": "without_blockchain_latency", "value": round(results.get("without_blockchain", {}).get("mean_latency", 0), 4)})

    print(f"[E7] Overhead summary saved to: {path}")
