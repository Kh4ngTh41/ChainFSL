"""
E2: Scalability Experiment.

Hypothesis: ChainFSL achieves near-linear scaling with number of nodes.

This experiment varies the number of nodes (N = 5, 10, 20, 50) and measures:
- Throughput (updates/second)
- Per-round latency
- Accuracy convergence rate

Expected: Throughput should scale sub-linearly due to coordination overhead.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.protocol.chainfsl import ChainFSLProtocol
from experiments.utils import build_config, save_results_csv, print_summary, ensure_dir


def run(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run E2 scalability experiment.

    Varies n_nodes across [5, 10, 20, 50] and measures scaling behavior.
    """
    print("=" * 60)
    print("E2: Scalability")
    print("Hypothesis: Near-linear throughput scaling with N")
    print("=" * 60)

    node_counts = [5, 10, 20, 50]
    results = {}

    for n_nodes in node_counts:
        print(f"\n--- n_nodes = {n_nodes} ---")
        cfg = {**config, "n_nodes": n_nodes, "global_rounds": 20}

        protocol = ChainFSLProtocol(
            config=cfg,
            device=None,
            db_path=f"/tmp/chainfsl_e2_n{n_nodes}.db",
        )

        metrics = protocol.run(total_rounds=cfg["global_rounds"], eval_every=10)
        metrics_dicts = [m.to_dict() for m in metrics]

        save_results_csv(f"e2_n{n_nodes}", metrics_dicts, config["log_dir"])
        print_summary(f"e2_n{n_nodes}", metrics_dicts)

        results[f"n_{n_nodes}"] = {
            "n_nodes": n_nodes,
            "metrics": metrics_dicts,
            "mean_latency": _mean([m.round_latency for m in metrics]),
            "throughput": n_nodes * cfg["global_rounds"] / sum(m.round_latency for m in metrics),
            "final_acc": metrics[-1].test_acc if metrics else 0.0,
        }

    # Print scaling table
    _print_scaling_table(results)

    # Save summary
    _save_scaling_summary(results, Path(config["log_dir"]) / "e2_scalability_summary.csv")

    return results


def _print_scaling_table(results: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("E2 SCALING TABLE")
    print("=" * 70)
    print(f"{'N Nodes':>10} {'Mean Latency':>14} {'Throughput':>12} {'Final Acc':>10}")
    print("-" * 70)
    for key, data in results.items():
        print(
            f"{data['n_nodes']:>10} "
            f"{data['mean_latency']:>13.2f}s "
            f"{data['throughput']:>11.2f} "
            f"{data['final_acc']:>9.2f}%"
        )
    print("=" * 70)


def _save_scaling_summary(results: Dict[str, Any], path: Path) -> None:
    ensure_dir(str(path.parent))
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["n_nodes", "mean_latency", "throughput", "final_acc"]
        )
        writer.writeheader()
        for data in results.values():
            writer.writerow({
                "n_nodes": data["n_nodes"],
                "mean_latency": round(data["mean_latency"], 3),
                "throughput": round(data["throughput"], 3),
                "final_acc": round(data["final_acc"], 2),
            })

    print(f"[E2] Scaling summary saved to: {path}")


def _mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0
