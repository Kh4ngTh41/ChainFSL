"""
E3: Non-IID Data Experiment.

Hypothesis: ChainFSL is robust across varying data heterogeneity.

This experiment varies the Dirichlet alpha parameter:
- alpha = 0.1 (highly non-IID, few classes per client)
- alpha = 0.5 (moderately non-IID)
- alpha = 1.0 (nearly IID)
- alpha = 10.0 (almost uniform IID)

Measures how accuracy degrades as data becomes more heterogeneous.
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
    Run E3 Non-IID experiment.

    Varies Dirichlet alpha across [0.1, 0.5, 1.0, 10.0].
    """
    print("=" * 60)
    print("E3: Non-IID Data Heterogeneity")
    print("Hypothesis: ChainFSL is robust across alpha ∈ [0.1, 10.0]")
    print("=" * 60)

    alphas = [0.1, 0.5, 1.0, 10.0]
    results = {}

    for alpha in alphas:
        print(f"\n--- Dirichlet alpha = {alpha} ---")
        cfg = {**config, "dirichlet_alpha": alpha, "global_rounds": 30}

        protocol = ChainFSLProtocol(
            config=cfg,
            device=None,
            db_path=f"/tmp/chainfsl_e3_a{alpha}.db",
        )

        metrics = protocol.run(total_rounds=cfg["global_rounds"], eval_every=5)
        metrics_dicts = [m.to_dict() for m in metrics]

        save_results_csv(f"e3_alpha_{alpha}", metrics_dicts, config["log_dir"])
        print_summary(f"e3_alpha_{alpha}", metrics_dicts)

        results[f"alpha_{alpha}"] = {
            "alpha": alpha,
            "metrics": metrics_dicts,
            "final_acc": metrics[-1].test_acc if metrics else 0.0,
            "best_acc": max((m.test_acc for m in metrics), default=0.0),
            "mean_fairness": _mean([m.fairness_index for m in metrics]),
        }

    _print_noniid_table(results)
    _save_noniid_summary(results, Path(config["log_dir"]) / "e3_noniid_summary.csv")

    return results


def _print_noniid_table(results: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("E3 NON-IID TABLE")
    print("=" * 70)
    print(f"{'Alpha':>10} {'Final Acc':>12} {'Best Acc':>10} {'Fairness':>10}")
    print("-" * 70)
    for key, data in results.items():
        print(
            f"{data['alpha']:>10.1f} "
            f"{data['final_acc']:>11.2f}% "
            f"{data['best_acc']:>9.2f}% "
            f"{data['mean_fairness']:>10.3f}"
        )
    print("=" * 70)


def _save_noniid_summary(results: Dict[str, Any], path: Path) -> None:
    ensure_dir(str(path.parent))
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "final_acc", "best_acc", "mean_fairness"])
        writer.writeheader()
        for data in results.values():
            writer.writerow({
                "alpha": data["alpha"],
                "final_acc": round(data["final_acc"], 2),
                "best_acc": round(data["best_acc"], 2),
                "mean_fairness": round(data["mean_fairness"], 3),
            })

    print(f"[E3] Non-IID summary saved to: {path}")


def _mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0
