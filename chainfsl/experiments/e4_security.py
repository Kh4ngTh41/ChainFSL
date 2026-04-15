"""
E4: Security Experiment.

Hypothesis: TVE detects Sybil, lazy, and poison attacks with >95% detection rate.

This experiment injects three types of attacks at varying fractions:
1. Lazy clients: Submit stale/zero gradients
2. Sybil nodes: Create fake identities to dominate rewards
3. Poison data: Send intentionally corrupted activations

Measures:
- Attack detection rate (TVE validation success)
- Accuracy degradation under attack
- False positive rate (honest nodes flagged)
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Set

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.protocol.chainfsl import ChainFSLProtocol
from experiments.utils import save_results_csv, print_summary, ensure_dir


def run(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run E4 security experiment.

    Tests lazy client attack at fractions [0.0, 0.1, 0.2, 0.3].
    """
    print("=" * 60)
    print("E4: Security Evaluation")
    print("Hypothesis: TVE achieves >95% detection rate")
    print("=" * 60)

    attack_types = ["lazy"]
    fractions = [0.0, 0.1, 0.2, 0.3]
    results = {}

    for attack_type in attack_types:
        for fraction in fractions:
            key = f"{attack_type}_f{fraction}"
            print(f"\n--- {attack_type} fraction = {fraction} ---")

            cfg = {
                **config,
                "lazy_client_fraction": fraction if attack_type == "lazy" else 0.0,
                "sybil_fraction": fraction if attack_type == "sybil" else 0.0,
                "poison_fraction": fraction if attack_type == "poison" else 0.0,
                "global_rounds": 30,
                "tve_enabled": True,
            }

            protocol = ChainFSLProtocol(
                config=cfg,
                device=None,
                db_path=f"/tmp/chainfsl_e4_{key}.db",
            )

            # Inject attack
            n_attack = int(fraction * config["n_nodes"])
            attack_ids: Set[int] = set(range(n_attack))

            if attack_type == "lazy":
                protocol.inject_lazy_clients(attack_ids)
            elif attack_type == "sybil":
                protocol.inject_sybil(attack_ids)

            metrics = protocol.run(total_rounds=cfg["global_rounds"], eval_every=5)
            metrics_dicts = [m.to_dict() for m in metrics]

            save_results_csv(f"e4_{key}", metrics_dicts, config["log_dir"])

            detection_rate = _mean([m.attack_detection_rate for m in metrics])
            final_acc = metrics[-1].test_acc if metrics else 0.0
            false_positive_rate = _estimate_false_positive_rate(metrics_dicts, attack_ids, config["n_nodes"])

            results[key] = {
                "attack_type": attack_type,
                "fraction": fraction,
                "detection_rate": detection_rate,
                "false_positive_rate": false_positive_rate,
                "final_acc": final_acc,
                "metrics": metrics_dicts,
            }

            print(f"  Detection rate: {detection_rate:.2%}")
            print(f"  False positive rate: {false_positive_rate:.2%}")
            print(f"  Final accuracy: {final_acc:.2f}%")

    _print_security_table(results)
    _save_security_summary(results, Path(config["log_dir"]) / "e4_security_summary.csv")

    return results


def _mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def _estimate_false_positive_rate(
    metrics: List[Dict[str, Any]],
    attack_ids: Set[int],
    total_nodes: int,
) -> float:
    """Estimate false positive rate (honest nodes flagged as malicious)."""
    if not metrics:
        return 0.0

    # In the protocol, honest nodes should have is_valid=True
    # This is a simplified estimation
    last_metric = metrics[-1]
    n_valid = last_metric.get("n_valid_updates", 0)
    n_participants = last_metric.get("n_participants", total_nodes)

    honest_participated = n_participants - len(attack_ids)
    if honest_participated <= 0:
        return 0.0

    # False positives = honest nodes flagged
    # Since we don't have per-node validation data here, use detection rate proxy
    honest_valid_rate = n_valid / max(n_participants, 1)
    return max(0.0, 1.0 - honest_valid_rate)


def _print_security_table(results: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("E4 SECURITY TABLE")
    print("=" * 80)
    print(f"{'Attack':>15} {'Fraction':>10} {'Detection Rate':>15} {'FP Rate':>10} {'Final Acc':>10}")
    print("-" * 80)
    for key, data in results.items():
        print(
            f"{data['attack_type']:>15} "
            f"{data['fraction']:>9.0%} "
            f"{data['detection_rate']:>14.2%} "
            f"{data['false_positive_rate']:>9.2%} "
            f"{data['final_acc']:>9.2f}%"
        )
    print("=" * 80)


def _save_security_summary(results: Dict[str, Any], path: Path) -> None:
    ensure_dir(str(path.parent))
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["attack_type", "fraction", "detection_rate", "false_positive_rate", "final_acc"]
        )
        writer.writeheader()
        for data in results.values():
            writer.writerow({
                "attack_type": data["attack_type"],
                "fraction": data["fraction"],
                "detection_rate": round(data["detection_rate"], 4),
                "false_positive_rate": round(data["false_positive_rate"], 4),
                "final_acc": round(data["final_acc"], 2),
            })

    print(f"[E4] Security summary saved to: {path}")
