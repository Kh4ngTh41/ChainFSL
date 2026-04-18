"""
E6: Ablation Study.

Hypothesis: Each module (HASO, TVE, GTM) contributes positively to overall performance.

This experiment systematically disables each module to quantify its contribution:

1. Full (HASO + TVE + GTM) — baseline
2. No HASO (TVE + GTM only) — evaluate split optimization value
3. No TVE (HASO + GTM only) — evaluate verification value
4. No GTM (HASO + TVE only) — evaluate incentive mechanism value
5. No HASO + No TVE (GTM only) — GTM alone
6. No HASO + No GTM (TVE only) — TVE alone

Metrics:
- Final accuracy
- Fairness index (Jain's)
- Attack resilience (if TVE disabled)
- Reward distribution
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.protocol.chainfsl import ChainFSLProtocol
from experiments.utils import save_results_csv, print_summary, ensure_dir


ABLATION_CONFIGS = {
    "full":           {"haso": True,  "tve": True,  "gtm": True},
    "no_haso":        {"haso": False, "tve": True,  "gtm": True},
    "no_tve":         {"haso": True,  "tve": False, "gtm": True},
    "no_gtm":         {"haso": True,  "tve": True,  "gtm": False},
    "no_haso_no_tve": {"haso": False, "tve": False, "gtm": True},
    "no_haso_no_gtm": {"haso": False, "tve": True,  "gtm": False},
}


def run(
    config: Dict[str, Any],
    ablation_type: str = None,
    resume: bool = False,
    checkpoint_dir: str = "./checkpoints",
    pretrained_orchestrator=None,
    pretrain_dir: str = "pretrainppo",
) -> Dict[str, Any]:
    """
    Run E6 ablation study.

    Args:
        config: Base config dict.
        ablation_type: If provided, run only that specific ablation variant.
                       Otherwise run all variants.
        resume: If True, resume from checkpoint.
        checkpoint_dir: Directory for checkpoint files.
        pretrained_orchestrator: Pre-trained HASOOrchestrator (if available).
        pretrain_dir: Directory containing pretrained models.
    """
    print("=" * 60)
    print("E6: Ablation Study")
    print("Hypothesis: Each module contributes positively")
    print("=" * 60)

    results = {}

    if ablation_type and ablation_type in ABLATION_CONFIGS:
        # Run only specified variant
        variants = [ablation_type]
    else:
        variants = list(ABLATION_CONFIGS.keys())

    for variant in variants:
        flags = ABLATION_CONFIGS[variant]
        print(f"\n--- Ablation: {variant} (HASO={flags['haso']}, TVE={flags['tve']}, GTM={flags['gtm']}) ---")

        cfg = {
            **config,
            "haso_enabled": flags["haso"],
            "tve_enabled": flags["tve"],
            "gtm_enabled": flags["gtm"],
            "global_rounds": 30,
        }

        protocol = ChainFSLProtocol(
            config=cfg,
            device=None,
            db_path=f"/tmp/chainfsl_e6_{variant}.db",
        )

        # Attach pretrained orchestrator only if HASO is enabled
        if pretrained_orchestrator is not None and flags["haso"]:
            print(f"  [E6:{variant}] Using pretrained orchestrator")
            protocol._orchestrator = pretrained_orchestrator

        metrics = protocol.run(total_rounds=cfg["global_rounds"], eval_every=5)
        metrics_dicts = [m.to_dict() for m in metrics]

        save_results_csv(f"e6_{variant}", metrics_dicts, config["log_dir"])
        print_summary(f"e6_{variant}", metrics_dicts)

        final_acc = metrics[-1].test_acc if metrics else 0.0
        best_acc = max((m.test_acc for m in metrics), default=0.0)
        mean_fairness = _mean([m.fairness_index for m in metrics])
        mean_latency = _mean([m.round_latency for m in metrics])

        results[variant] = {
            "metrics": metrics_dicts,
            "final_acc": final_acc,
            "best_acc": best_acc,
            "mean_fairness": mean_fairness,
            "mean_latency": mean_latency,
        }

    _print_ablation_table(results)
    _save_ablation_summary(results, Path(config["log_dir"]) / "e6_ablation_summary.csv")

    return results


def _mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def _print_ablation_table(results: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("E6 ABLATION TABLE")
    print("=" * 80)
    print(f"{'Variant':<20} {'Final Acc':>10} {'Best Acc':>10} {'Fairness':>10} {'Latency':>10}")
    print("-" * 80)
    for variant, data in results.items():
        print(
            f"{variant:<20} "
            f"{data['final_acc']:>9.2f}% "
            f"{data['best_acc']:>9.2f}% "
            f"{data['mean_fairness']:>10.3f} "
            f"{data['mean_latency']:>9.2f}s"
        )
    print("=" * 80)

    # Delta vs full
    if "full" in results:
        print("\nDELTA vs FULL (baseline):")
        full = results["full"]
        for variant, data in results.items():
            if variant == "full":
                continue
            delta_acc = data["final_acc"] - full["final_acc"]
            delta_fairness = data["mean_fairness"] - full["mean_fairness"]
            print(f"  {variant:<20}: acc={delta_acc:+.2f}%, fairness={delta_fairness:+.3f}")


def _save_ablation_summary(results: Dict[str, Any], path: Path) -> None:
    ensure_dir(str(path.parent))
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["variant", "final_acc", "best_acc", "mean_fairness", "mean_latency"]
        )
        writer.writeheader()
        for variant, data in results.items():
            writer.writerow({
                "variant": variant,
                "final_acc": round(data["final_acc"], 2),
                "best_acc": round(data["best_acc"], 2),
                "mean_fairness": round(data["mean_fairness"], 4),
                "mean_latency": round(data["mean_latency"], 3),
            })

    print(f"[E6] Ablation summary saved to: {path}")
