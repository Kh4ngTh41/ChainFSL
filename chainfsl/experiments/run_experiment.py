#!/usr/bin/env python3
"""
ChainFSL Experiment Runner.

Usage:
    python experiments/run_experiment.py --exp e1
    python experiments/run_experiment.py --exp e1 --n_nodes 20 --global_rounds 50
    python experiments/run_experiment.py --exp e4 --lazy_fraction 0.2
    python experiments/run_experiment.py --exp e6 --ablation full
    python experiments/run_experiment.py --exp all --global_rounds 20

Examples:
    # E1: HASO effectiveness (ChainFSL vs baselines)
    python experiments/run_experiment.py --exp e1

    # E2: Scalability (vary number of nodes)
    python experiments/run_experiment.py --exp e2

    # E3: Non-IID data (vary Dirichlet alpha)
    python experiments/run_experiment.py --exp e3 --alpha 0.1

    # E4: Security (lazy client attack)
    python experiments/run_experiment.py --exp e4 --lazy_fraction 0.2

    # E5: Incentive (Nash equilibrium validation)
    python experiments/run_experiment.py --exp e5

    # E6: Ablation study (remove each module)
    python experiments/run_experiment.py --exp e6

    # E7: Blockchain overhead
    python experiments/run_experiment.py --exp e7

    # Run all experiments
    python experiments/run_experiment.py --exp all
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments import (
    e1_haso_effectiveness,
    e2_scalability,
    e3_noniid,
    e4_security,
    e5_incentive,
    e6_ablation,
    e7_blockchain_overhead,
)
from experiments.utils import build_config, load_config, ensure_dir


EXPERIMENT_MAP = {
    "e1": e1_haso_effectiveness,
    "e2": e2_scalability,
    "e3": e3_noniid,
    "e4": e4_security,
    "e5": e5_incentive,
    "e6": e6_ablation,
    "e7": e7_blockchain_overhead,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="ChainFSL Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/run_experiment.py --exp e1
  python experiments/run_experiment.py --exp e4 --lazy_fraction 0.2
  python experiments/run_experiment.py --exp e6 --ablation no_tve
  python experiments/run_experiment.py --exp all --global_rounds 20
        """,
    )
    parser.add_argument(
        "--exp",
        required=True,
        choices=["e1", "e2", "e3", "e4", "e5", "e6", "e7", "all"],
        help="Experiment to run",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file (overrides defaults)",
    )
    parser.add_argument(
        "--n_nodes",
        type=int,
        default=None,
        help="Number of nodes (overrides config)",
    )
    parser.add_argument(
        "--global_rounds",
        type=int,
        default=None,
        help="Number of global rounds (overrides config)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Dirichlet alpha for non-IID data (overrides config)",
    )
    parser.add_argument(
        "--lazy_fraction",
        type=float,
        default=None,
        help="Fraction of lazy/malicious clients (E4)",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default=None,
        help="Ablation variant (e.g., 'no_haso', 'no_tve', 'no_gtm') for E6",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--log_dir",
        default="./logs",
        help="Directory for logs and CSV outputs",
    )
    parser.add_argument(
        "--skip_baselines",
        action="store_true",
        help="Skip baseline comparisons (E1 only)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint in checkpoint_dir",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="./checkpoints",
        help="Directory for checkpoint files",
    )
    parser.add_argument(
        "--pretrain_rounds",
        type=int,
        default=None,
        help="Load pretrained orchestrator with this many rounds (e.g., 200, 500)",
    )
    parser.add_argument(
        "--pretrain_dir",
        default="pretrainppo",
        help="Directory containing pretrained models (default: pretrainppo)",
    )
    parser.add_argument(
        "--offline_haso",
        action="store_true",
        help="Use PPO for orchestration only (disable online PPO updates during experiment).",
    )
    parser.add_argument(
        "--cluster_size",
        type=int,
        default=0,
        help="Nodes per cluster for hierarchical HASO (e.g., 5). 0=disabled (default per-node agents). Must divide n_nodes evenly.",
    )
    parser.add_argument(
        "--ppo_device",
        default=None,
        choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1"],
        help="Device for PPO policies (default: auto).",
    )
    return parser.parse_args()


def run_exp(exp_name: str, args) -> None:
    """Run a single experiment."""
    print(f"\n{'#' * 60}")
    print(f"# Running Experiment: {exp_name.upper()}")
    print(f"{'#' * 60}")

    # Build base config
    config = build_config(
        n_nodes=args.n_nodes or 10,
        global_rounds=args.global_rounds or 30,
        seed=args.seed,
    )

    # Override from YAML if provided
    if args.config:
        yaml_config = load_config(args.config)
        config.update(yaml_config)

    # Apply CLI overrides
    if args.n_nodes is not None:
        config["n_nodes"] = args.n_nodes
    if args.global_rounds is not None:
        config["global_rounds"] = args.global_rounds
    if args.alpha is not None:
        config["dirichlet_alpha"] = args.alpha
    if args.lazy_fraction is not None:
        config["lazy_client_fraction"] = args.lazy_fraction
    if args.cluster_size is not None and args.cluster_size > 0:
        config["cluster_size"] = args.cluster_size
    if args.ppo_device is not None:
        config["ppo_device"] = args.ppo_device
    if args.offline_haso:
        config["haso_online_update"] = False
    config["seed"] = args.seed
    config["log_dir"] = args.log_dir

    # Ensure log dir
    ensure_dir(args.log_dir)
    ensure_dir(args.checkpoint_dir)

    # Load pretrained orchestrator if specified
    pretrained_orchestrator = None
    if args.pretrain_rounds:
        from pretrain_pipeline import (
            check_pretrained_exists,
            load_orchestrator,
            pretrain_ppo,
            zip_pretrain,
        )
        from src.haso.orchestrator import create_orchestrator

        n_nodes = config["n_nodes"]

        if check_pretrained_exists(args.pretrain_rounds, args.pretrain_dir):
            print(f"[{exp_name}] Loading pretrained orchestrator: {args.pretrain_rounds} rounds")
            pretrained_orchestrator = load_orchestrator(
                rounds=args.pretrain_rounds,
                n_nodes=n_nodes,
                config=config,
                base_dir=args.pretrain_dir,
            )
            if pretrained_orchestrator:
                print(f"[{exp_name}] SUCCESS: Loaded pretrained orchestrator")
            else:
                print(f"[{exp_name}] Failed to load, training fresh...")
                orchestrator = pretrain_ppo(
                    n_nodes=n_nodes,
                    pretrain_rounds=args.pretrain_rounds,
                    seed=args.seed,
                    force_retrain=True,
                )
                zip_pretrain(args.pretrain_rounds, args.pretrain_dir)
                pretrained_orchestrator = load_orchestrator(
                    rounds=args.pretrain_rounds,
                    n_nodes=n_nodes,
                    config=config,
                    base_dir=args.pretrain_dir,
                )
        else:
            print(f"[{exp_name}] No pretrained model found at pretrainppo/{args.pretrain_rounds}/")
            print(f"[{exp_name}] Starting pretrain first...")
            orchestrator = pretrain_ppo(
                n_nodes=n_nodes,
                pretrain_rounds=args.pretrain_rounds,
                seed=args.seed,
                force_retrain=False,
            )
            zip_path = zip_pretrain(args.pretrain_rounds, args.pretrain_dir)
            print(f"[{exp_name}] Pretrained and zipped to: {zip_path}")
            pretrained_orchestrator = load_orchestrator(
                rounds=args.pretrain_rounds,
                n_nodes=n_nodes,
                config=config,
                base_dir=args.pretrain_dir,
            )

    # Resume from checkpoint if requested
    if args.resume:
        checkpoint_path = _get_latest_checkpoint(args.checkpoint_dir, exp_name)
        if checkpoint_path:
            print(f"[{exp_name}] Resuming from checkpoint: {checkpoint_path}")
        else:
            print(f"[{exp_name}] No checkpoint found in {args.checkpoint_dir}, starting fresh")

    # Dispatch
    module = EXPERIMENT_MAP[exp_name]

    if exp_name == "e1":
        module.run(
            config,
            skip_baselines=getattr(args, "skip_baselines", False),
            resume=args.resume,
            checkpoint_dir=args.checkpoint_dir,
            pretrained_orchestrator=pretrained_orchestrator,
            pretrain_dir=args.pretrain_dir,
        )
    elif exp_name == "e6":
        module.run(
            config,
            ablation_type=args.ablation,
            resume=args.resume,
            checkpoint_dir=args.checkpoint_dir,
            pretrained_orchestrator=pretrained_orchestrator,
            pretrain_dir=args.pretrain_dir,
        )
    elif exp_name == "e2":
        module.run(config, pretrained_orchestrator=pretrained_orchestrator, pretrain_dir=args.pretrain_dir)
    else:
        module.run(config, pretrained_orchestrator=pretrained_orchestrator, pretrain_dir=args.pretrain_dir)

    print(f"[{exp_name}] Done!")


def _get_latest_checkpoint(checkpoint_dir: str, exp_name: str) -> str:
    """Find latest checkpoint for an experiment."""
    import os
    prefix = f"{exp_name}_checkpoint_"
    if not os.path.exists(checkpoint_dir):
        return ""
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith(prefix) and f.endswith(".pkl")]
    if not files:
        return ""
    files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    return os.path.join(checkpoint_dir, files[-1])


def main():
    args = parse_args()

    if args.exp == "all":
        # Run all experiments sequentially
        for exp_name in ["e1", "e2", "e3", "e4", "e5", "e6", "e7"]:
            run_exp(exp_name, args)
            print()  # blank line between experiments
    else:
        run_exp(args.exp, args)


if __name__ == "__main__":
    main()
