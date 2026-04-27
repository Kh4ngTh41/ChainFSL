#!/usr/bin/env python3
"""
Complete PPO Pretrain Pipeline for ChainFSL HASO.

Workflow:
1. Check if pretrained model exists at pretrainppo/{rounds}/
2. If exists: load directly, skip training
3. If not: pretrain PPO, save to pretrainppo/{rounds}/
4. ZIP the whole directory for easy sharing
5. Experiments can load from ZIP or directory

Usage:
    python pretrain_pipeline.py --rounds 200 --n_nodes 10
    python pretrain_pipeline.py --rounds 500 --n_nodes 20 --zip_only
"""

import argparse
import shutil
import os
import sys
import json
import time
import signal
from pathlib import Path
from datetime import datetime

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.protocol.chainfsl import ChainFSLProtocol
from src.haso.orchestrator import create_orchestrator
from experiments.utils import build_config, ensure_dir


PRETRAIN_BASE = "pretrainppo"


def check_pretrained_exists(rounds: int, base_dir: str = PRETRAIN_BASE) -> bool:
    """Check if pretrained model directory exists and has valid model."""
    dir_path = Path(base_dir) / str(rounds)
    orchestrator_path = dir_path / "orchestrator.zip"
    metrics_path = dir_path / "pretrain_metrics.json"

    if not dir_path.exists():
        return False
    if not orchestrator_path.exists():
        return False
    if not metrics_path.exists():
        return False

    # Validate metrics
    with open(metrics_path) as f:
        metrics = json.load(f)
    if metrics.get("pretrain_rounds") != rounds:
        return False

    return True


def pretrain_ppo(
    n_nodes: int,
    pretrain_rounds: int,
    seed: int = 42,
    log_dir: str = "./logs",
    force_retrain: bool = False,
    ppo_device: str = "auto",
) -> dict:
    """
    Pretrain PPO orchestrator for specified rounds.

    Args:
        n_nodes: Number of nodes in the federation.
        pretrain_rounds: Number of rounds to pretrain.
        seed: Random seed.
        log_dir: Directory for logs.
        force_retrain: If True, retrain even if pretrained exists.

    Returns:
        Dict with pretrain stats.
    """
    save_dir = Path(PRETRAIN_BASE) / str(pretrain_rounds)

    # Check if already pretrained
    if not force_retrain and check_pretrained_exists(pretrain_rounds):
        print(f"[{pretrain_rounds}] Pretrained model FOUND at {save_dir}")
        print(f"[{pretrain_rounds}] Skipping training, will load from disk.")

        # Load and verify
        metrics_path = save_dir / "pretrain_metrics.json"
        with open(metrics_path) as f:
            stats = json.load(f)

        return {
            "status": "loaded",
            "save_dir": str(save_dir),
            "pretrain_rounds": pretrain_rounds,
            "elapsed_seconds": stats.get("elapsed_seconds", 0),
            "final_loss": stats.get("final_loss"),
            "final_acc": stats.get("final_acc"),
            "mean_latency": stats.get("mean_latency"),
        }

    print(f"[{pretrain_rounds}] No pretrained model found. Training from scratch...")
    print(f"  n_nodes: {n_nodes}")
    print(f"  seed: {seed}")
    print("-" * 50)

    # Ensure directories
    ensure_dir(str(save_dir))
    ensure_dir(log_dir)

    config = build_config(
        n_nodes=n_nodes,
        global_rounds=pretrain_rounds,
        haso_enabled=True,
        tve_enabled=True,
        gtm_enabled=True,
        seed=seed,
    )
    config["ppo_device"] = ppo_device
    config["log_dir"] = log_dir

    db_path = str(Path(log_dir) / f"pretrain_{seed}.db")

    start_time = time.perf_counter()

    # Create protocol
    protocol = ChainFSLProtocol(
        config=config,
        device=None,
        db_path=db_path,
    )

    # Create and set orchestrator
    orchestrator = create_orchestrator(n_nodes, protocol.nodes, config)
    protocol._orchestrator = orchestrator

    # Run pretraining
    print(f"[{pretrain_rounds}] Training PPO for {pretrain_rounds} rounds...")
    metrics_history = protocol.run(
        total_rounds=pretrain_rounds,
        eval_every=pretrain_rounds,  # Only eval at end
    )

    elapsed = time.perf_counter() - start_time

    # Save orchestrator
    orchestrator.save(str(save_dir / "orchestrator.zip"))
    print(f"[{pretrain_rounds}] Saved orchestrator to {save_dir}")

    # Save metrics
    final_metrics = metrics_history[-1] if metrics_history else None

    timing_data = {}
    if metrics_history:
        timing_data = {
            "rounds": pretrain_rounds,
            "mean_ppo_update_time": sum(p.ppo_update_time for p in metrics_history) / len(metrics_history),
            "mean_shapley_time": sum(p.shapley_time for p in metrics_history) / len(metrics_history),
            "mean_train_time": sum(p.train_time for p in metrics_history) / len(metrics_history),
            "mean_verification_time": sum(p.verification_time for p in metrics_history) / len(metrics_history),
            "mean_comm_time": sum(p.comm_time for p in metrics_history) / len(metrics_history),
            "mean_round_latency": sum(m.round_latency for m in metrics_history) / len(metrics_history),
            "per_round_latency": [m.round_latency for m in metrics_history],
            "per_round_ppo": [m.ppo_update_time for m in metrics_history],
            "per_round_shapley": [m.shapley_time for m in metrics_history],
        }

    metrics_data = {
        "pretrain_rounds": pretrain_rounds,
        "n_nodes": n_nodes,
        "seed": seed,
        "elapsed_seconds": elapsed,
        "final_round": pretrain_rounds,
        "final_loss": final_metrics.train_loss if final_metrics else None,
        "final_acc": final_metrics.test_acc if final_metrics else None,
        "mean_latency": timing_data.get("mean_round_latency"),
        "timestamp": datetime.now().isoformat(),
    }

    metrics_path = save_dir / "pretrain_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[{pretrain_rounds}] Saved metrics to {metrics_path}")

    # Save timing breakdown
    timing_path = save_dir / "timing_breakdown.json"
    with open(timing_path, "w") as f:
        json.dump(timing_data, f, indent=2)
    print(f"[{pretrain_rounds}] Saved timing breakdown to {timing_path}")

    # Save config used
    config_path = save_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 50)
    print(f"PRETRAIN COMPLETE: {pretrain_rounds} rounds")
    print("=" * 50)
    print(f"  Time: {elapsed:.1f}s")
    if final_metrics:
        print(f"  Final Loss: {final_metrics.train_loss:.4f}")
        print(f"  Final Acc: {final_metrics.test_acc:.2f}%")
    print(f"  Saved to: {save_dir}")
    print("=" * 50)

    return {
        "status": "trained",
        "save_dir": str(save_dir),
        "pretrain_rounds": pretrain_rounds,
        "elapsed_seconds": elapsed,
        "final_loss": final_metrics.train_loss if final_metrics else None,
        "final_acc": final_metrics.test_acc if final_metrics else None,
        "mean_latency": timing_data.get("mean_round_latency"),
    }


def zip_pretrain(rounds: int, base_dir: str = PRETRAIN_BASE, zip_dir: str = "zipped") -> str:
    """
    ZIP the pretrained model directory.

    Args:
        rounds: Number of pretrain rounds.
        base_dir: Base directory containing pretrainppo/.
        zip_dir: Directory to save ZIP file.

    Returns:
        Path to created ZIP file.
    """
    src_path = Path(base_dir) / str(rounds)
    if not src_path.exists():
        raise FileNotFoundError(f"Pretrain directory not found: {src_path}")

    ensure_dir(zip_dir)
    zip_path = Path(zip_dir) / f"pretrainppo_{rounds}.zip"

    print(f"Zipping {src_path} -> {zip_path}")
    shutil.make_archive(
        base_name=str(zip_path.with_suffix('')),
        format='zip',
        root_dir=src_path,
    )

    print(f"Created ZIP: {zip_path} ({os.path.getsize(zip_path) / 1024:.1f} KB)")
    return str(zip_path)


def unzip_pretrain(rounds: int, zip_path: str, base_dir: str = PRETRAIN_BASE) -> str:
    """
    Extract pretrained model from ZIP.

    Args:
        rounds: Expected pretrain rounds (for validation).
        zip_path: Path to ZIP file.
        base_dir: Base directory to extract to.

    Returns:
        Path to extracted directory.
    """
    extract_path = Path(base_dir) / str(rounds)

    if extract_path.exists():
        print(f"Extraction target exists: {extract_path}")
        print("Removing old directory...")
        shutil.rmtree(extract_path)

    ensure_dir(base_dir)

    print(f"Extracting {zip_path} -> {extract_path}")
    shutil.unpack_archive(zip_path, extract_path)

    # Validate
    if not check_pretrained_exists(rounds, base_dir):
        raise ValueError(f"Extracted content invalid for rounds={rounds}")

    print(f"Extracted successfully to: {extract_path}")
    return str(extract_path)


def load_orchestrator(rounds: int, n_nodes: int, config: dict, base_dir: str = PRETRAIN_BASE):
    """
    Load pretrained orchestrator from disk.

    Checks multiple locations in order:
    1. pretrainppo/{rounds}/orchestrator.zip (extracted)
    2. zipped/pretrainppo_{rounds}.zip (zipped)

    Args:
        rounds: Pretrain rounds.
        n_nodes: Number of nodes.
        config: Config dict.
        base_dir: Base directory.

    Returns:
        HASOOrchestrator instance, or None if not found.
    """
    from src.haso.orchestrator import HASOOrchestrator
    from src.emulator.tier_factory import create_nodes, TierDistribution

    print(f"[LOAD] Preparing to load orchestrator (rounds={rounds}, n_nodes={n_nodes})", flush=True)

    # Try extracted directory first
    extracted_path = Path(base_dir) / str(rounds) / "orchestrator.zip"

    # Try ZIP
    zip_path = Path("zipped") / f"pretrainppo_{rounds}.zip"

    orchestrator_path = None

    if extracted_path.exists():
        orchestrator_path = extracted_path
        print(f"[LOAD] Found orchestrator at: {extracted_path}")
    elif zip_path.exists():
        # Extract ZIP first
        extract_path = unzip_pretrain(rounds, str(zip_path), base_dir)
        orchestrator_path = Path(extract_path) / "orchestrator.zip"
        print(f"[LOAD] Extracted from ZIP: {orchestrator_path}")
    else:
        print(f"[LOAD] No pretrained model found for {rounds} rounds")
        return None

    # Create proper node profiles for actual n_nodes needed by experiment
    print("[LOAD] Creating node profiles...", flush=True)
    tier_dist = TierDistribution(tiers=[1, 2, 3, 4], probabilities=[0.1, 0.3, 0.4, 0.2])
    node_profiles = create_nodes(n_nodes, distribution=tier_dist)

    # Determine original n_nodes the model was pre-trained on to prevent shape mismatch in PPO.load
    pretrained_n_nodes = n_nodes
    config_path = Path(base_dir) / str(rounds) / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                pretrained_n_nodes = json.load(f).get("n_nodes", n_nodes)
                print(f"[LOAD] Found original pre-trained n_nodes={pretrained_n_nodes} to avoid shape mismatches", flush=True)
        except Exception:
            pass

    # Create orchestrator and load
    print(f"[LOAD] Creating orchestrator with pre-trained n_nodes={pretrained_n_nodes} and ppo_device={config.get('ppo_device', 'auto')}...", flush=True)
    orchestrator = create_orchestrator(pretrained_n_nodes, node_profiles, config)
    print("[LOAD] Loading PPO weights into orchestrator...", flush=True)

    timeout_sec = int(config.get("pretrain_load_timeout_sec", 180))

    def _on_timeout(signum, frame):
        raise TimeoutError(f"orchestrator.load timed out after {timeout_sec}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        if timeout_sec > 0:
            signal.signal(signal.SIGALRM, _on_timeout)
            signal.alarm(timeout_sec)
        orchestrator.load(str(orchestrator_path))
    except TimeoutError as e:
        print(f"[LOAD] ERROR: {e}", flush=True)
        print("[LOAD] Falling back: return None to allow non-blocking experiment startup.", flush=True)
        return None
    except Exception as e:
        print(f"[LOAD] ERROR while loading orchestrator: {e}", flush=True)
        print("[LOAD] Falling back: return None to allow non-blocking experiment startup.", flush=True)
        return None
    finally:
        if timeout_sec > 0:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    print(f"[LOAD] Successfully loaded orchestrator from {orchestrator_path}", flush=True)
    return orchestrator


def analyze_pretrain_breakeven(
    pretrain_dir: str = PRETRAIN_BASE,
    baseline_latency: float = 137.0,
    nohaso_rounds: int = 50,
):
    """Analyze breakeven point for pretrained models."""
    import os

    print("\n" + "=" * 70)
    print("PPO BREAKEVEN ANALYSIS")
    print("=" * 70)
    print(f"Baseline (NoHASO) latency: {baseline_latency}s per round")
    print(f"Experiment rounds: {nohaso_rounds}")
    print()

    if not os.path.exists(pretrain_dir):
        print(f"Error: {pretrain_dir} not found. Run pretrain first.")
        return

    for rounds_dir in sorted(os.listdir(pretrain_dir)):
        metrics_path = Path(pretrain_dir) / rounds_dir / "pretrain_metrics.json"
        timing_path = Path(pretrain_dir) / rounds_dir / "timing_breakdown.json"

        if not metrics_path.exists():
            continue

        with open(metrics_path) as f:
            data = json.load(f)

        mean_lat = data.get("mean_latency", 0)
        elapsed = data.get("elapsed_seconds", 0)
        pretrain_rounds = int(rounds_dir)

        if mean_lat is None or mean_lat == 0:
            continue

        improvement = (baseline_latency - mean_lat) / baseline_latency * 100
        is_faster = mean_lat < baseline_latency

        print(f"  Pretrained {rounds_dir:>6} rounds:")
        print(f"    Mean latency: {mean_lat:.2f}s ({'+' if is_faster else ''}{improvement:.2f}%)")
        print(f"    Pretrain time: {elapsed:.1f}s")
        print(f"    Per-round savings vs NoHASO: {baseline_latency - mean_lat:.2f}s")

        if is_faster:
            savings_per_round = baseline_latency - mean_lat
            breakeven_rounds = elapsed / savings_per_round if savings_per_round > 0 else float('inf')
            print(f"    Breakeven after: {breakeven_rounds:.0f} experiment rounds "
                  f"({'✓ profiting' if breakeven_rounds < nohaso_rounds else f'✗ not profitable in {nohaso_rounds}rounds'})")

        if timing_path and timing_path.exists():
            with open(timing_path) as f:
                tdata = json.load(f)
            print(f"    Timing breakdown:")
            print(f"      PPO update: {tdata.get('mean_ppo_update_time', 0):.2f}s")
            print(f"      Shapley:     {tdata.get('mean_shapley_time', 0):.2f}s")
            print(f"      Train:       {tdata.get('mean_train_time', 0):.2f}s")
            print(f"      Verify:      {tdata.get('mean_verification_time', 0):.2f}s")

        print()

    print("Per-round latency progression (min/max/final):")
    print("-" * 70)
    for rounds_dir in sorted(os.listdir(pretrain_dir)):
        timing_path = Path(pretrain_dir) / rounds_dir / "timing_breakdown.json"
        if not timing_path:
            continue
        with open(timing_path) as f:
            tdata = json.load(f)

        per_round = tdata.get("per_round_latency", [])
        if not per_round:
            continue

        print(f"  {rounds_dir:>6} rounds: min={min(per_round):.1f}s, "
              f"max={max(per_round):.1f}s, final={per_round[-1]:.1f}s")

    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Complete PPO Pretrain Pipeline for ChainFSL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pretrain for 200 rounds (skip if already pretrained)
  python pretrain_pipeline.py --rounds 200 --n_nodes 10

  # Force retrain even if model exists
  python pretrain_pipeline.py --rounds 200 --force

  # ZIP existing pretrained model
  python pretrain_pipeline.py --rounds 200 --zip

  # Load and verify pretrained model exists
  python pretrain_pipeline.py --rounds 200 --check

  # Analyze breakeven
  python pretrain_pipeline.py --analyze

  # Pretrain multiple configs
  python pretrain_pipeline.py --rounds 100 --rounds 200 --rounds 500
        """,
    )
    parser.add_argument(
        "--rounds",
        type=int,
        nargs='+',
        default=[200],
        help="Number of rounds to pretrain (default: 200)",
    )
    parser.add_argument(
        "--n_nodes",
        type=int,
        default=10,
        help="Number of nodes (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--log_dir",
        default="./logs",
        help="Directory for logs (default: ./logs)",
    )
    parser.add_argument(
        "--ppo_device",
        default="auto",
        choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1"],
        help="Device for PPO pretraining (default: auto)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retrain even if pretrained exists",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="ZIP existing pretrained model",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if pretrained model exists",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze breakeven point",
    )

    args = parser.parse_args()

    if args.analyze:
        analyze_pretrain_breakeven()
        return

    for rounds in args.rounds:
        print(f"\n{'='*60}")
        print(f"Processing: --rounds {rounds} --n_nodes {args.n_nodes}")
        print(f"{'='*60}")

        if args.check:
            exists = check_pretrained_exists(rounds)
            status = "EXISTS" if exists else "NOT FOUND"
            print(f"[{rounds}] Pretrained model: {status}")
        elif args.zip:
            zip_path = zip_pretrain(rounds)
            print(f"[{rounds}] ZIP created: {zip_path}")
        else:
            result = pretrain_ppo(
                n_nodes=args.n_nodes,
                pretrain_rounds=rounds,
                seed=args.seed,
                log_dir=args.log_dir,
                force_retrain=args.force,
                ppo_device=args.ppo_device,
            )
            print(f"[{rounds}] Status: {result['status']}")
            print(f"[{rounds}] Saved to: {result['save_dir']}")


if __name__ == "__main__":
    main()