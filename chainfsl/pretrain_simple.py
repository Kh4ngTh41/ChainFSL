#!/usr/bin/env python3
"""
Lightweight PPO Pretrain for ChainFSL HASO.

Only trains the PPO agent WITHOUT full TVE/Shapley overhead.
This is for pretraining the HASO decision policy only.

Usage:
    python pretrain_simple.py --rounds 100 --n_nodes 10
"""

import argparse
import time
import json
from pathlib import Path
from datetime import datetime

_PROJECT_ROOT = Path(__file__).parent
import sys
sys.path.insert(0, str(_PROJECT_ROOT))

from src.haso.orchestrator import HASOOrchestrator
from src.haso.cluster import ClusterManager
from src.emulator.node_profile import generate_hardware_profiles


def simple_pretrain(
    n_nodes: int,
    pretrain_rounds: int,
    seed: int = 42,
    log_dir: str = "./logs",
) -> dict:
    """
    Lightweight PPO pretrain - no TVE/Shapley overhead.

    Args:
        n_nodes: Number of nodes.
        pretrain_rounds: Rounds to train PPO.
        seed: Random seed.
        log_dir: Directory for logs.

    Returns:
        Dict with pretrain stats.
    """
    print(f"[Pretrain] Lightweight mode: {pretrain_rounds} rounds, {n_nodes} nodes")

    # Generate fake node profiles
    profiles = generate_hardware_profiles(n_nodes, seed=seed)

    # Create orchestrator
    orchestrator = HASOOrchestrator(
        n_nodes=n_nodes,
        node_profiles=profiles,
        reward_weights=(1.0, 0.5, 0.1),
        learning_rate=3e-4,
        n_steps=128,  # Smaller for speed
        batch_size=32,  # Smaller for speed
        n_epochs=5,    # Fewer epochs
        verbose=1,
    )

    print(f"[Pretrain] Training PPO...")
    start_time = time.perf_counter()

    # Train PPO directly without full protocol
    # Each "step" = one HASO decision round
    for round_i in range(pretrain_rounds):
        # Simulate a training round
        obs = orchestrator.env.reset()
        done = False
        total_reward = 0

        # Run a few steps per round
        for step_i in range(3):
            action, _ = orchestrator.model.predict(obs, deterministic=False)
            obs, reward, done, _, info = orchestrator.env.step(action)
            total_reward += reward
            if done:
                obs = orchestrator.env.reset()

        # Learn after each round
        orchestrator.model.learn(
            total_timesteps=128,
            reset_num_timesteps=False,
            progress_bar=False,
        )

        if (round_i + 1) % 20 == 0:
            elapsed = time.perf_counter() - start_time
            print(f"  Round {round_i+1}/{pretrain_rounds} - elapsed: {elapsed:.1f}s")

    elapsed = time.perf_counter() - start_time
    print(f"[Pretrain] Done in {elapsed:.1f}s")

    return {
        "status": "trained",
        "elapsed_seconds": elapsed,
        "pretrain_rounds": pretrain_rounds,
        "n_nodes": n_nodes,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description="Lightweight PPO Pretrain")
    parser.add_argument("--rounds", type=int, default=100, help="Pretrain rounds")
    parser.add_argument("--n_nodes", type=int, default=10, help="Number of nodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_dir", default="./logs", help="Log directory")
    parser.add_argument("--save_dir", default="pretrainppo", help="Save directory")

    args = parser.parse_args()

    # Create save dir
    save_dir = Path(args.save_dir) / str(args.rounds)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Run pretrain
    result = simple_pretrain(
        n_nodes=args.n_nodes,
        pretrain_rounds=args.rounds,
        seed=args.seed,
        log_dir=args.log_dir,
    )

    # Create orchestrator again to save
    profiles = generate_hardware_profiles(args.n_nodes, seed=args.seed)
    orchestrator = HASOOrchestrator(
        n_nodes=args.n_nodes,
        node_profiles=profiles,
        verbose=0,
    )

    # Train briefly to initialize
    orchestrator.model.learn(total_timesteps=100, reset_num_timesteps=True, progress_bar=False)

    # Save
    save_path = save_dir / "orchestrator.zip"
    orchestrator.save(str(save_path))
    print(f"[Pretrain] Saved to {save_path}")

    # Save metrics
    metrics = {
        "pretrain_rounds": args.rounds,
        "n_nodes": args.n_nodes,
        "seed": args.seed,
        "elapsed_seconds": result["elapsed_seconds"],
        "timestamp": datetime.now().isoformat(),
    }
    with open(save_dir / "pretrain_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[Pretrain] Complete!")


if __name__ == "__main__":
    main()
