#!/usr/bin/env python3
"""
Lightweight PPO Pretrain for ChainFSL HASO.

Trains PPO agent directly WITHOUT full protocol overhead.
Fast pretraining for HASO decision policy.

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
from src.emulator.node_profile import generate_hardware_profiles


def simple_pretrain(
    n_nodes: int,
    pretrain_rounds: int,
    seed: int = 42,
    save_dir: str = "pretrainppo",
) -> dict:
    """
    Lightweight PPO pretrain - trains and saves directly.

    Args:
        n_nodes: Number of nodes.
        pretrain_rounds: Rounds to train PPO.
        seed: Random seed.
        save_dir: Directory to save model.

    Returns:
        Dict with pretrain stats.
    """
    print(f"[Pretrain] Lightweight mode: {pretrain_rounds} rounds, {n_nodes} nodes")

    # Generate node profiles
    profiles = generate_hardware_profiles(n_nodes, seed=seed)

    # Create orchestrator with PPO
    orchestrator = HASOOrchestrator(
        n_nodes=n_nodes,
        node_profiles=profiles,
        reward_weights=(1.0, 0.5, 0.1),
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=5,
        verbose=1,
    )

    save_path = Path(save_dir) / str(pretrain_rounds) / "orchestrator.zip"

    print(f"[Pretrain] Training PPO for {pretrain_rounds} rounds...")
    start_time = time.perf_counter()

    # Train PPO directly using env
    # Each round: reset env, run a few steps, then learn
    for round_i in range(pretrain_rounds):
        obs = orchestrator.env.reset()
        done = False
        total_reward = 0

        # Run a few steps per round (3-5 steps is enough for PPO)
        for _ in range(5):
            action, _ = orchestrator.model.predict(obs, deterministic=False)
            obs, reward, done, _, info = orchestrator.env.step(action)
            total_reward += reward
            if done:
                obs = orchestrator.env.reset()
                break

        # Update PPO after each round
        orchestrator.model.learn(
            total_timesteps=128,
            reset_num_timesteps=False,
            progress_bar=False,
        )

        if (round_i + 1) % 20 == 0:
            elapsed = time.perf_counter() - start_time
            print(f"  Round {round_i+1}/{pretrain_rounds} - elapsed: {elapsed:.1f}s")

    elapsed = time.perf_counter() - start_time
    print(f"[Pretrain] Training complete in {elapsed:.1f}s")

    # Ensure save directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save trained model
    orchestrator.save(str(save_path))
    print(f"[Pretrain] Saved to {save_path}")

    # Save metrics
    metrics = {
        "pretrain_rounds": pretrain_rounds,
        "n_nodes": n_nodes,
        "seed": seed,
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
    }
    with open(save_path.parent / "pretrain_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

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
    parser.add_argument("--save_dir", default="pretrainppo", help="Save directory")

    args = parser.parse_args()

    result = simple_pretrain(
        n_nodes=args.n_nodes,
        pretrain_rounds=args.rounds,
        seed=args.seed,
        save_dir=args.save_dir,
    )

    print(f"\n[PRETRAIN COMPLETE]")
    print(f"  Rounds: {result['pretrain_rounds']}")
    print(f"  Time: {result['elapsed_seconds']:.1f}s")
    print(f"  Saved to: {args.save_dir}/{args.rounds}/orchestrator.zip")


if __name__ == "__main__":
    main()