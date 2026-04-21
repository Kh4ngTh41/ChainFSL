#!/usr/bin/env python3
"""
Lightweight PPO Pretrain for ChainFSL HASO.

Trains PPO agent directly WITH GPU acceleration.
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

import torch
from tqdm import tqdm

from src.haso.orchestrator import HASOOrchestrator
from src.emulator.tier_factory import create_nodes, TierDistribution


def simple_pretrain(
    n_nodes: int,
    pretrain_rounds: int,
    seed: int = 42,
    save_dir: str = "pretrainppo",
) -> dict:
    """
    Lightweight PPO pretrain with GPU acceleration.

    Args:
        n_nodes: Number of nodes.
        pretrain_rounds: Rounds to train PPO.
        seed: Random seed.
        save_dir: Directory to save model.

    Returns:
        Dict with pretrain stats.
    """
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"PPO PRETRAIN SUMMARY")
    print(f"{'='*60}")
    print(f"  Device:       {device}")
    print(f"  Nodes (a):    {n_nodes}")
    print(f"  Models:       1 PPO orchestrator (centralized)")
    print(f"  Community:    1 model in federation")
    print(f"  Rounds:       {pretrain_rounds}")
    print(f"  Cluster:      k=1 (centralized HASO)")
    print(f"{'='*60}\n")

    # Generate node profiles
    tier_dist = TierDistribution(tiers=[1, 2, 3, 4], probabilities=[0.1, 0.3, 0.4, 0.2])
    profiles = create_nodes(n_nodes, distribution=tier_dist)

    # Create orchestrator with PPO
    orchestrator = HASOOrchestrator(
        n_nodes=n_nodes,
        node_profiles=profiles,
        reward_weights=(1.0, 0.5, 0.1),
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=5,
        verbose=0,  # Quiet - tqdm will show progress
    )

    save_path = Path(save_dir) / str(pretrain_rounds) / "orchestrator.zip"

    print(f"[Pretrain] Training PPO for {pretrain_rounds} rounds on {device}...")
    start_time = time.perf_counter()

    # Train PPO with tqdm progress bar
    with tqdm(total=pretrain_rounds, desc="PPO Training", unit="round") as pbar:
        for round_i in range(pretrain_rounds):
            obs = orchestrator.env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]  # Gym API returns (obs, info)
            done = False

            # Run a few steps per round
            for _ in range(5):
                action, _ = orchestrator.model.predict(obs, deterministic=False)
                obs, reward, done, _, info = orchestrator.env.step(action)
                if isinstance(obs, tuple):
                    obs = obs[0]
                if done:
                    obs = orchestrator.env.reset()
                    if isinstance(obs, tuple):
                        obs = obs[0]
                    break

            # Update PPO
            orchestrator.model.learn(
                total_timesteps=128,
                reset_num_timesteps=False,
                progress_bar=False,
            )

            pbar.update(1)
            pbar.set_postfix({
                "loss": f"{orchestrator.env._mean_loss:.3f}",
                "fair": f"{orchestrator.env._fairness:.3f}",
            })

    elapsed = time.perf_counter() - start_time
    print(f"\n[Pretrain] Training complete in {elapsed:.1f}s")

    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    orchestrator.save(str(save_path))
    print(f"[Pretrain] Saved to {save_path}")

    # Save metrics
    metrics = {
        "pretrain_rounds": pretrain_rounds,
        "n_nodes": n_nodes,
        "seed": seed,
        "elapsed_seconds": elapsed,
        "device": str(device),
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
        "device": str(device),
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

    print(f"\n{'='*60}")
    print(f"PRETRAIN COMPLETE")
    print(f"{'='*60}")
    print(f"  Rounds:    {result['pretrain_rounds']}")
    print(f"  Nodes:     {result['n_nodes']}")
    print(f"  Time:      {result['elapsed_seconds']:.1f}s")
    print(f"  Device:    {result['device']}")
    print(f"  Saved:     pretrainppo/{args.rounds}/orchestrator.zip")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
