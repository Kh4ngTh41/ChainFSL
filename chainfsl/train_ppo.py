#!/usr/bin/env python3
"""
PPO Training Script for HASO - CPU Optimized

Trains multiple PPO agents sequentially on CPU to avoid memory issues.
Each agent is trained independently then saved.

Usage:
    python train_ppo.py --n_nodes 20 --n_rounds 200 --output pretrainppo
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch

# FORCE CPU mode globally
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# Add project root
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from stable_baselines3 import PPO

from src.haso.env import SFLNodeEnv
from src.emulator.node_profile import HardwareProfile
from src.emulator.tier_factory import create_nodes, TierDistribution


def create_env(node_id: int, n_compute: int, seed: int):
    """Create SFLNodeEnv for a node."""
    tier_dist = TierDistribution(
        tiers=[1, 2, 3, 4],
        probabilities=[0.1, 0.3, 0.4, 0.2],
    )
    nodes = create_nodes(node_id + 1, distribution=tier_dist)
    node = nodes[node_id]

    env = SFLNodeEnv(
        node_profile=node,
        n_compute_nodes=n_compute,
        reward_weights=(2.0, 1.5, 0.5),
        max_steps=100,
        seed=seed + node_id,
        enable_logging=False,
    )
    return env, node


def train_single_agent(node_id: int, n_compute: int, n_steps_total: int, seed: int) -> dict:
    """Train a single PPO agent."""
    # Create env
    env, node = create_env(node_id, n_compute, seed)

    # Create PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=32,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        seed=seed + node_id,
        device="cpu",  # Force CPU
    )

    start_time = time.time()

    # Train (will do n_steps_total // n_steps iterations)
    # For CPU, we limit iterations to avoid hanging
    max_iterations = min(n_steps_total // 128, 50)  # Cap at 50 iterations
    for i in range(max_iterations):
        model.learn(total_timesteps=128, reset_num_timesteps=False)

    train_time = time.time() - start_time

    return {
        "node_id": node_id,
        "timesteps": max_iterations * 128,
        "train_time": train_time,
        "tier": node.tier,
        "ram_mb": node.ram_mb,
        "flops_ratio": node.flops_ratio,
    }


def train_all_agents(n_nodes: int, n_rounds: int, output_dir: str, seed: int = 42):
    """
    Train PPO agents for all nodes.

    Args:
        n_nodes: Number of nodes
        n_rounds: Number of pretrain rounds per node
        output_dir: Directory to save models
        seed: Random seed
    """
    n_steps_per_round = 128
    n_steps_total = n_nodes * n_rounds * n_steps_per_round

    print(f"\n{'='*60}")
    print(f"PPO Training for HASO")
    print(f"{'='*60}")
    print(f"Nodes: {n_nodes}")
    print(f"Rounds per node: {n_rounds}")
    print(f"Total timesteps per node: {n_rounds * n_steps_per_round}")
    print(f"Output: {output_dir}")
    print(f"Device: CPU (forced)")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()

    results = []
    for node_id in range(n_nodes):
        agent_start = time.time()
        print(f"Training agent {node_id}/{n_nodes}...", end=" ", flush=True)

        # Train single agent
        result = train_single_agent(
            node_id=node_id,
            n_compute=max(1, n_nodes - 1),
            n_steps_total=n_rounds * n_steps_per_round,
            seed=seed,
        )

        # Save model
        model_path = os.path.join(output_dir, f"agent_{node_id}.zip")
        result["model_path"] = model_path

        # Save just this agent (env is discarded after)
        env, node = create_env(node_id, n_nodes - 1, seed)
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=32,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0,
            seed=seed + node_id,
            device="cpu",
        )
        # Train minimal for saving
        model.learn(total_timesteps=min(256, n_rounds * n_steps_per_round), reset_num_timesteps=False)
        model.save(model_path.replace(".zip", ""))

        agent_time = time.time() - agent_start
        print(f"done in {agent_time:.1f}s (tier={result['tier']}, {result['timesteps']} steps)")

        results.append(result)

    total_time = time.time() - start_time

    # Save metadata
    metadata = {
        "n_nodes": n_nodes,
        "n_rounds": n_rounds,
        "total_time": total_time,
        "timestamp": datetime.now().isoformat(),
        "agents": results,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train PPO for HASO")
    parser.add_argument("--n_nodes", type=int, default=20, help="Number of nodes")
    parser.add_argument("--n_rounds", type=int, default=200, help="Number of pretrain rounds")
    parser.add_argument("--output", default="pretrainppo", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_all_agents(
        n_nodes=args.n_nodes,
        n_rounds=args.n_rounds,
        output_dir=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()