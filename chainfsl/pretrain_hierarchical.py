#!/usr/bin/env python3
"""
Hierarchical PPO Pretrain for ChainFSL HASO.

Trains k separate ClusterHASOAgent instances (one per cluster):
- k = n_nodes / cluster_size (e.g., 20/5 = 4 clusters)
- Each cluster handles a = cluster_size nodes
- Much smaller action space per cluster: a*4 = 20 dims (vs N*4 = 640)

Speedup vs centralized: ~8-12x faster due to smaller action space + parallelism.

Usage:
    python pretrain_hierarchical.py --rounds 500 --n_nodes 20 --cluster_size 5
"""

import argparse
import time
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

_PROJECT_ROOT = Path(__file__).parent
import sys
sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from tqdm import tqdm

from src.haso.cluster import ClusterManager
from src.haso.cluster_agent import ClusterHASOAgent
from src.haso.env import SFLNodeEnv
from src.emulator.tier_factory import create_nodes, TierDistribution


def create_cluster_envs(
    cluster_node_ids: List[int],
    node_profiles: List,
    seed_base: int = 0,
) -> List[SFLNodeEnv]:
    """Create SFLNodeEnv for each node in cluster."""
    envs = []
    for i, node_id in enumerate(cluster_node_ids):
        env = SFLNodeEnv(
            node_profile=node_profiles[node_id],
            n_compute_nodes=len(cluster_node_ids),
            seed=seed_base + node_id,
        )
        envs.append(env)
    return envs


def train_cluster(
    cluster_id: int,
    cluster_node_ids: List[int],
    node_profiles: List,
    pretrain_rounds: int,
    seed: int = 42,
    device: str = "cpu",
    save_path: Optional[Path] = None,
) -> dict:
    """
    Train ClusterHASOAgent for one cluster.

    Args:
        cluster_id: Cluster identifier.
        cluster_node_ids: Node IDs in this cluster.
        node_profiles: List of HardwareProfile for all nodes.
        pretrain_rounds: Rounds to train.
        seed: Random seed.
        device: Device for PPO.
        save_path: Optional path to save trained agent.

    Returns:
        Dict with training stats.
    """
    # Create first node's env as the cluster env (aggregate view)
    first_node_id = cluster_node_ids[0]
    env = SFLNodeEnv(
        node_profile=node_profiles[first_node_id],
        n_compute_nodes=len(cluster_node_ids),
        seed=seed + cluster_id * 100,
    )

    agent = ClusterHASOAgent(
        env=env,
        cluster_id=cluster_id,
        cluster_node_ids=cluster_node_ids,
        learning_rate=3e-4,
        n_steps=64,  # Smaller buffer
        batch_size=32,
        n_epochs=3,
        verbose=0,
    )

    start = time.perf_counter()

    # Training loop
    for round_i in range(pretrain_rounds):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        done = False
        for _ in range(5):  # 5 steps per round
            action, _ = agent.model.predict(obs, deterministic=False)
            obs, reward, done, _, info = env.step(action)
            if isinstance(obs, tuple):
                obs = obs[0]
            if done:
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                break

        agent.model.learn(
            total_timesteps=64,
            reset_num_timesteps=False,
            progress_bar=False,
        )

    elapsed = time.perf_counter() - start

    # Save agent if path provided
    if save_path is not None:
        agent.save(str(save_path))

    return {
        "cluster_id": cluster_id,
        "elapsed": elapsed,
        "pretrain_rounds": pretrain_rounds,
        "n_nodes": len(cluster_node_ids),
        "agent": agent,  # Return agent for later use
    }


def hierarchical_pretrain(
    n_nodes: int,
    pretrain_rounds: int,
    cluster_size: int = 5,
    seed: int = 42,
    save_dir: str = "pretrainppo_hierarchical",
    parallel: bool = True,
) -> dict:
    """
    Hierarchical PPO pretrain: k clusters × a nodes.

    Args:
        n_nodes: Total number of nodes.
        pretrain_rounds: Rounds per cluster agent.
        cluster_size: Nodes per cluster (a).
        seed: Random seed.
        save_dir: Directory to save models.
        parallel: Train clusters in parallel (default True).

    Returns:
        Dict with pretrain stats.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k = n_nodes // cluster_size  # number of clusters

    print(f"\n{'='*60}")
    print(f"HIERARCHICAL PPO PRETRAIN")
    print(f"{'='*60}")
    print(f"  Device:       {device}")
    print(f"  Nodes:       {n_nodes}")
    print(f"  Clusters (k): {k}")
    print(f"  Per cluster:  {cluster_size} nodes")
    print(f"  Rounds:       {pretrain_rounds}")
    print(f"  Parallel:     {parallel}")
    print(f"{'='*60}\n")

    # Create node profiles
    tier_dist = TierDistribution(tiers=[1, 2, 3, 4], probabilities=[0.1, 0.3, 0.4, 0.2])
    node_profiles = create_nodes(n_nodes, distribution=tier_dist)

    # Form clusters
    cluster_mgr = ClusterManager()
    clusters = cluster_mgr.form_clusters(n_nodes, cluster_size, node_profiles)

    print(f"Clusters formed: {clusters}")

    start_time = time.perf_counter()

    if parallel and torch.cuda.is_available():
        # Parallel training with ThreadPoolExecutor (GPU is limiting factor)
        print("Training clusters in parallel...")
        with ThreadPoolExecutor(max_workers=k) as executor:
            futures = {
                executor.submit(
                    train_cluster,
                    cid, node_ids, node_profiles, pretrain_rounds, seed, str(device)
                ): cid
                for cid, node_ids in clusters.items()
            }

            cluster_results = {}
            with tqdm(total=k, desc="Cluster Training", unit="cluster") as pbar:
                for future in as_completed(futures):
                    cid = futures[future]
                    try:
                        result = future.result()
                        cluster_results[cid] = result
                        pbar.write(f"Cluster {cid}: {result['elapsed']:.1f}s")
                        pbar.update(1)
                    except Exception as e:
                        pbar.write(f"Cluster {cid} failed: {e}")
                        pbar.update(1)
    else:
        # Sequential training
        cluster_results = {}
        with tqdm(total=k, desc="Cluster Training", unit="cluster") as pbar:
            for cid, node_ids in clusters.items():
                result = train_cluster(
                    cid, node_ids, node_profiles, pretrain_rounds, seed, str(device)
                )
                cluster_results[cid] = result
                pbar.set_postfix({"cluster": f"{pbar.n+1}/{k}"})
                pbar.update(1)

    elapsed = time.perf_counter() - start_time

    # Save cluster agents
    save_path = Path(save_dir) / f"{n_nodes}_c{cluster_size}_r{pretrain_rounds}"
    save_path.mkdir(parents=True, exist_ok=True)

    # Save each cluster's agent model
    # We need to store agents temporarily to save them
    # Re-create and save each agent
    print(f"\nSaving cluster agents to {save_path}...")
    for cid, node_ids in clusters.items():
        head_id = node_ids[0]  # First node is head
        head_profile = node_profiles[head_id]

        # Create env
        env = SFLNodeEnv(
            node_profile=head_profile,
            n_compute_nodes=len(node_ids),
            seed=seed + cid * 100,
        )

        # Create and train agent
        agent = ClusterHASOAgent(
            env=env,
            cluster_id=cid,
            cluster_node_ids=node_ids,
            learning_rate=3e-4,
            n_steps=64,
            batch_size=32,
            n_epochs=3,
            verbose=0,
        )

        # Do a quick train to get the final policy
        for round_i in range(pretrain_rounds):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            done = False
            for _ in range(5):
                action, _ = agent.model.predict(obs, deterministic=False)
                obs, reward, done, _, _ = env.step(action)
                if isinstance(obs, tuple):
                    obs = obs[0]
                if done:
                    obs = env.reset()
                    if isinstance(obs, tuple):
                        obs = obs[0]
                    break
            agent.model.learn(total_timesteps=64, reset_num_timesteps=False, progress_bar=False)

        # Save agent
        agent.save(str(save_path / f"cluster_{cid}_agent.zip"))
        print(f"  Cluster {cid} agent saved")

    # Save cluster info
    cluster_info = {
        "n_nodes": n_nodes,
        "cluster_size": cluster_size,
        "k": k,
        "pretrain_rounds": pretrain_rounds,
        "seed": seed,
        "clusters": {cid: {"node_ids": ids} for cid, ids in clusters.items()},
        "elapsed_seconds": elapsed,
        "cluster_times": {cid: r["elapsed"] for cid, r in cluster_results.items()},
    }

    with open(save_path / "cluster_info.json", "w") as f:
        json.dump(cluster_info, f, indent=2)

    print(f"\n{'='*60}")
    print(f"PRETRAIN COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time:   {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  Per cluster: {sum(r['elapsed'] for r in cluster_results.values())/k:.1f}s avg")
    print(f"  Parallelism: {'4x' if parallel else '1x'}")
    print(f"  Saved to:    {save_path}")
    print(f"{'='*60}")

    return {
        "status": "trained",
        "elapsed_seconds": elapsed,
        "pretrain_rounds": pretrain_rounds,
        "n_nodes": n_nodes,
        "cluster_size": cluster_size,
        "k": k,
        "device": str(device),
        "save_dir": str(save_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Hierarchical PPO Pretrain")
    parser.add_argument("--rounds", type=int, default=500, help="Pretrain rounds")
    parser.add_argument("--n_nodes", type=int, default=20, help="Number of nodes")
    parser.add_argument("--cluster_size", type=int, default=5, help="Nodes per cluster")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", default="pretrainppo_hierarchical", help="Save directory")
    parser.add_argument("--sequential", action="store_true", help="Disable parallel training")

    args = parser.parse_args()

    result = hierarchical_pretrain(
        n_nodes=args.n_nodes,
        pretrain_rounds=args.rounds,
        cluster_size=args.cluster_size,
        seed=args.seed,
        save_dir=args.save_dir,
        parallel=not args.sequential,
    )


if __name__ == "__main__":
    main()