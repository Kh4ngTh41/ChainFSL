"""
ClusterHASOAgent - PPO agent for cluster-level split optimization.

Each cluster has ONE ClusterHASOAgent (running on cluster-head node).
The agent decides: cut_layer, batch_size, H for ALL nodes in the cluster.

This is hierarchical MA-HASO:
- Cluster-level: ClusterHASOAgent decides config for cluster members
- Inter-cluster: Cluster-heads coordinate via gossip
"""

import numpy as np
from typing import List, Dict, Any, Optional

from stable_baselines3 import PPO

from .env import SFLNodeEnv


class ClusterHASOAgent:
    """
    PPO agent that manages split optimization for a cluster of nodes.

    Operates on cluster-local observation (aggregate of cluster members).
    Decides configuration for all cluster members in ONE action vector.

    Unlike HaSOAgent (per-node), this is ONE PPO for a cluster (a nodes).
    """

    def __init__(
        self,
        env: SFLNodeEnv,
        cluster_id: int,
        cluster_node_ids: List[int],
        learning_rate: float = 3e-4,
        n_steps: int = 512,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        ppo_device: str = "auto",
        verbose: int = 0,
    ):
        """
        Args:
            env: SFLNodeEnv instance for this cluster-head.
            cluster_id: Cluster identifier.
            cluster_node_ids: List of node_ids in this cluster.
            learning_rate: PPO learning rate.
            n_steps: PPO n_steps.
            batch_size: PPO batch size.
            n_epochs: PPO epochs per update.
            gamma: Discount factor.
            verbose: PPO verbosity level.
        """
        self.env = env
        self.cluster_id = cluster_id
        self.cluster_node_ids = cluster_node_ids

        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            device=ppo_device,
            verbose=verbose,
            seed=cluster_id,  # Use cluster_id as seed
        )

    def decide(self, obs: np.ndarray, deterministic: bool = False) -> Dict[str, Any]:
        """
        Choose action given cluster-level observation.

        Args:
            obs: Cluster-level observation (aggregated from member nodes).
            deterministic: If True, use mean policy (no sampling).

        Returns:
            Decoded action dict with keys: cut_layer, batch_size, H, target_compute_node.
            Also includes cluster_id and cluster_member_ids.
        """
        action, _ = self.model.predict(obs, deterministic=deterministic)
        decoded = self.env.action_to_dict(action)
        decoded["cluster_id"] = self.cluster_id
        decoded["cluster_member_ids"] = self.cluster_node_ids
        decoded["head_node_id"] = self.cluster_node_ids[0]  # Head is first node
        return decoded

    def decide_per_node(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Decide configuration for EACH node in the cluster separately.

        Use this when cluster members need INDIVIDUAL configs (not shared).

        Args:
            obs: Cluster-level observation.
            deterministic: If True, use mean policy.

        Returns:
            List of action dicts, one per cluster member.
        """
        action, _ = self.model.predict(obs, deterministic=deterministic)
        base_decoded = self.env.action_to_dict(action)

        # Apply same base config to all members
        # In a more advanced version, could add per-node variation
        results = []
        for node_id in self.cluster_node_ids:
            decoded = base_decoded.copy()
            decoded["node_id"] = node_id
            decoded["cluster_id"] = self.cluster_id
            results.append(decoded)

        return results

    def learn(self, total_timesteps: int) -> None:
        """Update PPO policy for cluster."""
        self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)

    def save(self, path: str) -> None:
        """Save model to disk."""
        self.model.save(path)

    def load(self, path: str) -> None:
        """Load model from disk."""
        self.model = PPO.load(path, env=self.env)

    def update_shapley(self, phi: float) -> None:
        """Propagate Shapley value to environment for reward shaping."""
        self.env.update_shapley(phi)

    def update_cluster_shapley(self, shapley_dict: Dict[int, float]) -> None:
        """
        Update Shapley values for multiple nodes in cluster.

        Args:
            shapley_dict: Dict mapping node_id → Shapley value.
        """
        # Average Shapley for cluster reward signal
        node_phis = [shapley_dict[nid] for nid in self.cluster_node_ids if nid in shapley_dict]
        if node_phis:
            avg_phi = sum(node_phis) / len(node_phis)
            self.env.update_shapley(avg_phi)


class ClusterAgentPool:
    """
    Manages all ClusterHASOAgents in the federation.

    One ClusterHASOAgent per cluster, selected by cluster-head election.
    """

    def __init__(self, cluster_manager, node_profiles: List):
        """
        Args:
            cluster_manager: ClusterManager instance.
            node_profiles: List of HardwareProfile for each node.
        """
        self.cluster_manager = cluster_manager
        self.node_profiles = node_profiles
        self.agents: Dict[int, ClusterHASOAgent] = {}  # cluster_id → agent

    def create_agents(
        self,
        env_builder,  # callable(node_profile, cluster_id) → SFLNodeEnv
        learning_rate: float = 3e-4,
        n_steps: int = 512,
        batch_size: int = 64,
        n_epochs: int = 10,
        ppo_device: str = "auto",
        verbose: int = 0,
    ) -> None:
        """
        Create ClusterHASOAgent for each cluster.

        Args:
            env_builder: Function that creates SFLNodeEnv for a cluster-head.
                         Signature: env_builder(node_profile, cluster_id, n_compute) → SFLNodeEnv
        """
        for cluster_id, node_ids in self.cluster_manager.clusters.items():
            head_id = self.cluster_manager.get_cluster_head(cluster_id)
            head_profile = self.node_profiles[head_id] if head_id < len(self.node_profiles) else None

            if head_profile is None:
                continue

            # Create env for cluster-head
            n_compute = len(self.cluster_manager.clusters)  # Use clusters as compute nodes
            env = env_builder(head_profile, cluster_id, n_compute)

            # Create agent
            agent = ClusterHASOAgent(
                env=env,
                cluster_id=cluster_id,
                cluster_node_ids=node_ids,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                ppo_device=ppo_device,
                verbose=verbose,
            )
            self.agents[cluster_id] = agent

    def decide_cluster(self, cluster_id: int, obs: np.ndarray, deterministic: bool = False) -> Optional[Dict[str, Any]]:
        """Get decision from a specific cluster's agent."""
        agent = self.agents.get(cluster_id)
        if agent is None:
            return None
        return agent.decide(obs, deterministic=deterministic)

    def decide_all_clusters(self, obs: np.ndarray, deterministic: bool = False) -> Dict[int, Dict[str, Any]]:
        """
        Get decisions from ALL clusters.

        Returns:
            Dict mapping cluster_id → decision dict.
        """
        return {
            cid: self.agents[cid].decide(obs, deterministic=deterministic)
            for cid in self.agents
        }

    def learn_all(self, total_timesteps: int) -> None:
        """Update PPO for all cluster agents."""
        for agent in self.agents.values():
            agent.learn(total_timesteps=total_timesteps)

    def get_agent(self, cluster_id: int) -> Optional[ClusterHASOAgent]:
        """Get agent for a specific cluster."""
        return self.agents.get(cluster_id)