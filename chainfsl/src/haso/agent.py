"""
PPO-based agent wrapper for MA-HASO.

Each data node runs its own HaSOAgent with a Stable-Baselines3 PPO policy.
Agents learn from their local SFLNodeEnv and receive Shapley values from GTM.
"""

import numpy as np
from typing import Optional, Dict, Any

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from .env import SFLNodeEnv


class HaSOAgent:
    """
    Wrapper for PPO agent of a single Data Node.

    Each node trains its own policy independently (decentralized training,
    centralized inference for evaluation).
    """

    def __init__(
        self,
        env: SFLNodeEnv,
        node_id: int,
        learning_rate: float = 3e-4,
        n_steps: int = 512,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        verbose: int = 0,
    ):
        """
        Args:
            env: SFLNodeEnv instance for this node.
            node_id: Node identifier.
            learning_rate: PPO learning rate.
            n_steps: PPO n_steps.
            batch_size: PPO batch size.
            n_epochs: PPO epochs per update.
            gamma: Discount factor.
            verbose: PPO verbosity level.
        """
        self.env = env
        self.node_id = node_id

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
            verbose=verbose,
            # Use a separate seed per agent
            seed=node_id,
        )

    def decide(self, obs: np.ndarray, deterministic: bool = False) -> Dict[str, Any]:
        """
        Choose action given observation.

        Args:
            obs: Current observation.
            deterministic: If True, use mean policy (no sampling).

        Returns:
            Decoded action dict with keys: cut_layer, batch_size, H, target_compute_node.
        """
        action, _ = self.model.predict(obs, deterministic=deterministic)
        decoded = self.env.action_to_dict(action)
        decoded["node_id"] = self.node_id
        return decoded

    def learn(self, total_timesteps: int) -> None:
        """Update policy."""
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


class HaSOAgentPool:
    """
    Manages a pool of HaSOAgents for all nodes in the federation.

    Provides batch operations for coordinated training and evaluation.
    """

    def __init__(
        self,
        envs: list[SFLNodeEnv],
        learning_rate: float = 3e-4,
        n_steps: int = 512,
        batch_size: int = 64,
        n_epochs: int = 10,
        verbose: int = 0,
    ):
        """
        Args:
            envs: List of SFLNodeEnv instances.
            learning_rate: Shared PPO learning rate.
            n_steps: PPO n_steps.
            batch_size: PPO batch size.
            n_epochs: PPO epochs per update.
            verbose: PPO verbosity level.
        """
        self.agents: list[HaSOAgent] = [
            HaSOAgent(
                env=env,
                node_id=i,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                verbose=verbose,
            )
            for i, env in enumerate(envs)
        ]

    def decide_all(self, obs_list: list[np.ndarray], deterministic: bool = False) -> list[Dict[str, Any]]:
        """Collect decisions from all agents."""
        return [agent.decide(obs, deterministic=deterministic) for agent, obs in zip(self.agents, obs_list)]

    def learn_all(self, total_timesteps: int) -> None:
        """Update all agent policies."""
        for agent in self.agents:
            agent.learn(total_timesteps)

    def update_shapley_all(self, shapley_dict: Dict[int, float]) -> None:
        """Update Shapley values for all agents."""
        for node_id, phi in shapley_dict.items():
            if 0 <= node_id < len(self.agents):
                self.agents[node_id].update_shapley(phi)

    def save_all(self, directory: str) -> None:
        """Save all agents to disk."""
        import os
        os.makedirs(directory, exist_ok=True)
        for i, agent in enumerate(self.agents):
            agent.save(os.path.join(directory, f"agent_{i}.zip"))

    def load_all(self, directory: str) -> None:
        """Load all agents from disk."""
        import os
        for i, agent in enumerate(self.agents):
            path = os.path.join(directory, f"agent_{i}.zip")
            if os.path.exists(path):
                agent.load(path)

    def __len__(self) -> int:
        return len(self.agents)