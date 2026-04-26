"""
MA-HASO Orchestrator - Centralized DRL orchestrator for split optimization.

According to ChainFSL_Implementation_Plan.md Section 1.1:
- One Orchestrator process runs PPO to decide (cut_layer, batch_size, H) for all nodes
- Data Nodes just do SFL training (no PPO overhead)

This separates concerns:
- Orchestrator: DRL decision making, PPO training (1 process)
- Data Nodes: SFL training only, receive configs from orchestrator (N processes)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time

from stable_baselines3 import PPO

from .env import SFLNodeEnv


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    n_nodes: int
    n_actions_per_node: int = 4  # cut_layer, batch_size, H, target
    # Routing: each node gets its decision from orchestrator's action output
    # For simplicity: orchestrator outputs actions for ALL nodes in one vector


class HASOOrchestrator:
    """
    Centralized HASO orchestrator that runs PPO for the entire federation.

    Instead of running PPO on every data node (expensive!), we run ONE PPO
    agent that decides configurations for all nodes. This is according to
    the architecture in ChainFSL_Implementation_Plan.md Section 1.1 where:
    - Coordinator Process runs PPO + Shapley + VRF
    - Worker Processes only do training
    """

    def __init__(
        self,
        n_nodes: int,
        node_profiles: List[Any],
        reward_weights: tuple = (1.0, 0.5, 0.1),
        learning_rate: float = 3e-4,
        n_steps: int = 512,
        batch_size: int = 64,
        n_epochs: int = 10,
        ppo_device: str = "auto",
        verbose: int = 0,
    ):
        """
        Args:
            n_nodes: Number of data nodes in the federation.
            node_profiles: List of HardwareProfile for each node.
            reward_weights: (alpha, beta, gamma) for reward function.
        """
        self.n_nodes = n_nodes
        self.node_profiles = node_profiles
        self.alpha, self.beta, self.gamma = reward_weights

        # Create ONE environment for the orchestrator
        # This env models the "global" decision space
        self.env = OrchestratorEnv(
            n_nodes=n_nodes,
            node_profiles=node_profiles,
            reward_weights=reward_weights,
        )

        # Single PPO model for centralized decision making
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            device=ppo_device,
            verbose=verbose,
            seed=42,
        )

        # Per-node Shapley tracking for reward shaping
        self._shapley_ema: Dict[int, float] = {i: 0.1 for i in range(n_nodes)}
        self._last_rewards: Dict[int, float] = {i: 0.0 for i in range(n_nodes)}

        # Timing stats
        self._decision_times: List[float] = []

    def decide(self, obs: np.ndarray, deterministic: bool = False) -> List[Dict[str, Any]]:
        """
        Decide configurations for all nodes given global observation.

        Args:
            obs: Global observation (not used - orchestrator tracks state internally)
            deterministic: If True, use mean policy.

        Returns:
            List of config dicts, one per node.
        """
        start = time.perf_counter()

        # Get action from PPO
        action, _ = self.model.predict(obs, deterministic=deterministic)

        # Decode action vector into per-node configs
        configs = self._decode_actions(action)

        self._decision_times.append(time.perf_counter() - start)
        return configs

    def _decode_actions(self, action: np.ndarray) -> List[Dict[str, Any]]:
        """
        Decode flat action vector into per-node configurations.

        Action vector layout: [cut_layer_0, batch_size_0, H_0, target_0, cut_layer_1, ...]
        For n_nodes=20: 80 actions total
        """
        configs = []
        idx = 0

        for node_id in range(self.n_nodes):
            cut_layer_idx = int(action[idx])
            batch_size_idx = int(action[idx + 1])
            H_idx = int(action[idx + 2])
            target = int(action[idx + 3])
            idx += 4

            # Map indices to actual values
            cut_layer = SFLNodeEnv.CUT_LAYERS[min(cut_layer_idx, len(SFLNodeEnv.CUT_LAYERS) - 1)]
            batch_size = SFLNodeEnv.BATCH_SIZES[min(batch_size_idx, len(SFLNodeEnv.BATCH_SIZES) - 1)]
            H = SFLNodeEnv.H_CHOICES[min(H_idx, len(SFLNodeEnv.H_CHOICES) - 1)]

            # Enforce memory constraint
            profile = self.node_profiles[node_id]
            memory_map = self.env.memory_map

            # Find deepest valid cut layer
            valid_cut = None
            for cl in sorted(memory_map.keys(), reverse=True):
                if profile.can_fit_cut_layer(cl, memory_map):
                    valid_cut = cl
                    break

            if valid_cut is not None and cut_layer > valid_cut:
                cut_layer = valid_cut

            configs.append({
                "node_id": node_id,
                "cut_layer": cut_layer,
                "batch_size": batch_size,
                "H": H,
                "target_compute_node": target,
            })

        return configs

    def learn(self, total_timesteps: int) -> None:
        """Update PPO policy."""
        self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)

    def update_shapley(self, shapley_dict: Dict[int, float]) -> None:
        """Update Shapley values for reward shaping."""
        beta = 0.9
        for node_id, phi in shapley_dict.items():
            if node_id in self._shapley_ema:
                self._shapley_ema[node_id] = beta * self._shapley_ema[node_id] + (1 - beta) * phi

    def get_mean_shapley(self) -> float:
        """Get mean Shapley value across all nodes."""
        if not self._shapley_ema:
            return 0.1
        return np.mean(list(self._shapley_ema.values()))

    def save(self, path: str) -> None:
        """Save orchestrator PPO model."""
        self.model.save(path)

    def load(self, path: str) -> None:
        """Load orchestrator PPO model."""
        self.model = PPO.load(path, env=self.env)

    def get_decision_time_stats(self) -> Dict[str, float]:
        """Get decision timing statistics."""
        if not self._decision_times:
            return {"mean": 0.0, "max": 0.0, "min": 0.0}
        return {
            "mean": np.mean(self._decision_times),
            "max": np.max(self._decision_times),
            "min": np.min(self._decision_times),
            "total": np.sum(self._decision_times),
        }


class OrchestratorEnv(gym.Env):
    """
    Gymnasium environment for the centralized orchestrator.

    The orchestrator observes aggregate state and decides configurations
    for ALL nodes in a single action vector.

    State: [mean_cpu, mean_mem, mean_bw, mean_loss, mean_shapley, fairness, n_valid]
    Action: MultiDiscrete with n_nodes * 4 dimensions
    """

    # Same action choices as SFLNodeEnv
    CUT_LAYERS = [1, 2, 3, 4]
    BATCH_SIZES = [8, 16, 32, 64]
    H_CHOICES = [1, 2, 3, 5]

    STATE_LOW = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    STATE_HIGH = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_nodes: int,
        node_profiles: List[Any],
        reward_weights: tuple = (1.0, 0.5, 0.1),
        max_steps: int = 200,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_nodes: Number of data nodes to configure.
            node_profiles: List of HardwareProfile for each node.
            reward_weights: (alpha, beta, gamma) for reward.
            max_steps: Maximum steps per episode.
            seed: Random seed.
        """
        super().__init__()
        self.n_nodes = n_nodes
        self.profiles = node_profiles
        self.alpha, self.beta, self.gamma = reward_weights
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)

        # Memory map for constraint checking
        from ..sfl.models import SplittableResNet18
        self.memory_map = SplittableResNet18.MEMORY_WITH_ADAM_MB

        # Observation space: global aggregate state
        self.observation_space = spaces.Box(
            low=self.STATE_LOW,
            high=self.STATE_HIGH,
            dtype=np.float32,
        )

        # Action space: n_nodes * [cut_layer, batch_size, H, target]
        # Each node gets 4 discrete actions
        self.action_space = spaces.MultiDiscrete([
            len(self.CUT_LAYERS),    # cut_layer
            len(self.BATCH_SIZES),   # batch_size
            len(self.H_CHOICES),     # H
            max(1, n_nodes),         # target node (1 to n_nodes-1)
        ] * n_nodes)

        # Internal state
        self._step_count = 0
        # Historical metrics - MUST be initialized BEFORE _get_obs() call
        self._shapley_ema = 0.1
        self._mean_loss = 5.0
        self._fairness = 0.5

        # Compute initial observation AFTER historical metrics are set
        self._state = self._get_obs()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple:
        """Reset environment."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._shapley_ema = 0.1
        self._mean_loss = self._rng.uniform(5.0, 8.0)
        self._fairness = self._rng.uniform(0.3, 0.6)

        self._state = self._get_obs()
        return self._state, {}

    def step(self, action: np.ndarray) -> tuple:
        """
        Execute one step.

        Args:
            action: Flat action vector of size n_nodes * 4

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        # Decode actions per node
        configs = self._decode_actions(action)

        # Compute global reward based on all node configs
        reward = self._compute_global_reward(configs)

        # Update state
        self._update_state(configs)
        self._step_count += 1

        terminated = self._check_termination()
        truncated = self._step_count >= self.max_steps

        obs = self._get_obs()
        info = {"configs": configs, "reward": reward}

        return obs, reward, terminated, truncated, info

    def _decode_actions(self, action: np.ndarray) -> List[Dict[str, Any]]:
        """Decode flat action vector into per-node configs."""
        configs = []
        idx = 0

        for _ in range(self.n_nodes):
            cut_layer_idx = int(action[idx])
            batch_size_idx = int(action[idx + 1])
            H_idx = int(action[idx + 2])
            target = int(action[idx + 3])
            idx += 4

            cut_layer = self.CUT_LAYERS[cut_layer_idx]
            batch_size = self.BATCH_SIZES[batch_size_idx]
            H = self.H_CHOICES[H_idx]

            configs.append({
                "cut_layer": cut_layer,
                "batch_size": batch_size,
                "H": H,
                "target": target,
            })

        return configs

    def _compute_global_reward(self, configs: List[Dict[str, Any]]) -> float:
        """
        Compute global reward based on all node configurations.

        Reward = -alpha * mean_T_comp - beta * mean_T_comm + gamma * shapley * delta_F
        """
        total_T_comp = 0.0
        total_T_comm = 0.0

        for cfg in configs:
            cut_layer = cfg["cut_layer"]
            batch_size = cfg["batch_size"]

            # Estimate computation time
            base_flops = 1e8 * (cut_layer / 4.0) * (batch_size / 32.0)
            # Use mean flops_ratio across all profiles as proxy
            mean_flops = np.mean([p.flops_ratio for p in self.profiles])
            T_comp = base_flops / (mean_flops * 1e9) if mean_flops > 0 else 1.0

            # Estimate communication time
            smashed_bytes = self._estimate_smashed_bytes(cut_layer, batch_size)
            mean_bw = np.mean([p.bandwidth_mbps for p in self.profiles])
            T_comm = smashed_bytes / (mean_bw * 1e6 / 8) if mean_bw > 0 else 0.1

            total_T_comp += T_comp
            total_T_comm += T_comm

        mean_T_comp = total_T_comp / max(len(configs), 1)
        mean_T_comm = total_T_comm / max(len(configs), 1)

        # Reward formula
        delta_F = max(0.0, self._mean_loss - (self._mean_loss * 0.95))
        reward = -self.alpha * mean_T_comp - self.beta * mean_T_comm + self.gamma * self._shapley_ema * delta_F

        return float(reward)

    def _estimate_smashed_bytes(self, cut_layer: int, batch_size: int) -> int:
        """Estimate smashed data size in bytes."""
        size_map = {
            1: 64 * 56 * 56 * 4,
            2: 128 * 28 * 28 * 4,
            3: 256 * 14 * 14 * 4,
            4: 512 * 7 * 7 * 4,
        }
        return size_map.get(cut_layer, 512 * 7 * 7 * 4) * batch_size

    def _update_state(self, configs: List[Dict[str, Any]]) -> None:
        """Update internal state after action."""
        # Simulate loss improvement
        improvement = 0.1 + 0.1 * np.random.rand()
        self._mean_loss = max(0.5, self._mean_loss - improvement)

        # Update Shapley EMA
        self._shapley_ema = 0.9 * self._shapley_ema + 0.1 * np.random.rand()

    def _get_obs(self) -> np.ndarray:
        """Build normalized observation vector."""
        mean_flops = np.mean([p.flops_ratio for p in self.profiles])
        mean_ram = np.mean([p.ram_mb for p in self.profiles])
        mean_bw = np.mean([p.bandwidth_mbps for p in self.profiles])

        return np.array([
            float(mean_flops),
            float(1.0 - mean_ram / 8192.0),
            float(mean_bw / 100.0),
            float(self._mean_loss / 10.0),
            float(self._shapley_ema),
            float(self._fairness),
            float(self.n_nodes / 50.0),  # normalize node count
        ], dtype=np.float32)

    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        return self._mean_loss < 0.5

    def update_shapley(self, mean_phi: float) -> None:
        """Update Shapley value (called by protocol)."""
        beta = 0.9
        self._shapley_ema = beta * self._shapley_ema + (1 - beta) * mean_phi


def create_orchestrator(
    n_nodes: int,
    node_profiles: List[Any],
    config: Optional[Dict[str, Any]] = None,
) -> HASOOrchestrator:
    """
    Factory function to create an orchestrator from config.

    Args:
        n_nodes: Number of data nodes.
        node_profiles: List of HardwareProfile.
        config: Optional config dict with HASO parameters.

    Returns:
        HASOOrchestrator instance.
    """
    cfg = config or {}

    return HASOOrchestrator(
        n_nodes=n_nodes,
        node_profiles=node_profiles,
        reward_weights=(
            cfg.get("reward_alpha", 1.0),
            cfg.get("reward_beta", 0.5),
            cfg.get("reward_gamma", 0.1),
        ),
        learning_rate=cfg.get("ppo_learning_rate", 3e-4),
        n_steps=cfg.get("ppo_n_steps", 512),
        batch_size=cfg.get("ppo_batch_size", 64),
        n_epochs=cfg.get("ppo_n_epochs", 10),
        ppo_device=cfg.get("ppo_device", "auto"),
        verbose=cfg.get("verbose", 0),
    )