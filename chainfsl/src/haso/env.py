"""
Gymnasium environment for MA-HASO (Multi-Agent Hierarchical Adaptive Split Optimization).

Models the SPLIT-FEDERATED LEARNING decision-making as an MDP where each edge node
chooses: cut_layer, batch_size, H (local epochs), and target_compute_node.

Reward function (Eq. 7 from ChainFSL paper):
    r_t = -α·T_comp - β·T_comm + γ·φ·ΔF

Where:
- T_comp: Computation time based on cut_layer and node flops_ratio
- T_comm: Communication time based on smashed data size and bandwidth
- φ: Shapley value (EMA-smoothed contribution estimate)
- ΔF: Model accuracy improvement in this step
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from ..emulator.node_profile import HardwareProfile, RESNET18_MEMORY_MAP
from ..emulator.network_emulator import GossipProtocol


class SFLNodeEnv(gym.Env):
    """
    Custom Gymnasium env for a single Data Node in MA-HASO.

    State (7-dim, normalized [0,1]):
        [cpu_util, mem_util, energy_ratio, bandwidth_norm,
         current_loss, loss_std, neighbor_avail]

    Action (MultiDiscrete):
        [cut_layer_idx, batch_size_idx, H_idx, target_node_idx]

    Cut layer choices: 1, 2, 3, 4 (ResNet-18 residual block boundaries)
    Batch size choices: [8, 16, 32, 64]
    H choices: [1, 2, 3, 5]
    """

    metadata = {"render_modes": []}

    # Canonical split points for ResNet-18
    CUT_LAYERS = [1, 2, 3, 4]
    BATCH_SIZES = [8, 16, 32, 64]
    H_CHOICES = [1, 2, 3, 5]

    # State bounds (low, high)
    STATE_LOW = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    STATE_HIGH = np.array([1.0, 1.0, 1.0, 1.0, 10.0, 5.0, 1.0], dtype=np.float32)

    def __init__(
        self,
        node_profile: HardwareProfile,
        n_compute_nodes: int,
        memory_map: Optional[dict] = None,
        reward_weights: Tuple[float, float, float] = (1.0, 0.5, 0.1),
        max_steps: int = 200,
        seed: Optional[int] = None,
    ):
        """
        Args:
            node_profile: HardwareProfile for this node.
            n_compute_nodes: Number of available compute (server) nodes.
            memory_map: cut_layer -> memory MB mapping. Defaults to RESNET18_MEMORY_MAP.
            reward_weights: (α, β, γ) for Eq. 7.
            max_steps: Maximum steps per episode.
            seed: Random seed.
        """
        super().__init__()
        self.profile = node_profile
        self.n_compute = n_compute_nodes
        self.memory_map = memory_map or RESNET18_MEMORY_MAP
        self.alpha, self.beta, self.gamma = reward_weights
        self.max_steps = max_steps

        # --- Observation space ---
        self.observation_space = spaces.Box(
            low=self.STATE_LOW,
            high=self.STATE_HIGH,
            dtype=np.float32,
        )

        # --- Action space ---
        # [cut_layer_idx, batch_size_idx, H_idx, target_compute_node]
        self.action_space = spaces.MultiDiscrete([
            len(self.CUT_LAYERS),    # cut_layer: 0-3 → layers 1-4
            len(self.BATCH_SIZES),   # batch_size: 0-3 → [8,16,32,64]
            len(self.H_CHOICES),     # H: 0-3 → [1,2,3,5]
            n_compute_nodes,          # target node: 0 to n_compute_nodes-1
        ])

        # --- Internal state ---
        self._rng = np.random.default_rng(seed)
        self._state: Optional[np.ndarray] = None
        self._step_count = 0

        # --- Historical metrics for Shapley estimation ---
        self._shapley_ema = 0.1
        self._loss_ema = 5.0
        self._loss_std = 1.0
        self._neighbor_avail = 0.5

        # --- Gossip integration ---
        self._gossip: Optional[GossipProtocol] = None

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._shapley_ema = 0.1
        self._loss_ema = self._rng.uniform(5.0, 8.0)
        self._loss_std = self._rng.uniform(0.5, 2.0)
        self._neighbor_avail = self._rng.uniform(0.3, 0.8)

        self._state = self._get_obs()
        return self._state, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step.

        Args:
            action: [cut_layer_idx, batch_size_idx, H_idx, target_node_idx]

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        cut_layer_idx = int(action[0])
        batch_size_idx = int(action[1])
        H_idx = int(action[2])
        target_node = int(action[3])

        cut_layer = self.CUT_LAYERS[cut_layer_idx]
        batch_size = self.BATCH_SIZES[batch_size_idx]
        H = self.H_CHOICES[H_idx]

        # Validate action against memory constraint
        cut_layer, batch_size = self._apply_memory_constraint(cut_layer, batch_size)

        # Compute resource cost (T_comp + T_comm)
        T_comp = self._compute_time_comp(cut_layer, batch_size)
        T_comm = self._compute_time_comm(cut_layer, batch_size)

        # Simulate training performance
        performance_gain = self._simulate_performance(cut_layer, batch_size, H, target_node)
        delta_F = max(0.0, self._loss_ema - performance_gain)

        # Reward: Eq. 7
        reward = -self.alpha * T_comp - self.beta * T_comm + self.gamma * self._shapley_ema * delta_F

        # Update internal state
        self._update_state(cut_layer, batch_size, H)
        self._loss_ema = performance_gain
        self._loss_std = max(0.1, self._loss_std * 0.98)

        self._step_count += 1
        terminated = self._check_termination()
        truncated = self._step_count >= self.max_steps

        info = {
            "cut_layer": cut_layer,
            "batch_size": batch_size,
            "H": H,
            "target_node": target_node,
            "T_comp": T_comp,
            "T_comm": T_comm,
            "delta_F": delta_F,
            "shapley_ema": self._shapley_ema,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def set_gossip(self, gossip: GossipProtocol) -> None:
        """Inject gossip protocol for neighbor availability."""
        self._gossip = gossip

    def update_shapley(self, phi: float) -> None:
        """
        GTM calls this to inject the computed Shapley value for reward shaping.

        Args:
            phi: Computed Shapley value from TMCS
        """
        beta = 0.9
        self._shapley_ema = beta * self._shapley_ema + (1 - beta) * phi

    def update_loss(self, loss: float, loss_std: float) -> None:
        """Called by protocol to update observed loss metrics."""
        self._loss_ema = min(loss, 10.0)
        self._loss_std = min(loss_std, 5.0)

    # --------------------------------------------------------------------- #
    # Observation
    # --------------------------------------------------------------------- #

    def _get_obs(self) -> np.ndarray:
        """Build normalized observation vector."""
        cpu_util = self.profile.flops_ratio  # proxy: tier capacity
        mem_util = 1.0 - (self.profile.ram_mb / 8192.0)
        energy_ratio = self.profile.energy_remaining / max(self.profile.energy_budget, 1.0)
        bw_norm = self.profile.bandwidth_mbps / 100.0
        loss_norm = self._loss_ema / 10.0
        loss_std_norm = self._loss_std / 5.0

        # Neighbor availability from gossip
        if self._gossip is not None:
            self._neighbor_avail = self._gossip.mean_neighbor_availability(self.profile.node_id)

        return np.array([
            float(cpu_util),
            float(mem_util),
            float(energy_ratio),
            float(bw_norm),
            float(loss_norm),
            float(loss_std_norm),
            float(self._neighbor_avail),
        ], dtype=np.float32)

    # --------------------------------------------------------------------- #
    # Resource computation
    # --------------------------------------------------------------------- #

    def _apply_memory_constraint(self, cut_layer: int, batch_size: int) -> Tuple[int, int]:
        """Clamp cut_layer and batch_size to fit node memory."""
        if not self.profile.can_fit_cut_layer(cut_layer, self.memory_map):
            # Fall back to shallowest cut that fits
            for cl in sorted(self.memory_map.keys()):
                if self.profile.can_fit_cut_layer(cl, self.memory_map):
                    cut_layer = cl
                    break
            else:
                cut_layer = 1

        # Reduce batch size if still doesn't fit
        for bs in self.BATCH_SIZES:
            if bs <= batch_size:
                required = self._estimate_activation_mb(cut_layer, bs)
                if required <= self.profile.ram_mb * 0.5:  # keep within 50% of RAM
                    batch_size = bs
                    break
        return cut_layer, batch_size

    def _compute_time_comp(self, cut_layer: int, batch_size: int) -> float:
        """Compute time in seconds for local computation."""
        # base_flops scales with cut_layer and batch
        base_flops = 1e8 * (cut_layer / 4.0) * (batch_size / 32.0)
        return self.profile.compute_time(base_flops)

    def _compute_time_comm(self, cut_layer: int, batch_size: int) -> float:
        """Compute communication time in seconds."""
        activation_bytes = self._estimate_activation_bytes(cut_layer, batch_size)
        return self.profile.comm_time(activation_bytes)

    def _estimate_activation_bytes(self, cut_layer: int, batch_size: int) -> int:
        """Estimate smashed data size in bytes."""
        from ..sfl.models import SplittableResNet18
        return SplittableResNet18.smashed_data_size(cut_layer, batch_size)

    def _estimate_activation_mb(self, cut_layer: int, batch_size: int) -> float:
        return self._estimate_activation_bytes(cut_layer, batch_size) / (1024 ** 2)

    # --------------------------------------------------------------------- #
    # Training simulation
    # --------------------------------------------------------------------- #

    def _simulate_performance(
        self, cut_layer: int, batch_size: int, H: int, target_node: int
    ) -> float:
        """
        Simulate local training performance.

        Returns:
            Simulated loss after H local epochs.
        """
        # Cut layer effect: deeper cut → more local computation → better feature extraction
        cut_factor = 0.7 + 0.3 * (cut_layer / 4.0)

        # Batch size effect: larger batch → better gradient estimates → stable convergence
        batch_factor = min(1.0, batch_size / 32.0)

        # H effect: more local epochs → better local model but risk overfitting
        H_factor = np.log(H + 1) / np.log(6.0)
        H_factor = min(1.0, H_factor)

        # Loss improvement per step (simplified convergence)
        improvement = (0.3 + 0.2 * self._neighbor_avail) * cut_factor * batch_factor * H_factor

        # Add noise
        noise = self._rng.normal(0, 0.05)
        simulated_loss = max(0.1, self._loss_ema - improvement + noise)
        return simulated_loss

    def _update_state(self, cut_layer: int, batch_size: int, H: int) -> None:
        """Update node resource state after a step."""
        comp_load = (cut_layer / 4.0) * (batch_size / 32.0)

        # Simulate CPU/RAM usage reduction
        # (real implementation would track actual resource consumption)
        energy_consumed = comp_load * 0.5 + (batch_size / 64.0) * 0.2
        if self.profile.energy_remaining is not None:
            self.profile.energy_remaining = max(
                0.0, self.profile.energy_remaining - energy_consumed
            )

        # Simulate network fluctuation
        if self._gossip is not None:
            self._neighbor_avail = self._gossip.mean_neighbor_availability(self.profile.node_id)

    def _check_termination(self) -> bool:
        """Episode terminates on resource exhaustion or convergence."""
        if self._loss_ema < 0.5:
            return True
        if self.profile.energy_remaining is not None and self.profile.energy_remaining < 5.0:
            return True
        if self.profile.flops_ratio > 0 and self._neighbor_avail < 0.1:
            return True
        return False

    # --------------------------------------------------------------------- #
    # Info utilities
    # --------------------------------------------------------------------- #

    def get_valid_actions(self) -> np.ndarray:
        """
        Return a mask of valid actions given current node resources.

        Returns:
            Boolean mask of shape (4,) indicating which actions are valid.
        """
        mask = np.array([True, True, True, True], dtype=bool)

        # Cut layer must fit in memory
        for i, cl in enumerate(self.CUT_LAYERS):
            if not self.profile.can_fit_cut_layer(cl, self.memory_map):
                mask[0] = False
                break

        return mask

    def action_to_dict(self, action: np.ndarray) -> dict:
        """Convert action index to meaningful labels."""
        return {
            "cut_layer": self.CUT_LAYERS[int(action[0])],
            "batch_size": self.BATCH_SIZES[int(action[1])],
            "H": self.H_CHOICES[int(action[2])],
            "target_node": int(action[3]),
        }


class MultiAgentSFLEnv(gym.Env):
    """
    Multi-agent wrapper over SFLNodeEnv.

    Allows a coordinator to step all agents simultaneously and collect
    joint observations, rewards, and termination flags.

    Useful for centralized training with decentralized execution (CTDE).
    """

    def __init__(
        self,
        node_profiles: list[HardwareProfile],
        n_compute_nodes: int,
        memory_map: Optional[dict] = None,
        reward_weights: Tuple[float, float, float] = (1.0, 0.5, 0.1),
        max_steps: int = 200,
        seed: Optional[int] = None,
    ):
        """
        Args:
            node_profiles: List of HardwareProfile, one per agent.
            n_compute_nodes: Number of server nodes.
            memory_map: Memory requirements per cut layer.
            reward_weights: (α, β, γ) passed to each SFLNodeEnv.
            max_steps: Max steps per episode.
            seed: Random seed.
        """
        super().__init__()
        self.n_agents = len(node_profiles)
        self._envs = [
            SFLNodeEnv(
                node_profile=profile,
                n_compute_nodes=n_compute_nodes,
                memory_map=memory_map,
                reward_weights=reward_weights,
                max_steps=max_steps,
                seed=None,
            )
            for profile in node_profiles
        ]
        self._rng = np.random.default_rng(seed)
        self._step_count = 0

        # Dict spaces for multi-agent
        self.observation_space = spaces.Dict({
            f"agent_{i}": env.observation_space
            for i, env in enumerate(self._envs)
        })
        self.action_space = spaces.Dict({
            f"agent_{i}": env.action_space
            for i, env in enumerate(self._envs)
        })

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[dict, dict]:
        """Reset all agents."""
        obss = {}
        infos = {}
        for i, env in enumerate(self._envs):
            o, info = env.reset(seed=seed, options=options)
            obss[f"agent_{i}"] = o
            infos[f"agent_{i}"] = info
        self._step_count = 0
        return obss, infos

    def step(self, actions: dict) -> Tuple[dict, dict, dict, dict, dict]:
        """Step all agents simultaneously."""
        obss = {}
        rewards = {}
        terminateds = {}
        truncateds = {}
        infos = {}

        for i, env in enumerate(self._envs):
            key = f"agent_{i}"
            o, r, term, trunc, info = env.step(actions[key])
            obss[key] = o
            rewards[key] = r
            terminateds[key] = term
            truncateds[key] = trunc
            infos[key] = info

        self._step_count += 1
        return obss, rewards, terminateds, truncateds, infos

    def update_shapley_all(self, shapley_dict: dict) -> None:
        """Update Shapley values for all agents."""
        for i, phi in shapley_dict.items():
            self._envs[i].update_shapley(phi)

    @property
    def agents(self) -> list[SFLNodeEnv]:
        return self._envs