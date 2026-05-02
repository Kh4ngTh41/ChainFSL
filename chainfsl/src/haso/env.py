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

Enhanced with:
- ComputeNode queue management
- RoutingPolicy for target selection
- RewardFunction with fusion bonus and overlap penalty
- LogManager for full decision logging
- ScenarioB/C support for sparse/dynamic topologies
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List

from ..emulator.node_profile import HardwareProfile, RESNET18_MEMORY_MAP
from ..emulator.network_emulator import GossipProtocol
from ..sfl.models import SplittableResNet18

from .compute_node import ComputeNode, TIER_CONFIGS, create_compute_node
from .state_action import StateBuilder, ActionBuilder, STATE_NAMES
from .reward_function import RewardFunction, RewardConfig
from .routing import RoutingPolicy, RoutingDecision, layer_compatibility
from .logging_framework import LogManager, RoundLog, DecisionLog
from .scenarios import ScenarioB, ScenarioC


class SFLNodeEnv(gym.Env):
    """
    Custom Gymnasium env for a single Data Node in MA-HASO.

    State (11-dim, normalized [0,1]):
        [cpu_util, ram_util, gpu_util, bandwidth,
         current_loss, loss_std, neighbor_avail,
         compute_queue, fusion_candidates, energy_ratio, shard_available]

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

    # State bounds (low, high) - 16 dimensions (11 base + 4 tier + 1 compute ratio)
    STATE_LOW = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    STATE_HIGH = np.array([1.0, 1.0, 1.0, 1.0, 10.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 5.0], dtype=np.float32)

    def __init__(
        self,
        node_profile: HardwareProfile,
        n_compute_nodes: int,
        memory_map: Optional[dict] = None,
        reward_weights: Tuple[float, float, float] = (2.0, 1.5, 0.5),
        max_steps: int = 200,
        seed: Optional[int] = None,
        enable_logging: bool = True,
        log_dir: Optional[str] = None,
        compute_nodes: Optional[List[ComputeNode]] = None,
    ):
        """
        Args:
            node_profile: HardwareProfile for this node.
            n_compute_nodes: Number of available compute (server) nodes.
            memory_map: cut_layer -> memory MB mapping. Defaults to RESNET18_MEMORY_MAP.
            reward_weights: (α, β, γ) for Eq. 7 reward computation.
            max_steps: Maximum steps per episode.
            seed: Random seed.
            enable_logging: Enable decision logging.
            log_dir: Directory for log output.
            compute_nodes: List of ComputeNode instances for routing.
        """
        super().__init__()
        self.profile = node_profile
        self.n_compute = n_compute_nodes
        self.memory_map = memory_map or RESNET18_MEMORY_MAP
        self.alpha, self.beta, self.gamma = reward_weights
        self.max_steps = max_steps
        self.enable_logging = enable_logging
        self.log_dir = log_dir

        # --- Compute nodes and routing ---
        self.compute_nodes = compute_nodes or []
        self.routing_policy: Optional[RoutingPolicy] = None
        if self.compute_nodes:
            self.routing_policy = RoutingPolicy(
                compute_nodes=self.compute_nodes,
                n_data_nodes=n_compute_nodes
            )

        # --- State and action builders ---
        self.state_builder = StateBuilder()
        self.action_builder = ActionBuilder()

        # --- Reward function ---
        reward_config = RewardConfig(
            alpha=reward_weights[0],
            beta=reward_weights[1],
            gamma=reward_weights[2],
            lambda_fusion=0.3,
            mu_overlap=0.4,
            min_acc_threshold=0.60
        )
        self.reward_function = RewardFunction(reward_config)

        # --- Logging ---
        self.log_manager: Optional[LogManager] = None
        if enable_logging and log_dir:
            self.log_manager = LogManager(log_dir, "haso_experiment")
            self.log_manager.setup_run({"n_compute_nodes": n_compute_nodes})

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
        self._round_count = 0

        # --- Historical metrics for Shapley estimation ---
        self._shapley_ema = 0.1
        self._loss_ema = 5.0
        self._loss_std = 1.0
        self._neighbor_avail = 0.5

        # --- Fusion and routing state ---
        self._compute_queue = 0.0
        self._fusion_candidates = 0.0
        self._shard_available = 1.0
        self._routing_decisions: List[RoutingDecision] = []

        # --- Scenario support ---
        self._scenario: Optional[Any] = None
        self._scenario_type: Optional[str] = None

        # --- Gossip integration ---
        self._gossip: Optional[GossipProtocol] = None

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def set_compute_nodes(self, compute_nodes: List[ComputeNode]) -> None:
        """Inject compute nodes for routing decisions."""
        self.compute_nodes = compute_nodes
        if compute_nodes:
            self.routing_policy = RoutingPolicy(
                compute_nodes=self.compute_nodes,
                n_data_nodes=self.n_compute
            )

    def set_log_manager(self, log_manager: LogManager) -> None:
        """Inject log manager for decision logging."""
        self.log_manager = log_manager

    def set_scenario_b(self, N: int, k: int) -> ScenarioB:
        """Set Scenario B (fixed sparse topology: k data nodes, N-k relay only)."""
        self._scenario = ScenarioB(N=N, k=k)
        self._scenario_type = "B"
        return self._scenario

    def set_scenario_c(
        self, N: int, k_init: int, p_dropout_base: float = 0.1,
        p_join: float = 0.02, k_min: int = 5, k_max: Optional[int] = None
    ) -> ScenarioC:
        """Set Scenario C (dynamic dropout: k(t) changes over rounds)."""
        self._scenario = ScenarioC(
            N=N, k_init=k_init, p_dropout_base=p_dropout_base,
            p_join=p_join, k_min=k_min, k_max=k_max
        )
        self._scenario_type = "C"
        return self._scenario

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._round_count += 1
        self._shapley_ema = 0.1
        self._loss_ema = self._rng.uniform(5.0, 8.0)
        self._loss_std = self._rng.uniform(0.5, 2.0)
        self._neighbor_avail = self._rng.uniform(0.3, 0.8)

        # Reset fusion and compute state
        self._compute_queue = 0.0
        self._fusion_candidates = 0.0
        self._shard_available = 1.0
        self._routing_decisions = []

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

        # Validate action against memory constraint (includes optimizer state)
        memory_map = SplittableResNet18.MEMORY_WITH_ADAM_MB
        valid_cut = self._find_deepest_valid_cut_layer(memory_map)

        if valid_cut is None:
            # Node cannot train at any cut layer — severe penalty, terminate step
            terminated = True
            reward = -100.0  # Large penalty for OOM condition
            info = {
                "cut_layer": 0, "batch_size": 0, "H": 0,
                "target_node": target_node, "T_comp": 0, "T_comm": 0,
                "delta_F": 0, "shapley_ema": self._shapley_ema,
                "fusion_bonus": 0, "overlap_penalty": 0,
                "compute_queue": self._compute_queue,
                "routing_mode": "parallel",
                "fusion_partners": [],
                "oom": True, "error": "Node cannot fit any cut layer",
            }
            return self._get_obs(), float(reward), terminated, False, info

        # Clamp to deepest valid cut
        cut_layer, batch_size = self._apply_memory_constraint(cut_layer, batch_size)

        # Get compute node for this target and update queue
        if target_node < len(self.compute_nodes):
            cn = self.compute_nodes[target_node]
            self._compute_queue = cn.queue_utilization
            cn.add_task(cut_layer)
        else:
            self._compute_queue = 0.0

        # Compute routing decision using RoutingPolicy
        routing_mode = "parallel"
        fusion_partners = []
        if self.routing_policy is not None:
            compute_node_loads = {
                cn.node_id: cn.current_queue for cn in self.compute_nodes
            }
            # Use routing policy to get better target if needed
            best_target = self.routing_policy.select_target_node(
                self.profile, cut_layer, compute_node_loads
            )
            # Only override if target_node is overloaded
            if self.compute_nodes and best_target != target_node:
                # Check if current target has acceptable load
                if self.compute_nodes[target_node].current_queue > 5:
                    target_node = best_target

            # Detect fusion opportunities
            temp_decision = RoutingDecision(
                node_id=getattr(self.profile, 'node_id', 0),
                cut_layer=cut_layer,
                batch_size=batch_size,
                H=H,
                target_compute_node=target_node,
                fusion_partners=[],
                routing_mode='parallel'
            )
            self._routing_decisions.append(temp_decision)
            fusion_partners = self.routing_policy.detect_fusion_opportunities(
                self._routing_decisions, temp_decision.node_id
            )
            if len(fusion_partners) > 0:
                routing_mode = "fusion"

        # Compute resource cost (T_comp + T_comm)
        T_comp = self._compute_time_comp(cut_layer, batch_size)
        T_comm = self._compute_time_comm(cut_layer, batch_size)

        # Compute fusion bonus and overlap penalty
        if routing_mode == "fusion" and fusion_partners:
            layer_compat = layer_compatibility(
                cut_layer, cut_layer  # self-compatibility for now
            )
            fusion_bonus = self.reward_function.compute_fusion_bonus(
                [self.profile.node_id] + fusion_partners, layer_compat
            )
        else:
            fusion_bonus = 0.0

        overlap_penalty = 0.0
        if self.routing_policy is not None and self._routing_decisions:
            conflicts = self.routing_policy.compute_overlap_conflicts(
                self._routing_decisions
            )
            for cn_id, conflict_score in conflicts:
                if cn_id == target_node:
                    overlap_penalty = self.reward_function.compute_overlap_penalty(
                        conflict_score
                    )
                    break

        # Simulate training performance
        performance_gain = self._simulate_performance(cut_layer, batch_size, H, target_node)
        delta_F = max(0.0, self._loss_ema - performance_gain)

        # Compute reward with new RewardFunction
        # CRITICAL FIX: Pass tier-aware H penalty and node capability
        node_tier = getattr(self.profile, 'tier', 2)
        reward = self.reward_function.compute(
            T_comp=T_comp,
            T_comm=T_comm,
            delta_F=delta_F,
            shapley_phi=self._shapley_ema,
            fusion_bonus=fusion_bonus,
            overlap_penalty=overlap_penalty,
            H=H,  # Pass local epochs for penalty computation
            current_accuracy=1.0 - performance_gain,  # approx accuracy
            node_tier=node_tier,  # Tier-aware H penalty
        )

        # Update internal state
        self._update_state(cut_layer, batch_size, H)
        self._loss_ema = performance_gain
        self._loss_std = max(0.1, self._loss_std * 0.98)
        self._fusion_candidates = len(fusion_partners) / max(len(self.compute_nodes), 1)

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
            "fusion_bonus": fusion_bonus,
            "overlap_penalty": overlap_penalty,
            "compute_queue": self._compute_queue,
            "routing_mode": routing_mode,
            "fusion_partners": fusion_partners,
        }

        # Log decision
        if self.log_manager is not None:
            try:
                decision_log = DecisionLog(
                    node_id=getattr(self.profile, 'node_id', 0),
                    round=self._round_count,
                    step=self._step_count,
                    state=self._state.tolist() if self._state is not None else [],
                    action=info.copy(),
                    reward=reward,
                    done=terminated or truncated,
                    info=info
                )
                self.log_manager.log_decision(
                    getattr(self.profile, 'node_id', 0), decision_log
                )
            except Exception:
                pass  # Logging should not crash the env

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
        """Build normalized 16-dim observation vector (11 base + 4 tier + 1 compute ratio)."""
        # Build profile dict for StateBuilder
        profile_dict = {
            'cpu_util': self.profile.flops_ratio,
            'ram_util': 1.0 - (self.profile.ram_mb / 8192.0),
            'gpu_util': getattr(self.profile, 'gpu_util', 0.5),
            'bandwidth': self.profile.bandwidth_mbps / 100.0,
            'flops_ratio': self.profile.flops_ratio,
        }

        # Neighbor availability from gossip
        if self._gossip is not None:
            self._neighbor_avail = self._gossip.mean_neighbor_availability(self.profile.node_id)

        # Energy ratio
        energy_ratio = 1.0
        if getattr(self.profile, 'energy_remaining', None) is not None:
            budget = getattr(self.profile, 'energy_budget', 1000.0)
            energy_ratio = self.profile.energy_remaining / max(budget, 1.0)

        # Get tier for tier-aware state encoding
        node_tier = getattr(self.profile, 'tier', 2)

        # Use StateBuilder to create normalized 16-dim state (with tier info)
        state = self.state_builder.build_state(
            profile=profile_dict,
            loss_ema=self._loss_ema,
            loss_std=self._loss_std,
            neighbor_avail=self._neighbor_avail,
            compute_queue=self._compute_queue,
            fusion_candidates=self._fusion_candidates,
            energy_ratio=energy_ratio,
            shard_available=self._shard_available,
            node_tier=node_tier,
            ref_flops=1.0,  # Tier 1 reference
        )

        self._state = state
        return state

    # --------------------------------------------------------------------- #
    # Resource computation
    # --------------------------------------------------------------------- #

    def _find_deepest_valid_cut_layer(self, memory_map: dict) -> Optional[int]:
        """
        Find deepest cut_layer that fits node RAM, including optimizer state.

        Returns None if no cut layer fits (node cannot train).
        """
        for cl in sorted(memory_map.keys(), reverse=True):
            if self.profile.can_fit_cut_layer(cl, memory_map):
                return cl
        return None

    def _apply_memory_constraint(self, cut_layer: int, batch_size: int) -> Tuple[int, int]:
        """Clamp cut_layer and batch_size to fit node memory (with Adam optimizer state)."""
        memory_map = SplittableResNet18.MEMORY_WITH_ADAM_MB

        # First, find deepest valid cut layer
        valid_cut = self._find_deepest_valid_cut_layer(memory_map)
        if valid_cut is None:
            # Node cannot fit any cut layer — return invalid sentinel
            # The step() function will handle this by skipping training
            return cut_layer, batch_size

        # Clamp to deepest valid
        if not self.profile.can_fit_cut_layer(cut_layer, memory_map):
            cut_layer = valid_cut

        # Reduce batch size if needed to stay within 50% RAM
        for bs in self.BATCH_SIZES:
            if bs <= batch_size:
                required = self._estimate_activation_mb(cut_layer, bs)
                if required <= self.profile.ram_mb * 0.5:
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
        # CRITICAL FIX: Use LINEAR penalty, not logarithmic
        # log(H+1)/log(6) for H=5 → 1.0 (full benefit, no penalty awareness)
        # Linear: H=1 → 0.33, H=2 → 0.67, H=3 → 1.0, H=5 → 1.67
        # But we also penalize in reward, so here we just cap the benefit
        # H_factor = np.log(H + 1) / np.log(6.0)  # OLD: too lenient
        # New: cap H_factor at 1.5 to prevent over-estimation of H benefit
        H_factor = min(1.5, (H / 4.0))  # H=5 → 1.25, H=3 → 0.75, H=1 → 0.25

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
            Uses MEMORY_WITH_ADAM_MB (includes optimizer state).
        """
        mask = np.array([True, True, True, True], dtype=bool)

        # Cut layer must fit in memory (including Adam optimizer state)
        memory_map = SplittableResNet18.MEMORY_WITH_ADAM_MB
        for i, cl in enumerate(self.CUT_LAYERS):
            if not self.profile.can_fit_cut_layer(cl, memory_map):
                mask[0] = False
                break

        return mask

    def action_to_dict(self, action: np.ndarray) -> dict:
        """Convert action index to meaningful labels."""
        return {
            "cut_layer": self.CUT_LAYERS[int(action[0])],
            "batch_size": self.BATCH_SIZES[int(action[1])],
            "H": self.H_CHOICES[int(action[2])],
            "target_compute_node": int(action[3]),
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