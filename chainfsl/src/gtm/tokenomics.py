"""
GTM Tokenomics — Reward distribution with deflationary schedule.

Implements:
- Deflationary reward schedule (Eq. 15)
- Shapley-weighted reward distribution (Eq. 14)
- Slashing for lazy/poison nodes
- Nash Equilibrium validation

Also provides NashValidator for Experiment E5.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Callable, Tuple
from collections import defaultdict
import math


@dataclass
class TokenomicsConfig:
    """Configuration for TokenomicsEngine."""

    R0: float = 1000.0       # Initial reward pool (R_0)
    R_min: float = 10.0      # Minimum reward floor (R_min)
    T_halving: int = 50      # Halving period in rounds
    decay_mode: str = "exponential"  # 'exponential' | 'linear' | 'step'
    step_schedule: Optional[List[Tuple[int, float]]] = None  # For step mode

    # Slashing
    lazy_penalty: float = 0.3     # Penalty rate for lazy nodes
    poison_penalty: float = 1.0  # Full penalty for poison nodes
    quality_threshold: float = 0.01  # q_min for superlinear penalty

    # Sybil prevention
    stake_min: float = 10.0      # Minimum stake to participate
    PENALTY_LAZY: float = 1000.0  # Fixed slashing penalty (>> reward)

    # Nash equilibrium
    enable_nash_check: bool = True


class TokenomicsEngine:
    """
    Game-theoretic tokenomics with Shapley-based reward distribution.

    Implements Eq. 14-15 from ChainFSL paper:
    - Deflationary reward schedule
    - Shapley-proportional distribution
    - Slashing for lazy/poison nodes
    - Sybil attack prevention via Theorem 3 check
    """

    def __init__(self, config: Optional[TokenomicsConfig] = None):
        """
        Args:
            config: TokenomicsConfig. Defaults to TokenomicsConfig().
        """
        self.config = config or TokenomicsConfig()
        self.current_round = 0
        self.total_distributed = 0.0
        self.slashing_history: Dict[int, List[dict]] = defaultdict(list)
        self._round_rewards: Dict[int, Dict[int, float]] = {}

    def total_reward(self, t: int) -> float:
        """
        Eq. 15: Deflationary schedule R_total^(t) = max(R_min, R0 * (1 - t/T_halving)+)

        Args:
            t: Round number (0-indexed).

        Returns:
            Base reward pool for round t.
        """
        if self.config.decay_mode == "exponential":
            r = self.config.R0 * max(0.0, 1.0 - t / self.config.T_halving)
        elif self.config.decay_mode == "linear":
            r = max(0.0, self.config.R0 - (self.config.R0 - self.config.R_min) * t / self.config.T_halving)
        elif self.config.decay_mode == "step" and self.config.step_schedule:
            r = self._step_reward(t)
        else:
            r = self.config.R0 * (self.config.R0 / self.config.R0) ** (t / self.config.T_halving)
            r = max(self.config.R_min, r * (self.config.R0 / max(self.config.R0, 1)))

        return max(self.config.R_min, r)

    def _step_reward(self, t: int) -> float:
        """Step-function reward schedule."""
        if not self.config.step_schedule:
            return self.config.R0
        for threshold, reward in sorted(self.config.step_schedule, reverse=True):
            if t >= threshold:
                return reward
        return self.config.R0

    def detect_lazy_nodes(
        self,
        shapley_values: Dict[int, float],
    ) -> Set[int]:
        """
        Detect lazy nodes based on low Shapley contribution.

        Args:
            shapley_values: Computed Shapley values.

        Returns:
            Set of lazy node IDs.
        """
        if not shapley_values:
            return set()

        max_phi = max(shapley_values.values())
        threshold = self.config.quality_threshold * max_phi

        return {
            node_id for node_id, phi in shapley_values.items()
            if phi < threshold
        }

    def detect_poison_nodes(
        self,
        node_ids: List[int],
        validation_fn: Callable[[int], Tuple[bool, float]],
    ) -> Set[int]:
        """
        Detect poison nodes via validation function.

        Args:
            node_ids: List of node IDs to check.
            validation_fn: Function node_id → (is_valid, confidence).

        Returns:
            Set of poison node IDs.
        """
        poison = set()
        for node_id in node_ids:
            is_valid, confidence = validation_fn(node_id)
            if not is_valid and confidence > 0.9:
                poison.add(node_id)
        return poison

    def calculate_slashing(
        self,
        node_id: int,
        lazy_nodes: Set[int],
        poison_nodes: Set[int],
    ) -> float:
        """Calculate slashing rate for a node."""
        if node_id in poison_nodes:
            rate = self.config.poison_penalty
            self._record_slashing(node_id, "poison", rate)
        elif node_id in lazy_nodes:
            rate = self.config.lazy_penalty
            self._record_slashing(node_id, "lazy", rate)
        else:
            rate = 0.0
        return rate

    def distribute(
        self,
        shapley_values: Dict[int, float],
        verification_results: Dict[int, dict],
    ) -> Dict[int, float]:
        """
        Distribute rewards per Eq. 14.

        R_i = R_total * (φ_i / Σφ_j) * (1 - s_i)

        Args:
            shapley_values: Computed Shapley values.
            verification_results: node_id -> {valid, penalty, ...}.

        Returns:
            Dict node_id -> net reward (can be negative if slashed).
        """
        R_total = self.total_reward(self.current_round)
        total_phi = sum(max(0.0, phi) for phi in shapley_values.values())

        if total_phi == 0:
            return {nid: 0.0 for nid in shapley_values}

        # Detect lazy/poison
        lazy_nodes = self.detect_lazy_nodes(shapley_values)
        poison_nodes = set()
        for nid, res in verification_results.items():
            if not res.get("valid", True):
                poison_nodes.add(nid)

        rewards = {}
        for node_id, phi in shapley_values.items():
            # Base reward proportional to Shapley
            reward = R_total * max(0.0, phi) / total_phi

            # Apply slashing
            slashing_rate = self.calculate_slashing(node_id, lazy_nodes, poison_nodes)
            net_reward = reward * (1.0 - slashing_rate)

            rewards[node_id] = net_reward
            self.total_distributed += net_reward

        self._round_rewards[self.current_round] = rewards
        self.current_round += 1
        return rewards

    def _record_slashing(self, node_id: int, reason: str, rate: float) -> None:
        """Record slashing event for audit."""
        self.slashing_history[node_id].append({
            "round": self.current_round,
            "reason": reason,
            "rate": rate,
        })

    def check_sybil_profitable(
        self,
        m_sybil: int,
        N_total: int,
        R_total: Optional[float] = None,
    ) -> bool:
        """
        Theorem 3: Check if Sybil attack is profitable.

        E[ProfitSybil] = m/(N+m) * R_total - m * S_min

        Args:
            m_sybil: Number of Sybil nodes.
            N_total: Number of honest nodes.
            R_total: Current reward pool. Uses current_round if None.

        Returns:
            True if attack is profitable (should be prevented).
        """
        if R_total is None:
            R_total = self.total_reward(self.current_round)

        expected_profit = (m_sybil / (N_total + m_sybil)) * R_total - m_sybil * self.config.stake_min
        return expected_profit > 0

    def get_cumulative_reward(self, node_id: int) -> float:
        """Sum of all rewards received by node across all rounds."""
        total = 0.0
        for round_rewards in self._round_rewards.values():
            total += round_rewards.get(node_id, 0.0)
        return total

    def get_total_supply_schedule(self, max_rounds: int) -> float:
        """Compute total tokens that will be issued over max_rounds."""
        return sum(self.total_reward(t) for t in range(max_rounds))


class NashValidator:
    """
    Empirical Nash Equilibrium validator for Experiment E5.

    Simulates rational agents making honest vs. lazy decisions
    and verifies that honest behavior is a Nash equilibrium.
    """

    def __init__(self, tokenomics: TokenomicsEngine):
        self.tk = tokenomics

    def simulate_agent_decision(
        self,
        node_id: int,
        cost_honest: float,
        honest_phi: float,
        lazy_phi: float,
        R_total: float,
        N: int,
    ) -> str:
        """
        Simulate decision for a rational agent.

        Compares utility of honest vs. lazy behavior:
        - honest: u = R * φ_honest / Σφ - cost
        - lazy: u = R * φ_lazy / Σφ - penalty

        Returns:
            'honest' | 'lazy' | 'abstain'
        """
        # Assume normalized: Σφ = 1.0
        u_honest = R_total * honest_phi - cost_honest
        u_lazy = R_total * lazy_phi - self.tk.config.PENALTY_LAZY

        if u_honest > u_lazy and u_honest > 0:
            return "honest"
        elif u_lazy > u_honest and u_lazy > 0:
            return "lazy"
        else:
            return "abstain"

    def verify_nash_equilibrium(
        self,
        results: Dict[int, str],
        shapley_values: Dict[int, float],
        R_total: float,
    ) -> Tuple[bool, float]:
        """
        Verify Nash equilibrium condition.

        At equilibrium, no agent can improve by unilateral deviation.

        Args:
            results: node_id -> 'honest' | 'lazy' | 'abstain'
            shapley_values: Shapley values for all nodes.
            R_total: Current reward pool.

        Returns:
            (is_equilibrium, max_deviation) tuple.
        """
        honest_agents = [nid for nid, r in results.items() if r == "honest"]
        lazy_agents = [nid for nid, r in results.items() if r == "lazy"]

        if not honest_agents:
            return False, float("inf")

        # All honest agents should have no profitable deviation
        max_deviation = 0.0
        for node_id in honest_agents:
            honest_util = R_total * shapley_values.get(node_id, 0.0)
            lazy_util = R_total * shapley_values.get(node_id, 0.0) * 0.3 - self.tk.config.PENALTY_LAZY

            deviation = honest_util - lazy_util
            max_deviation = max(max_deviation, abs(deviation))

        is_equilibrium = all(
            results.get(nid) == "honest"
            for nid in honest_agents
        ) and max_deviation < 1e-6

        return is_equilibrium, max_deviation

    def run_simulation(
        self,
        node_costs: Dict[int, float],
        shapley_values: Dict[int, float],
        R_total: float,
        N: int,
    ) -> Dict[int, str]:
        """
        Run Nash equilibrium simulation for all agents.

        Args:
            node_costs: node_id -> cost of being honest.
            shapley_values: node_id -> Shapley value.
            R_total: Current reward pool.
            N: Total number of nodes.

        Returns:
            node_id -> decision ('honest' | 'lazy' | 'abstain').
        """
        results = {}
        for node_id, cost in node_costs.items():
            honest_phi = shapley_values.get(node_id, 0.0)
            lazy_phi = honest_phi * 0.3  # Lazy nodes contribute less

            results[node_id] = self.simulate_agent_decision(
                node_id, cost, honest_phi, lazy_phi, R_total, N
            )
        return results