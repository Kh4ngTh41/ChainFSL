"""
Reward function for MA-HASO (Eq. 7 from ChainFSL paper).

r_t = -α·T_comp - β·T_comm + γ·φ·ΔF

Components:
- T_comp: Computation time (based on cut_layer, node flops_ratio, batch_size)
- T_comm: Communication time (based on activation size, node bandwidth)
- φ: Shapley value (EMA-smoothed contribution estimate from GTM)
- ΔF: Model accuracy improvement (validation loss reduction)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class RewardConfig:
    """Configuration for reward function hyperparameters."""

    alpha: float = 1.0    # Computation cost weight
    beta: float = 0.5     # Communication cost weight
    gamma: float = 0.1    # Shapley-weighted accuracy gain
    shapley_decay: float = 0.9  # EMA decay for Shapley smoothing


class RewardFunction:
    """
    Computes per-step reward for MA-HASO.

    Encapsulates all reward computation logic so it can be tested in isolation
    and reused across environments.
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()

    def compute(
        self,
        T_comp: float,
        T_comm: float,
        shapley_ema: float,
        delta_F: float,
    ) -> float:
        """
        Compute reward for one step.

        Args:
            T_comp: Computation time in seconds.
            T_comm: Communication time in seconds.
            shapley_ema: EMA-smoothed Shapley value.
            delta_F: Model accuracy improvement (reduction in loss).

        Returns:
            Scalar reward.
        """
        return (
            -self.config.alpha * T_comp
            - self.config.beta * T_comm
            + self.config.gamma * shapley_ema * delta_F
        )

    def compute_from_info(self, info: dict) -> float:
        """
        Compute reward directly from step info dict.

        Args:
            info: Step info dict with keys: T_comp, T_comm, shapley_ema, delta_F.

        Returns:
            Scalar reward.
        """
        return self.compute(
            T_comp=info.get("T_comp", 0.0),
            T_comm=info.get("T_comm", 0.0),
            shapley_ema=info.get("shapley_ema", 0.0),
            delta_F=info.get("delta_F", 0.0),
        )

    def normalize_reward(self, reward: float, window: list[float] = None) -> float:
        """
        Normalize reward using running statistics.

        Args:
            reward: Raw reward.
            window: Recent reward history for normalization.

        Returns:
            Normalized reward.
        """
        if window is None or len(window) < 10:
            return reward

        mean = np.mean(window)
        std = np.std(window)
        if std < 1e-6:
            return reward
        return (reward - mean) / (std + 1e-6)


class FairnessPenalty:
    """
    Adds fairness penalty based on Gini coefficient of contribution distribution.

    Encourages MA-HASO to select configurations that don't starve low-tier nodes.
    """

    @staticmethod
    def gini_coefficient(values: list[float]) -> float:
        """
        Compute Gini coefficient (0 = perfect equality, 1 = maximum inequality).

        Args:
            values: List of contribution values (e.g., rewards or Shapley values).

        Returns:
            Gini coefficient in [0, 1].
        """
        if not values:
            return 0.0

        arr = np.array(sorted(values))
        n = len(arr)
        if n == 0 or np.sum(arr) == 0:
            return 0.0

        cumsum = np.cumsum(arr)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

    @staticmethod
    def fairness_bonus(rewards: list[float], nu: float = 0.2) -> float:
        """
        Compute fairness bonus to add to reward.

        Args:
            rewards: List of rewards for all nodes this round.
            nu: Fairness weight (0.2 as per paper).

        Returns:
            Bonus to add to individual reward.
        """
        gini = FairnessPenalty.gini_coefficient(rewards)
        return -nu * gini


def validate_reward_shape(rewards: list[float], tolerance: float = 5.0) -> bool:
    """
    Validate that reward values are within expected bounds.

    Args:
        rewards: List of reward values.
        tolerance: Max absolute reward magnitude.

    Returns:
        True if all rewards are within [-tolerance, tolerance].
    """
    return all(-tolerance <= r <= tolerance for r in rewards)