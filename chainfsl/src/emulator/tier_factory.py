"""
Tier-based factory for creating heterogeneous device populations.

Creates N nodes according to a specified tier distribution,
supporting reproducible simulations via seed control.
"""

import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Union

from .node_profile import HardwareProfile, create_profile


@dataclass
class TierDistribution:
    """
    Probability distribution over hardware tiers.

    Attributes:
        tiers: List of tier integers (e.g., [1, 2, 3, 4]).
        probabilities: Probability weight for each tier (must sum ~1.0).
    """

    tiers: List[int]
    probabilities: List[float]

    def __post_init__(self) -> None:
        """Validate that probabilities sum to 1.0."""
        if len(self.tiers) != len(self.probabilities):
            raise ValueError("tiers and probabilities must have same length")
        total = sum(self.probabilities)
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Probabilities must sum to 1.0, got {total}")


class TierFactory:
    """
    Factory for creating device populations with tier-based heterogeneity.

    Supports probability-based tier assignment and reproducible randomness.
    Can be constructed from a TierDistribution or directly with tier/probability lists.

    Example:
        factory = TierFactory(
            tiers=[1, 2, 3, 4],
            probabilities=[0.1, 0.3, 0.4, 0.2],
            seed=42
        )
        nodes = factory.create_nodes(50)
    """

    def __init__(
        self,
        tiers: Optional[List[int]] = None,
        probabilities: Optional[List[float]] = None,
        seed: Optional[int] = None,
        distribution: Optional[TierDistribution] = None,
    ):
        """
        Initialize factory.

        Args:
            tiers: List of tier integers.
            probabilities: Probability weights per tier.
            seed: Random seed for reproducibility.
            distribution: TierDistribution instance (alternative to tiers/probabilities).
        """
        if distribution is not None:
            self.tiers = distribution.tiers
            self.probabilities = distribution.probabilities
        elif tiers is not None and probabilities is not None:
            self.tiers = tiers
            self.probabilities = probabilities
        else:
            raise ValueError("Must provide either distribution or (tiers, probabilities)")

        self._rng = random.Random(seed)
        total = sum(self.probabilities)
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Probabilities must sum to 1.0, got {total}")

    def create_nodes(self, n: int) -> List[HardwareProfile]:
        """
        Create N nodes sampled from the tier distribution.

        Args:
            n: Number of nodes to create.

        Returns:
            List of HardwareProfile instances.
        """
        nodes = []
        for i in range(n):
            tier = self._rng.choices(self.tiers, weights=self.probabilities)[0]
            nodes.append(create_profile(node_id=i, tier=tier))
        return nodes

    def create_balanced_nodes(self, n: int) -> List[HardwareProfile]:
        """
        Create N nodes with equal representation per tier (round-robin).

        Useful for deterministic ablation studies.

        Args:
            n: Total number of nodes (should be divisible by len(tiers)).

        Returns:
            List of HardwareProfile instances.
        """
        if n % len(self.tiers) != 0:
            raise ValueError(f"n={n} not divisible by {len(self.tiers)} tiers")
        per_tier = n // len(self.tiers)
        nodes = []
        node_id = 0
        for tier in self.tiers:
            for _ in range(per_tier):
                nodes.append(create_profile(node_id=node_id, tier=tier))
                node_id += 1
        return nodes

    def get_tier_counts(self, nodes: List[HardwareProfile]) -> Dict[int, int]:
        """Count nodes per tier."""
        counts: Dict[int, int] = {t: 0 for t in self.tiers}
        for node in nodes:
            counts[node.tier] = counts.get(node.tier, 0) + 1
        return counts

    def filter_by_tier(self, nodes: List[HardwareProfile], tier: int) -> List[HardwareProfile]:
        """Return only nodes of a specific tier."""
        return [n for n in nodes if n.tier == tier]

    def filter_by_memory(self, nodes: List[HardwareProfile], min_ram_mb: int) -> List[HardwareProfile]:
        """Return nodes with at least min_ram_mb available."""
        return [n for n in nodes if n.ram_mb >= min_ram_mb]


# Module-level default distribution matching config/default.yaml
DEFAULT_DISTRIBUTION = TierDistribution(
    tiers=[1, 2, 3, 4],
    probabilities=[0.1, 0.3, 0.4, 0.2],
)

DEFAULT_FACTORY = TierFactory(distribution=DEFAULT_DISTRIBUTION, seed=42)


def create_nodes(n: int, distribution: Optional[Union[TierDistribution, TierFactory]] = None) -> List[HardwareProfile]:
    """
    Convenience function to create N nodes using default distribution.

    Args:
        n: Number of nodes.
        distribution: Optional TierDistribution or TierFactory.
                       Uses DEFAULT_FACTORY if None.

    Returns:
        List of HardwareProfile instances.
    """
    if distribution is None:
        factory = DEFAULT_FACTORY
    elif isinstance(distribution, TierFactory):
        factory = distribution
    elif isinstance(distribution, TierDistribution):
        factory = TierFactory(distribution=distribution, seed=42)
    else:
        raise TypeError(f"Expected TierDistribution or TierFactory, got {type(distribution)}")

    return factory.create_nodes(n)