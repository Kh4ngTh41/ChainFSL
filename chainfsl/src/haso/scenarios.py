"""
MA-HASO Scenarios: Fixed Sparse (B) and Dynamic Dropout (C)

Scenario B: k nodes have data, N-k nodes relay only (IoT network)
Scenario C: k(t) changes over rounds due to dropout/join (mobile networks)
"""

import random
from typing import Dict, List, Tuple, Union, Optional


class ScenarioB:
    """
    Fixed sparse topology: k nodes have training data, N-k nodes relay only.

    Use case: Realistic IoT network where only a subset of devices collect data.
    """

    def __init__(self, N: int, k: int, seed: int = None):
        """
        Initialize Scenario B.

        Args:
            N: Total number of nodes
            k: Number of nodes that have training data
            seed: Random seed for reproducibility
        """
        if k > N:
            raise ValueError(f"k ({k}) cannot exceed N ({N})")
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")

        self.N = N
        self.k = k

        if seed is not None:
            random.seed(seed)

        # Shuffle node IDs and select k as data nodes
        all_nodes = list(range(N))
        random.shuffle(all_nodes)
        self._data_nodes = set(all_nodes[:k])
        self._relay_nodes = set(all_nodes[k:])

    def get_active_nodes(self) -> List[int]:
        """Returns k node IDs that have data."""
        return list(self._data_nodes)

    def get_relay_nodes(self) -> List[int]:
        """Returns N-k node IDs that are relay only."""
        return list(self._relay_nodes)

    def is_data_node(self, node_id: int) -> bool:
        """Check if node has training data."""
        return node_id in self._data_nodes

    def assign_data_shards(self, n_shards: int) -> Dict[int, List[int]]:
        """
        Assign data shards to k data nodes in a balanced manner.

        Args:
            n_shards: Total number of data shards to distribute

        Returns:
            Dict mapping node_id -> list of shard IDs
        """
        shards_per_node = n_shards // self.k
        remainder = n_shards % self.k

        assignment = {}
        shard_id = 0

        for node_id in sorted(self._data_nodes):
            num_shards = shards_per_node + (1 if node_id in list(self._data_nodes)[:remainder] else 0)
            assignment[node_id] = list(range(shard_id, shard_id + num_shards))
            shard_id += num_shards

        return assignment


class ScenarioC:
    """
    Dynamic dropout topology: k(t) changes over round t due to node dropout/join.

    Use case: Mobile networks with unreliable connectivity.
    """

    def __init__(
        self,
        N: int,
        k_init: int,
        p_dropout_base: float = 0.1,
        p_join: float = 0.02,
        k_min: int = 5,
        k_max: Optional[int] = None,
        seed: int = None
    ):
        """
        Initialize Scenario C.

        Args:
            N: Total number of nodes
            k_init: Initial number of nodes with data
            p_dropout_base: Base dropout probability per round
            p_join: Probability of a relay node joining as data node
            k_min: Minimum number of data nodes
            k_max: Maximum number of data nodes (defaults to N)
            seed: Random seed for reproducibility
        """
        if k_init > N:
            raise ValueError(f"k_init ({k_init}) cannot exceed N ({N})")
        if k_min < 1:
            raise ValueError(f"k_min must be at least 1, got {k_min}")

        self.N = N
        self.k_init = k_init
        self.p_dropout_base = p_dropout_base
        self.p_join = p_join
        self.k_min = k_min
        self.k_max = k_max if k_max is not None else N

        if seed is not None:
            random.seed(seed)

        # Initialize active (data) nodes
        all_nodes = list(range(N))
        random.shuffle(all_nodes)
        self._active_nodes = set(all_nodes[:k_init])
        self._round = 0
        self._dropout_history: List[Dict] = []

    def step(self) -> Tuple[List[int], List[int]]:
        """
        Simulate one round of dropout and join events.

        Returns:
            Tuple of (active_nodes, newly_joined_nodes)
        """
        self._round += 1
        newly_joined = set()

        # Determine nodes that drop out this round
        dropout_nodes = set()
        for node_id in self._active_nodes:
            # Tier 4 nodes (highest) have additional dropout probability
            # For simplicity, we use a heuristic: higher node IDs have higher tier
            tier = self._infer_tier(node_id)
            p_dropout = self.compute_node_failure_probability(tier)

            if random.random() < p_dropout:
                dropout_nodes.add(node_id)

        # Remove dropout nodes
        self._active_nodes -= dropout_nodes

        # Record dropout history
        if dropout_nodes:
            self._dropout_history.append({
                "round": self._round,
                "dropout_nodes": list(dropout_nodes),
                "k_before": len(self._active_nodes) + len(dropout_nodes),
                "k_after": len(self._active_nodes)
            })

        # Allow relay nodes to join as data nodes
        relay_nodes = set(range(self.N)) - self._active_nodes
        for node_id in relay_nodes:
            if random.random() < self.p_join:
                if len(self._active_nodes) < self.k_max:
                    self._active_nodes.add(node_id)
                    newly_joined.add(node_id)

        # Ensure we don't go below k_min
        while len(self._active_nodes) < self.k_min:
            # Force-join a relay node
            available_relays = set(range(self.N)) - self._active_nodes
            if not available_relays:
                break
            join_node = random.choice(list(available_relays))
            self._active_nodes.add(join_node)
            newly_joined.add(join_node)

        return list(self._active_nodes), list(newly_joined)

    def _infer_tier(self, node_id: int) -> int:
        """
        Infer node tier based on ID.

        Tier distribution:
        - Tier 4 (highest): top 10% of node IDs
        - Tier 3: next 20%
        - Tier 2: next 30%
        - Tier 1: remaining 40%
        """
        threshold = node_id / self.N
        if threshold >= 0.9:
            return 4
        elif threshold >= 0.7:
            return 3
        elif threshold >= 0.4:
            return 2
        else:
            return 1

    def get_active_nodes(self) -> List[int]:
        """Returns current k nodes with data."""
        return list(self._active_nodes)

    def get_dropout_history(self) -> List[Dict]:
        """Returns history of dropout events."""
        return self._dropout_history.copy()

    def compute_node_failure_probability(self, tier: int) -> float:
        """
        Compute dropout probability based on node tier.

        Higher tier = higher dropout probability.
        P(dropout) = p_dropout_base + 0.05 * (tier == 4)
        """
        base = self.p_dropout_base
        if tier == 4:
            return base + 0.05
        elif tier == 3:
            return base + 0.03
        elif tier == 2:
            return base + 0.01
        else:
            return base

    def get_current_k(self) -> int:
        """Returns current number of active data nodes."""
        return len(self._active_nodes)


def create_scenario(scenario_type: str, **kwargs) -> Union[ScenarioB, ScenarioC]:
    """
    Factory function to create a scenario instance.

    Args:
        scenario_type: "B" or "C"
        **kwargs: Arguments passed to the scenario constructor

    Returns:
        ScenarioB or ScenarioC instance
    """
    if scenario_type.upper() == "B":
        return ScenarioB(**kwargs)
    elif scenario_type.upper() == "C":
        return ScenarioC(**kwargs)
    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}. Use 'B' or 'C'.")


def simulate_round_distribution(N: int, tier_probs: List[float]) -> List[int]:
    """
    Assign tiers to N nodes based on probability distribution.

    Args:
        N: Total number of nodes
        tier_probs: Probability for each tier [p_tier1, p_tier2, p_tier3, p_tier4]

    Returns:
        List of tier assignments for each node ID (0 to N-1)
    """
    if len(tier_probs) != 4:
        raise ValueError("tier_probs must have exactly 4 elements")

    total = sum(tier_probs)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"tier_probs must sum to 1.0, got {total}")

    # Normalize probabilities
    tier_probs = [p / total for p in tier_probs]

    # Assign tiers based on cumulative distribution
    assignments = []
    cumulative = [0.0]
    for p in tier_probs:
        cumulative.append(cumulative[-1] + p)

    for _ in range(N):
        r = random.random()
        for tier in range(1, 5):
            if cumulative[tier - 1] <= r < cumulative[tier]:
                assignments.append(tier)
                break
        else:
            assignments.append(4)  # Fallback to highest tier

    return assignments


if __name__ == "__main__":
    print("=== Scenario B Test ===")
    s_b = ScenarioB(N=50, k=10, seed=42)
    active = s_b.get_active_nodes()
    relay = s_b.get_relay_nodes()
    print(f"Active nodes: {len(active)} (expected 10)")
    print(f"Relay nodes: {len(relay)} (expected 40)")
    assert len(active) == 10, f"Expected 10 active nodes, got {len(active)}"
    assert len(relay) == 40, f"Expected 40 relay nodes, got {len(relay)}"
    assert len(active) + len(relay) == 50, "Active + relay should equal N"
    assert s_b.is_data_node(active[0]), "Active node should be data node"

    # Test shard assignment
    shards = s_b.assign_data_shards(100)
    total_assigned = sum(len(v) for v in shards.values())
    print(f"Shard assignment total: {total_assigned} (expected 100)")
    assert total_assigned == 100

    print("\n=== Scenario C Test ===")
    s_c = ScenarioC(N=50, k_init=15, p_dropout_base=0.1, p_join=0.05, k_min=5, k_max=20, seed=42)
    k0 = s_c.get_current_k()
    print(f"Initial k: {k0} (expected 15)")

    # Run 10 rounds
    k_values = [k0]
    for i in range(10):
        active, joined = s_c.step()
        k_values.append(len(active))
        print(f"Round {i+1}: k={len(active)}, joined={len(joined)}")

        # k should stay within [k_min, k_max]
        assert s_c.k_min <= len(active) <= s_c.k_max, f"k={len(active)} out of bounds [{s_c.k_min}, {s_c.k_max}]"

    print(f"\nk values over rounds: {k_values}")
    print(f"Min k: {min(k_values)}, Max k: {max(k_values)}")

    # Check dropout history
    history = s_c.get_dropout_history()
    print(f"Dropout events recorded: {len(history)}")

    # Test tier probabilities
    tiers = simulate_round_distribution(1000, [0.4, 0.3, 0.2, 0.1])
    tier_counts = [tiers.count(i) for i in range(1, 5)]
    print(f"\nTier distribution for 1000 nodes: {tier_counts}")

    print("\n=== All tests passed! ===")
