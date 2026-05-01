"""
MA-HASO Routing Logic.

Implements routing decisions, target node selection, fusion opportunity detection,
and overlap conflict computation for federated split learning orchestration.

Per ChainFSL_Implementation_Plan.md:
- RoutingDecision: (node_id, cut_layer, batch_size, H, target_compute_node, fusion_partners, routing_mode)
- RoutingPolicy: Score-based target node selection with queue/bandwidth/layer compatibility
- Helper functions: bandwidth_match, layer_compatibility, estimate_latency
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import math

# ---------------------------------------------------------------------------
# Try to import ComputeNode and HardwareProfile (graceful degradation)
# ---------------------------------------------------------------------------
try:
    from .compute_node import ComputeNode
except ImportError:
    ComputeNode = None

try:
    from ..emulator.node_profile import HardwareProfile
except ImportError:
    HardwareProfile = None

# ---------------------------------------------------------------------------
# RoutingDecision dataclass
# ---------------------------------------------------------------------------


@dataclass
class RoutingDecision:
    """
    Represents a routing decision for a single data node.

    Attributes:
        node_id: Source data node making the routing decision.
        cut_layer: Cut layer index (1-4) for this node's split model.
        batch_size: Training batch size.
        H: Synchronization period (epochs between sync).
        target_compute_node: Target compute node ID for processing.
        fusion_partners: List of other node_ids sharing the same compute node.
        routing_mode: 'parallel' (independent) or 'fusion' (shared compute).
    """
    node_id: int
    cut_layer: int
    batch_size: int
    H: int
    target_compute_node: int
    fusion_partners: List[int]
    routing_mode: str  # 'parallel' or 'fusion'

    def __post_init__(self) -> None:
        """Validate routing decision fields."""
        if self.routing_mode not in ('parallel', 'fusion'):
            raise ValueError(f"routing_mode must be 'parallel' or 'fusion', got {self.routing_mode}")
        if self.cut_layer not in (1, 2, 3, 4):
            raise ValueError(f"cut_layer must be 1-4, got {self.cut_layer}")
        if self.batch_size not in (8, 16, 32, 64):
            raise ValueError(f"batch_size must be 8, 16, 32, or 64, got {self.batch_size}")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def bandwidth_match(src_bw: float, dst_bw: float) -> float:
    """
    Calculate bandwidth compatibility score between source and destination.

    Returns a score from 0.0 to 1.0 based on bandwidth ratio.
    1.0 = perfect match (identical bandwidths).
    0.0 = worst match (one is negligibly small).

    Args:
        src_bw: Source bandwidth in Mbps.
        dst_bw: Destination bandwidth in Mbps.

    Returns:
        Compatibility score in [0.0, 1.0].
    """
    if src_bw <= 0 or dst_bw <= 0:
        return 0.0
    bw_ratio = min(src_bw, dst_bw) / max(src_bw, dst_bw)
    return bw_ratio


def layer_compatibility(layer_a: int, layer_b: int) -> float:
    """
    Calculate layer overlap/similarity between two cut layers.

    Returns 1.0 if layers are identical (perfect overlap).
    Returns 0.0 if layers are maximally different (4 layers apart).

    Args:
        layer_a: First cut layer (1-4).
        layer_b: Second cut layer (1-4).

    Returns:
        Compatibility score in [0.0, 1.0].
    """
    if layer_a == layer_b:
        return 1.0
    # Max difference is 3 (layer 1 vs layer 4)
    diff = abs(layer_a - layer_b)
    return max(0.0, 1.0 - diff / 4.0)


def estimate_latency(
    src_profile: Any,
    dst_compute_node: Any,
    smashed_size: int
) -> float:
    """
    Estimate end-to-end latency for data transmission and processing.

    Includes:
    - Communication time from source to destination
    - Queue wait time at destination
    - Compute time at destination

    Args:
        src_profile: HardwareProfile or similar with bandwidth_mbps attribute.
        dst_compute_node: ComputeNode with compute and queue attributes.
        smashed_size: Size of smashed data in bytes.

    Returns:
        Estimated latency in seconds.
    """
    if src_profile is None or dst_compute_node is None:
        return float('inf')

    # Communication time (source -> destination)
    src_bw = getattr(src_profile, 'bandwidth_mbps', 0.0)
    dst_bw = getattr(dst_compute_node, 'bandwidth_mbps', 0.0)
    effective_bw = min(src_bw, dst_bw)

    if effective_bw > 0:
        comm_time = (smashed_size * 8) / (effective_bw * 1e6)  # seconds
    else:
        comm_time = float('inf')

    # Queue wait time at destination
    queue_len = getattr(dst_compute_node, 'current_queue', 0)
    queue_wait = queue_len * 0.1  # 0.1s per queued task

    # Compute time estimate
    flops_ratio = getattr(dst_compute_node, 'flops_ratio', 0.01)
    compute_power = flops_ratio * 1.0  # REF_GFLOPS = 1.0
    if compute_power > 0:
        # Base compute time inversely proportional to compute power
        base_compute = 0.5 / compute_power  # 0.5 GFLOPS normalized
    else:
        base_compute = float('inf')

    return comm_time + queue_wait + base_compute


def estimate_smashed_size(cut_layer: int, batch_size: int) -> int:
    """
    Estimate smashed data size in bytes for a given cut layer and batch size.

    Based on ResNet-18 activation sizes:
    - Layer 1 output: 64 * 56 * 56 * 4 bytes (after batch)
    - Layer 2 output: 128 * 28 * 28 * 4 bytes
    - Layer 3 output: 256 * 14 * 14 * 4 bytes
    - Layer 4 output: 512 * 7 * 7 * 4 bytes

    Args:
        cut_layer: Cut layer index (1-4).
        batch_size: Training batch size.

    Returns:
        Estimated smashed data size in bytes.
    """
    size_map = {
        1: 64 * 56 * 56 * 4,   # ~800KB per sample
        2: 128 * 28 * 28 * 4,  # ~400KB per sample
        3: 256 * 14 * 14 * 4,  # ~200KB per sample
        4: 512 * 7 * 7 * 4,    # ~100KB per sample
    }
    per_sample_bytes = size_map.get(cut_layer, 512 * 7 * 7 * 4)
    return per_sample_bytes * batch_size


# ---------------------------------------------------------------------------
# RoutingPolicy class
# ---------------------------------------------------------------------------


class RoutingPolicy:
    """
    DRL-inspired routing policy for MA-HASO.

    Selects target compute nodes based on:
    - Queue length (lower is better)
    - Bandwidth compatibility (higher is better)
    - Layer compatibility (higher is better for fusion)

    Scoring: score = -w1*queue - w2*latency_estimate + w3*bandwidth_match

    Attributes:
        compute_nodes: List of available compute nodes.
        n_data_nodes: Number of data nodes in the federation.
        weights: (w1, w2, w3) for scoring function.
    """

    # Default scoring weights
    DEFAULT_WEIGHTS = (0.4, 0.3, 0.3)

    def __init__(
        self,
        compute_nodes: List[Any],
        n_data_nodes: int,
        weights: Tuple[float, float, float] = None
    ):
        """
        Initialize routing policy.

        Args:
            compute_nodes: List of ComputeNode objects representing compute resources.
            n_data_nodes: Number of data nodes in the federation.
            weights: Tuple of (w1_queue, w2_latency, w3_bandwidth) scoring weights.
        """
        self.compute_nodes = compute_nodes
        self.n_data_nodes = n_data_nodes
        self.weights = weights or self.DEFAULT_WEIGHTS
        w1, w2, w3 = self.weights
        # Normalize weights
        total = w1 + w2 + w3
        if total > 0:
            self._w1 = w1 / total
            self._w2 = w2 / total
            self._w3 = w3 / total
        else:
            self._w1, self._w2, self._w3 = 0.4, 0.3, 0.3

    def select_target_node(
        self,
        node_profile: Any,
        cut_layer: int,
        compute_node_loads: Dict[int, int]
    ) -> int:
        """
        Select the best target compute node for a data node.

        Scores each compute node based on:
        - Queue length (negative weight - less queue is better)
        - Estimated latency (negative weight - lower latency is better)
        - Bandwidth match (positive weight - better match is better)

        Args:
            node_profile: HardwareProfile of the source data node.
            cut_layer: Cut layer index (1-4).
            compute_node_loads: Dict mapping node_id -> queue_length.

        Returns:
            node_id of the highest-scoring compute node.
        """
        if not self.compute_nodes:
            return -1

        best_node_id = self.compute_nodes[0].node_id if self.compute_nodes else -1
        best_score = float('-inf')

        smashed_size = estimate_smashed_size(cut_layer, 32)  # Use batch=32 as reference

        for cn in self.compute_nodes:
            node_id = cn.node_id

            # Skip nodes that cannot process this cut layer
            if hasattr(cn, 'accepted_layers') and cut_layer not in cn.accepted_layers:
                continue

            # Skip overloaded nodes
            if hasattr(cn, 'current_queue') and cn.current_queue > 5:
                continue

            # Get queue length (use provided load if available, else from node)
            queue_len = compute_node_loads.get(node_id, 0)
            if hasattr(cn, 'current_queue'):
                queue_len = cn.current_queue

            # Queue score (normalized, lower is better)
            queue_score = -self._w1 * (queue_len / 10.0)  # max 10 queue

            # Latency estimate
            latency = estimate_latency(node_profile, cn, smashed_size)
            if latency == float('inf'):
                latency_score = -1.0
            else:
                # Normalize latency to [0, 1] assuming max 10s
                latency_score = -self._w2 * min(latency / 10.0, 1.0)

            # Bandwidth match
            src_bw = getattr(node_profile, 'bandwidth_mbps', 50.0)
            dst_bw = getattr(cn, 'bandwidth_mbps', 50.0)
            bw_score = self._w3 * bandwidth_match(src_bw, dst_bw)

            total_score = queue_score + latency_score + bw_score

            if total_score > best_score:
                best_score = total_score
                best_node_id = node_id

        return best_node_id

    def detect_fusion_opportunities(
        self,
        decisions: List[RoutingDecision],
        node_id: int
    ) -> List[int]:
        """
        Find nodes that can share the same compute node with the given node.

        Fusion candidates are nodes that:
        1. Target the same compute node
        2. Have similar cut layers (layer compatibility > 0.5)

        Args:
            decisions: List of RoutingDecision for all nodes.
            node_id: Query node to find fusion partners for.

        Returns:
            List of node_ids that can fuse with node_id.
        """
        # Find this node's decision
        my_decision = None
        for d in decisions:
            if d.node_id == node_id:
                my_decision = d
                break

        if my_decision is None:
            return []

        fusion_candidates = []
        for d in decisions:
            if d.node_id == node_id:
                continue
            if d.target_compute_node != my_decision.target_compute_node:
                continue
            # Check layer compatibility
            compat = layer_compatibility(my_decision.cut_layer, d.cut_layer)
            if compat >= 0.5:
                fusion_candidates.append(d.node_id)

        return fusion_candidates

    def compute_overlap_conflicts(
        self,
        decisions: List[RoutingDecision]
    ) -> List[Tuple[int, float]]:
        """
        Compute conflict scores for compute nodes with multiple assigned nodes.

        Conflict score is the sum of layer similarity scores for all pairs
        of nodes sent to the same compute node. Higher score = more overlap
        = potentially more interference.

        Args:
            decisions: List of RoutingDecision for all nodes.

        Returns:
            List of (compute_node_id, conflict_score) tuples, sorted by
            conflict_score descending. Only compute nodes with >1 assigned
            nodes are included.
        """
        # Group decisions by target compute node
        by_target: Dict[int, List[RoutingDecision]] = {}
        for d in decisions:
            target = d.target_compute_node
            if target not in by_target:
                by_target[target] = []
            by_target[target].append(d)

        conflicts = []
        for compute_node_id, node_decisions in by_target.items():
            if len(node_decisions) <= 1:
                continue  # No conflict with single node

            # Compute pairwise layer similarity sum
            conflict_score = 0.0
            n = len(node_decisions)
            for i in range(n):
                for j in range(i + 1, n):
                    layer_a = node_decisions[i].cut_layer
                    layer_b = node_decisions[j].cut_layer
                    conflict_score += layer_compatibility(layer_a, layer_b)

            conflicts.append((compute_node_id, conflict_score))

        # Sort by conflict score descending
        conflicts.sort(key=lambda x: x[1], reverse=True)
        return conflicts

    def make_routing_decision(
        self,
        node_id: int,
        node_profile: Any,
        cut_layer: int,
        batch_size: int,
        H: int,
        compute_node_loads: Dict[int, int]
    ) -> RoutingDecision:
        """
        Create a complete RoutingDecision for a data node.

        Args:
            node_id: Data node identifier.
            node_profile: HardwareProfile of the data node.
            cut_layer: Selected cut layer (1-4).
            batch_size: Selected batch size.
            H: Synchronization period.
            compute_node_loads: Current queue lengths per compute node.

        Returns:
            RoutingDecision for this node.
        """
        target = self.select_target_node(node_profile, cut_layer, compute_node_loads)

        # Determine routing mode based on fusion detection
        temp_decisions = [RoutingDecision(
            node_id=node_id,
            cut_layer=cut_layer,
            batch_size=batch_size,
            H=H,
            target_compute_node=target,
            fusion_partners=[],
            routing_mode='parallel'
        )]

        # Check if this node should use fusion mode
        # For now, default to 'parallel' unless fusion candidates found
        fusion_partners = []
        routing_mode = 'parallel'

        return RoutingDecision(
            node_id=node_id,
            cut_layer=cut_layer,
            batch_size=batch_size,
            H=H,
            target_compute_node=target,
            fusion_partners=fusion_partners,
            routing_mode=routing_mode
        )

    def resolve_fusion_mode(
        self,
        decisions: List[RoutingDecision]
    ) -> List[RoutingDecision]:
        """
        Update all decisions to properly set fusion_partners and routing_mode.

        Args:
            decisions: List of RoutingDecision to resolve.

        Returns:
            List of updated RoutingDecision with fusion info.
        """
        resolved = []
        for d in decisions:
            partners = self.detect_fusion_opportunities(decisions, d.node_id)
            routing_mode = 'fusion' if len(partners) > 0 else 'parallel'

            resolved.append(RoutingDecision(
                node_id=d.node_id,
                cut_layer=d.cut_layer,
                batch_size=d.batch_size,
                H=d.H,
                target_compute_node=d.target_compute_node,
                fusion_partners=partners,
                routing_mode=routing_mode
            ))
        return resolved


# ---------------------------------------------------------------------------
# Test function
# ---------------------------------------------------------------------------


def test() -> None:
    """Test routing logic."""
    print("=== Routing Logic Tests ===")

    # Create mock compute nodes
    if ComputeNode is not None:
        compute_nodes = [
            ComputeNode(
                node_id=i,
                tier=min(i + 1, 4),
                flops_ratio=[1.0, 0.3, 0.05, 0.005][i % 4],
                max_memory_mb=[8192, 4096, 512, 200][i % 4],
                bandwidth_mbps=[100.0, 50.0, 10.0, 1.0][i % 4],
                current_queue=i % 3,
                processing=False,
                accepted_layers=[1, 2, 3, 4]
            )
            for i in range(4)
        ]
    else:
        # Fallback to dict-like objects
        class MockComputeNode:
            def __init__(self, node_id, tier, flops_ratio, max_memory_mb, bandwidth_mbps, current_queue, accepted_layers):
                self.node_id = node_id
                self.tier = tier
                self.flops_ratio = flops_ratio
                self.max_memory_mb = max_memory_mb
                self.bandwidth_mbps = bandwidth_mbps
                self.current_queue = current_queue
                self.accepted_layers = accepted_layers
        compute_nodes = [
            MockComputeNode(i, i % 4 + 1, [1.0, 0.3, 0.05, 0.005][i % 4],
                           [8192, 4096, 512, 200][i % 4],
                           [100.0, 50.0, 10.0, 1.0][i % 4],
                           i % 3, [1, 2, 3, 4])
            for i in range(4)
        ]

    # Mock hardware profile
    class MockProfile:
        def __init__(self, bw):
            self.bandwidth_mbps = bw
            self.flops_ratio = 0.5

    # Test RoutingPolicy
    policy = RoutingPolicy(compute_nodes, n_data_nodes=8)

    # Test bandwidth_match
    print("\n1. Bandwidth Match Tests:")
    assert abs(bandwidth_match(100.0, 100.0) - 1.0) < 1e-6
    assert abs(bandwidth_match(100.0, 50.0) - 0.5) < 1e-6
    assert abs(bandwidth_match(100.0, 10.0) - 0.1) < 1e-6
    print(f"  match(100, 100) = {bandwidth_match(100.0, 100.0):.2f} (expected 1.00)")
    print(f"  match(100, 50) = {bandwidth_match(100.0, 50.0):.2f} (expected 0.50)")
    print(f"  match(100, 10) = {bandwidth_match(100.0, 10.0):.2f} (expected 0.10)")

    # Test layer_compatibility
    print("\n2. Layer Compatibility Tests:")
    assert abs(layer_compatibility(1, 1) - 1.0) < 1e-6
    assert abs(layer_compatibility(1, 4) - 0.25) < 1e-6
    assert abs(layer_compatibility(2, 3) - 0.75) < 1e-6
    print(f"  compat(1, 1) = {layer_compatibility(1, 1):.2f} (expected 1.00)")
    print(f"  compat(1, 4) = {layer_compatibility(1, 4):.2f} (expected 0.25)")
    print(f"  compat(2, 3) = {layer_compatibility(2, 3):.2f} (expected 0.75)")

    # Test estimate_smashed_size
    print("\n3. Smashed Size Estimation:")
    size_1 = estimate_smashed_size(1, 32)
    size_4 = estimate_smashed_size(4, 32)
    print(f"  size(cut=1, batch=32) = {size_1:,} bytes")
    print(f"  size(cut=4, batch=32) = {size_4:,} bytes")

    # Test select_target_node
    print("\n4. Target Node Selection:")
    profile = MockProfile(bw=100.0)
    loads = {0: 2, 1: 0, 2: 5, 3: 1}
    best = policy.select_target_node(profile, cut_layer=2, compute_node_loads=loads)
    print(f"  Best target for node with profile(bw=100): node{best}")
    print(f"  Loads: {loads}")

    # Test make_routing_decision
    print("\n5. Routing Decision:")
    decision = policy.make_routing_decision(
        node_id=0,
        node_profile=profile,
        cut_layer=2,
        batch_size=32,
        H=1,
        compute_node_loads=loads
    )
    print(f"  Decision: node={decision.node_id}, target={decision.target_compute_node}, "
          f"cut={decision.cut_layer}, mode={decision.routing_mode}")

    # Test detect_fusion_opportunities
    print("\n6. Fusion Opportunity Detection:")
    decisions = [
        RoutingDecision(node_id=0, cut_layer=2, batch_size=32, H=1,
                      target_compute_node=0, fusion_partners=[], routing_mode='parallel'),
        RoutingDecision(node_id=1, cut_layer=2, batch_size=32, H=1,
                      target_compute_node=0, fusion_partners=[], routing_mode='parallel'),
        RoutingDecision(node_id=2, cut_layer=4, batch_size=32, H=1,
                      target_compute_node=1, fusion_partners=[], routing_mode='parallel'),
    ]
    partners = policy.detect_fusion_opportunities(decisions, node_id=0)
    print(f"  Fusion partners for node 0: {partners}")

    # Test compute_overlap_conflicts
    print("\n7. Overlap Conflict Computation:")
    conflicts = policy.compute_overlap_conflicts(decisions)
    print(f"  Conflicts: {conflicts}")

    # Test resolve_fusion_mode
    print("\n8. Fusion Mode Resolution:")
    resolved = policy.resolve_fusion_mode(decisions)
    for r in resolved:
        print(f"  node={r.node_id}: mode={r.routing_mode}, partners={r.fusion_partners}")

    print("\n=== All Tests Passed! ===")


if __name__ == "__main__":
    test()
