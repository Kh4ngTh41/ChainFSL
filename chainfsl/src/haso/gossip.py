"""
HASO Gossip module integration.

Provides lightweight resource heartbeat (LRH) broadcast and neighbor
availability queries for MA-HASO state observations.
"""

from typing import Optional
from ..emulator.network_emulator import GossipProtocol


class HASOGossip:
    """
    Gossip protocol wrapper tailored for MA-HASO.

    Each node periodically broadcasts its LRH (Lightweight Resource Heartbeat):
    - flops_ratio, ram_mb, bandwidth_mbps, reputation, load, timestamp

    Other nodes query neighbor availability to inform target_node selection
    and action masking.
    """

    def __init__(self, shared_table: Optional[dict] = None, fanout: int = 3):
        """
        Args:
            shared_table: Shared dict (e.g., Manager().dict()).
            fanout: Number of neighbors to sample per gossip round.
        """
        self._protocol = GossipProtocol(shared_table=shared_table, fanout=fanout)

    def broadcast_lrh(self, node_id: int, profile, current_load: float = 0.0) -> None:
        """
        Broadcast LRH for a node.

        Args:
            node_id: Node ID.
            profile: HardwareProfile instance.
            current_load: Current CPU/load estimate (0-1).
        """
        lrh = {
            "flops_ratio": profile.flops_ratio,
            "ram_mb": profile.ram_mb,
            "bandwidth_mbps": profile.bandwidth_mbps,
            "reputation": profile.reputation,
            "load": current_load,
        }
        self._protocol.broadcast(node_id, lrh)

    def get_neighbor_info(self, node_id: int, k: Optional[int] = None) -> list[dict]:
        """Return k neighbor LRH dicts."""
        return self._protocol.get_neighbors(node_id, k=k)

    def mean_neighbor_availability(self, node_id: int) -> float:
        """Return mean availability across neighbors."""
        return self._protocol.mean_neighbor_availability(node_id)

    def get_best_target(self, node_id: int, exclude_self: bool = True, k: int = 5) -> Optional[int]:
        """
        Return the node_id with highest reputation among neighbors.

        Args:
            node_id: Querying node.
            exclude_self: Whether to exclude self from candidates.
            k: Number of top candidates to consider.

        Returns:
            Best target node_id, or None if no neighbors.
        """
        # Build candidate list from the shared table: (node_id, lrh)
        all_candidates = [
            (nid, info) for nid, info in self._protocol._table.items()
            if nid != node_id or not exclude_self
        ]
        if not all_candidates:
            return None

        # Pick neighbor with highest reputation
        best_nid, _ = max(all_candidates, key=lambda x: x[1].get("reputation", 0.0))
        return best_nid

    def update_from_chainfsl(self, gossip_protocol: GossipProtocol) -> None:
        """
        Sync from an existing GossipProtocol instance (e.g., from emulator).

        Args:
            gossip_protocol: GossipProtocol instance to sync from.
        """
        self._protocol = gossip_protocol