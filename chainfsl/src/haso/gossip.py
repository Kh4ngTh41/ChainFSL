"""
HASO Gossip module integration.

Provides lightweight resource heartbeat (LRH) broadcast and neighbor
availability queries for MA-HASO state observations.

Extended for hierarchical MA-HASO:
- Intra-cluster gossip: cluster members communicate directly
- Inter-cluster gossip: cluster-heads coordinate across clusters
"""

from typing import Optional, List, Dict, Any
from ..emulator.network_emulator import GossipProtocol


class HASOGossip:
    """
    Gossip protocol wrapper tailored for MA-HASO.

    Each node periodically broadcasts its LRH (Lightweight Resource Heartbeat):
    - flops_ratio, ram_mb, bandwidth_mbps, reputation, load, timestamp

    Other nodes query neighbor availability to inform target_node selection
    and action masking.

    Extended for hierarchical clusters:
    - cluster_id included in LRH
    - Intra-cluster and inter-cluster routing supported
    """

    def __init__(
        self,
        shared_table: Optional[dict] = None,
        fanout: int = 3,
        cluster_manager=None,
    ):
        """
        Args:
            shared_table: Shared dict (e.g., Manager().dict()).
            fanout: Number of neighbors to sample per gossip round.
            cluster_manager: Optional ClusterManager for cluster-aware routing.
        """
        self._protocol = GossipProtocol(shared_table=shared_table, fanout=fanout)
        self._cluster_manager = cluster_manager

    def set_cluster_manager(self, cluster_manager) -> None:
        """Set cluster manager for cluster-aware gossip."""
        self._cluster_manager = cluster_manager

    def broadcast_lrh(
        self,
        node_id: int,
        profile,
        current_load: float = 0.0,
        cluster_id: Optional[int] = None,
    ) -> None:
        """
        Broadcast LRH for a node.

        Args:
            node_id: Node ID.
            profile: HardwareProfile instance.
            current_load: Current CPU/load estimate (0-1).
            cluster_id: Optional cluster ID for hierarchical gossip.
        """
        lrh = {
            "flops_ratio": profile.flops_ratio,
            "ram_mb": profile.ram_mb,
            "bandwidth_mbps": profile.bandwidth_mbps,
            "reputation": profile.reputation,
            "load": current_load,
            "cluster_id": cluster_id,
        }
        self._protocol.broadcast(node_id, lrh)

    def broadcast_intra_cluster(
        self,
        node_id: int,
        profile,
        current_load: float = 0.0,
        fanout: int = 3,
    ) -> None:
        """
        Broadcast LRH to intra-cluster neighbors only.

        Args:
            node_id: Node ID (should be cluster-head for meaningful broadcast).
            profile: HardwareProfile instance.
            current_load: Current CPU/load estimate (0-1).
            fanout: Number of cluster members to notify.
        """
        if self._cluster_manager is None:
            # Fallback to regular broadcast
            self.broadcast_lrh(node_id, profile, current_load)
            return

        cluster_id = self._cluster_manager.get_cluster_id(node_id)
        if cluster_id is None:
            self.broadcast_lrh(node_id, profile, current_load)
            return

        # Get cluster members (excluding self)
        intra_neighbors = self._cluster_manager.get_intra_cluster_neighbors(
            node_id, fanout=fanout, exclude_head=False
        )

        lrh = {
            "flops_ratio": profile.flops_ratio,
            "ram_mb": profile.ram_mb,
            "bandwidth_mbps": profile.bandwidth_mbps,
            "reputation": profile.reputation,
            "load": current_load,
            "cluster_id": cluster_id,
            "intra_broadcast": True,
        }

        # Broadcast to self and intra-cluster neighbors
        self._protocol.broadcast(node_id, lrh)
        for neighbor_id in intra_neighbors:
            self._protocol.broadcast(neighbor_id, lrh)

    def broadcast_inter_cluster(
        self,
        cluster_head_id: int,
        profile,
        current_load: float = 0.0,
    ) -> None:
        """
        Broadcast LRH to other cluster-heads (inter-cluster gossip).

        Only cluster-heads should call this for inter-cluster coordination.

        Args:
            cluster_head_id: Cluster-head node ID.
            profile: HardwareProfile instance.
            current_load: Current CPU/load estimate (0-1).
        """
        if self._cluster_manager is None:
            return

        cluster_id = self._cluster_manager.get_cluster_id(cluster_head_id)
        if cluster_id is None:
            return

        # Get other cluster-heads
        other_heads = self._cluster_manager.get_inter_cluster_neighbors(
            cluster_head_id, fanout=2
        )

        lrh = {
            "flops_ratio": profile.flops_ratio,
            "ram_mb": profile.ram_mb,
            "bandwidth_mbps": profile.bandwidth_mbps,
            "reputation": profile.reputation,
            "load": current_load,
            "cluster_id": cluster_id,
            "is_cluster_head": True,
            "inter_cluster_source": cluster_head_id,
        }

        # Broadcast to self (own cluster-head entry)
        self._protocol.broadcast(cluster_head_id, lrh)
        # Broadcast to other cluster-heads
        for head_id in other_heads:
            self._protocol.broadcast(head_id, lrh)

    def get_neighbor_info(self, node_id: int, k: Optional[int] = None) -> list[dict]:
        """Return k neighbor LRH dicts."""
        return self._protocol.get_neighbors(node_id, k=k)

    def get_intra_cluster_neighbors(self, node_id: int, k: int = 3) -> List[int]:
        """
        Get k intra-cluster neighbor node_ids.

        Args:
            node_id: Querying node.
            k: Number of neighbors.

        Returns:
            List of intra-cluster neighbor node_ids.
        """
        if self._cluster_manager is None:
            return []
        return self._cluster_manager.get_intra_cluster_neighbors(node_id, fanout=k)

    def get_inter_cluster_heads(self, node_id: int, k: int = 2) -> List[int]:
        """
        Get k cluster-head IDs from OTHER clusters.

        Args:
            node_id: Querying cluster-head node.
            k: Number of other cluster-heads.

        Returns:
            List of other cluster-head node_ids.
        """
        if self._cluster_manager is None:
            return []
        return self._cluster_manager.get_inter_cluster_neighbors(node_id, fanout=k)

    def mean_neighbor_availability(self, node_id: int) -> float:
        """Return mean availability across neighbors."""
        return self._protocol.mean_neighbor_availability(node_id)

    def mean_intra_cluster_availability(self, node_id: int) -> float:
        """
        Return mean availability within cluster (cluster members only).

        Useful for cluster-head to assess cluster health.
        """
        if self._cluster_manager is None:
            return self.mean_neighbor_availability(node_id)

        cluster_id = self._cluster_manager.get_cluster_id(node_id)
        if cluster_id is None:
            return self.mean_neighbor_availability(node_id)

        members = self._cluster_manager.get_cluster_members(cluster_id)
        availabilities = []

        for nid in members:
            if nid == node_id:
                continue
            info = self._protocol._table.get(nid, {})
            avail = 1.0 - info.get("load", 0.0)
            availabilities.append(avail)

        return sum(availabilities) / len(availabilities) if availabilities else 0.0

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
        all_candidates = [
            (nid, info) for nid, info in self._protocol._table.items()
            if nid != node_id or not exclude_self
        ]
        if not all_candidates:
            return None

        best_nid, _ = max(all_candidates, key=lambda x: x[1].get("reputation", 0.0))
        return best_nid

    def get_best_intra_target(self, node_id: int) -> Optional[int]:
        """
        Return intra-cluster neighbor with highest reputation.

        Used by cluster-head to select best cluster member.
        """
        if self._cluster_manager is None:
            return None

        cluster_id = self._cluster_manager.get_cluster_id(node_id)
        if cluster_id is None:
            return None

        members = self._cluster_manager.get_cluster_members(cluster_id)
        best_nid = None
        best_rep = -1

        for nid in members:
            if nid == node_id:
                continue
            info = self._protocol._table.get(nid, {})
            rep = info.get("reputation", 0.0)
            if rep > best_rep:
                best_rep = rep
                best_nid = nid

        return best_nid

    def update_from_chainfsl(self, gossip_protocol: GossipProtocol) -> None:
        """
        Sync from an existing GossipProtocol instance (e.g., from emulator).

        Args:
            gossip_protocol: GossipProtocol instance to sync from.
        """
        self._protocol = gossip_protocol