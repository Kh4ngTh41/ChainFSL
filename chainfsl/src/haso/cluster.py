"""
MA-HASO Cluster Management.

Implements hierarchical clustering: k clusters, a nodes each.
Cluster formation + resource-based cluster-head election.

Per ChainFSL_Implementation_Plan.md:
- a.k ~ N (a nodes per cluster, k clusters)
- cluster_size = configurable (default: 5)
- k = N / cluster_size (auto-calculated)
- Cluster-head election: resource-based (highest CPU+RAM+BW score)
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ClusterInfo:
    """Metadata for a cluster."""
    cluster_id: int
    node_ids: List[int]
    cluster_head_id: int
    cluster_size: int


class ClusterManager:
    """
    Manages hierarchical cluster formation and cluster-head election.

    Architecture:
        N nodes → k clusters → a = N/k nodes per cluster

    Each cluster has:
    - 1 cluster-head (runs ClusterHASOAgent PPO)
    - a-1 regular nodes (train only, no PPO)
    """

    def __init__(self):
        self.clusters: Dict[int, List[int]] = {}
        self.cluster_heads: Dict[int, int] = {}
        self.node_to_cluster: Dict[int, int] = {}
        self.n_clusters: int = 0
        self.cluster_size: int = 0

    def form_clusters(
        self,
        n_nodes: int,
        cluster_size: int = 5,
        node_profiles: Optional[List] = None,
    ) -> Dict[int, List[int]]:
        """
        Form clusters with equal number of nodes per cluster.

        Args:
            n_nodes: Total number of nodes in federation.
            cluster_size: Number of nodes per cluster (a).
            node_profiles: List of HardwareProfile for each node (for election).

        Returns:
            Dict mapping cluster_id → [node_ids]

        Raises:
            ValueError: If n_nodes not divisible by cluster_size.
        """
        if n_nodes % cluster_size != 0:
            raise ValueError(
                f"n_nodes={n_nodes} not divisible by cluster_size={cluster_size}. "
                f"Need n_nodes = k * cluster_size for integer k."
            )

        self.cluster_size = cluster_size
        self.n_clusters = n_nodes // cluster_size
        self.clusters = {}

        # Distribute nodes round-robin to clusters
        # Cluster 0: [0, k, 2k, ...]
        # Cluster 1: [1, k+1, 2k+1, ...]
        # etc.
        for c in range(self.n_clusters):
            self.clusters[c] = [
                c + (i * self.n_clusters)
                for i in range(cluster_size)
                if c + (i * self.n_clusters) < n_nodes
            ]

        # Elect cluster-head for each cluster
        self._elect_cluster_heads(node_profiles)

        return self.clusters

    def _elect_cluster_heads(self, node_profiles: Optional[List]) -> None:
        """
        Elect cluster-head for each cluster based on resource score.

        Args:
            node_profiles: List of HardwareProfile (or None).
                           If None, use node_id as tiebreaker (lowest = head).
        """
        self.cluster_heads = {}

        for cluster_id, node_ids in self.clusters.items():
            if node_profiles is None:
                # Fallback: lowest node_id becomes head
                head_id = min(node_ids)
            else:
                # Resource-based: highest combined CPU+RAM+BW score
                head_id = max(
                    node_ids,
                    key=lambda nid: self._resource_score(node_profiles[nid])
                )
            self.cluster_heads[cluster_id] = head_id

        # Build reverse mapping: node_id → cluster_id
        self.node_to_cluster = {}
        for cluster_id, node_ids in self.clusters.items():
            for nid in node_ids:
                self.node_to_cluster[nid] = cluster_id

    def _resource_score(self, profile) -> float:
        """
        Compute resource score for cluster-head election.

        Score = w1*cpu_norm + w2*ram_norm + w3*bw_norm

        Weights: CPU=0.4, RAM=0.3, BW=0.3 (can be configured)
        """
        cpu = getattr(profile, 'flops_ratio', 0.0)
        ram = getattr(profile, 'ram_mb', 0.0) / 32768.0  # normalize to 32GB
        bw = getattr(profile, 'bandwidth_mbps', 0.0) / 1000.0  # normalize to 1Gbps

        return 0.4 * cpu + 0.3 * ram + 0.3 * bw

    def get_cluster_id(self, node_id: int) -> Optional[int]:
        """Return cluster_id for a node."""
        return self.node_to_cluster.get(node_id)

    def get_cluster_head(self, cluster_id: int) -> Optional[int]:
        """Return cluster-head node_id for a cluster."""
        return self.cluster_heads.get(cluster_id)

    def get_cluster_members(self, cluster_id: int) -> List[int]:
        """Return node_ids in a cluster."""
        return self.clusters.get(cluster_id, [])

    def get_non_head_nodes(self, cluster_id: int) -> List[int]:
        """Return all nodes in cluster except the head."""
        head = self.cluster_heads.get(cluster_id)
        return [n for n in self.clusters.get(cluster_id, []) if n != head]

    def is_cluster_head(self, node_id: int) -> bool:
        """Check if a node is a cluster head."""
        return node_id in self.cluster_heads.values()

    def get_all_cluster_heads(self) -> List[int]:
        """Return list of all cluster-head node_ids."""
        return list(self.cluster_heads.values())

    def get_intra_cluster_neighbors(
        self,
        node_id: int,
        fanout: int = 3,
        exclude_head: bool = False,
    ) -> List[int]:
        """
        Get neighbors within the same cluster.

        Args:
            node_id: Querying node.
            fanout: Number of neighbors to return.
            exclude_head: If True, exclude cluster-head from results.

        Returns:
            List of neighbor node_ids (within same cluster).
        """
        cluster_id = self.get_cluster_id(node_id)
        if cluster_id is None:
            return []

        members = self.get_cluster_members(cluster_id)
        if exclude_head:
            head = self.get_cluster_head(cluster_id)
            members = [m for m in members if m != head]

        # Exclude self
        members = [m for m in members if m != node_id]

        # Return up to fanout
        return members[:fanout]

    def get_inter_cluster_neighbors(
        self,
        node_id: int,
        fanout: int = 2,
    ) -> List[int]:
        """
        Get cluster-heads of OTHER clusters (for inter-cluster gossip).

        Args:
            node_id: Querying node (presumably a cluster-head).
            fanout: Number of other cluster-heads to return.

        Returns:
            List of other cluster-head node_ids.
        """
        all_heads = self.get_all_cluster_heads()
        my_cluster = self.get_cluster_id(node_id)
        my_head = self.get_cluster_head(my_cluster) if my_cluster is not None else None

        # Exclude self (own cluster-head)
        other_heads = [h for h in all_heads if h != my_head]
        return other_heads[:fanout]

    def get_cluster_info(self, cluster_id: int) -> Optional[ClusterInfo]:
        """Get full cluster metadata."""
        if cluster_id not in self.clusters:
            return None
        return ClusterInfo(
            cluster_id=cluster_id,
            node_ids=self.clusters[cluster_id],
            cluster_head_id=self.cluster_heads[cluster_id],
            cluster_size=self.cluster_size,
        )

    def summary(self) -> str:
        """Return human-readable cluster summary."""
        lines = [f"Clusters: k={self.n_clusters}, a={self.cluster_size}"]
        for cid, nids in self.clusters.items():
            head = self.cluster_heads[cid]
            lines.append(
                f"  Cluster {cid}: nodes={nids}, head=node{head}"
            )
        return "\n".join(lines)