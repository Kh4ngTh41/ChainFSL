"""
Network emulator for modeling P2P bandwidth and latency.

Implements Markov-like bandwidth fluctuation as described in
the ChainFSL paper Section 6.1.3.
"""

import random
import asyncio
from typing import Dict, Optional
from dataclasses import dataclass

from .node_profile import HardwareProfile


@dataclass
class NetworkEmulator:
    """
    Emulates P2P network characteristics: bandwidth, latency, packet loss.

    Uses variance-based fluctuation to simulate real-world network conditions.

    Attributes:
        variance: Bandwidth fluctuation range (±fraction, e.g., 0.3 = ±30%).
        min_bandwidth_factor: Minimum effective bandwidth as fraction of nominal.
        latency_ms: Base network latency in milliseconds.
    """

    variance: float = 0.3
    min_bandwidth_factor: float = 0.1
    latency_ms: float = 50.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.variance <= 1.0):
            raise ValueError(f"variance must be in [0,1], got {self.variance}")

    def effective_bandwidth(self, nominal_mbps: float, rng: Optional[random.Random] = None) -> float:
        """
        Compute effective bandwidth with Markov-like fluctuation.

        Args:
            nominal_mbps: Nominal bandwidth in Mbps.
            rng: Random instance for reproducibility.

        Returns:
            Effective bandwidth in Mbps.
        """
        if rng is None:
            rng = random.Random()

        factor = 1.0 + rng.uniform(-self.variance, self.variance)
        effective = nominal_mbps * factor
        return max(nominal_mbps * self.min_bandwidth_factor, effective)

    def comm_time(self, src: HardwareProfile, dst: HardwareProfile, data_bytes: float) -> float:
        """
        Estimate round-trip communication time between two nodes.

        Uses the bottleneck bandwidth (min of both nodes' effective bandwidth).

        Args:
            src: Source node profile.
            dst: Destination node profile.
            data_bytes: Data size in bytes.

        Returns:
            Estimated time in seconds.
        """
        bw = min(
            self.effective_bandwidth(src.bandwidth_mbps),
            self.effective_bandwidth(dst.bandwidth_mbps)
        )
        bandwidth_bytes_per_sec = bw * 1e6 / 8
        transmission_time = data_bytes / bandwidth_bytes_per_sec
        latency_time = self.latency_ms / 1000.0
        return transmission_time + latency_time

    async def async_send(
        self,
        src: HardwareProfile,
        dst: HardwareProfile,
        data_bytes: float,
    ) -> float:
        """
        Async version of comm_time with actual sleep delay.

        Use this for co-routine based simulations.

        Args:
            src: Source node profile.
            dst: Destination node profile.
            data_bytes: Data size in bytes.

        Returns:
            Actual delay in seconds.
        """
        delay = self.comm_time(src, dst, data_bytes)
        await asyncio.sleep(delay)
        return delay

    def comm_overhead_ratio(self, data_bytes: float) -> float:
        """
        Compute communication overhead as fraction of bandwidth.

        Used for TVE proof size estimation.

        Args:
            data_bytes: Data size in bytes.

        Returns:
            Ratio of overhead to data (0-1).
        """
        # Estimate: 32 bytes VRF proof per 1MB data = 0.0032%
        proof_overhead_bytes = 32
        return min(proof_overhead_bytes / max(data_bytes, 1), 1.0)


class GossipProtocol:
    """
    Gossip-based peer state sharing using shared memory.

    Each node broadcasts a Lightweight Resource Heartbeat (LRH):
    - flops_ratio, ram_available, bandwidth, reputation, load

    Other nodes read from shared dict to assess neighbor availability.
    """

    def __init__(self, shared_table: Optional[dict] = None, fanout: int = 3):
        """
        Args:
            shared_table: Dict-like object (e.g., Manager().dict()).
            fanout: Number of neighbors to sample per gossip round.
        """
        self._table = shared_table if shared_table is not None else {}
        self.fanout = fanout

    def broadcast(self, node_id: int, lrh: dict) -> None:
        """
        Publish LRH for a node.

        Args:
            node_id: Node ID.
            lrh: Heartbeat dict with keys: flops_ratio, ram_mb, bandwidth_mbps,
                 reputation, load, timestamp.
        """
        self._table[node_id] = {**lrh, "timestamp": self._current_time()}

    def get_neighbors(self, node_id: int, k: Optional[int] = None) -> list[dict]:
        """
        Return k neighbor LRHs (excluding self), sorted by reputation.

        Args:
            node_id: Querying node ID.
            k: Number of neighbors. Defaults to fanout.

        Returns:
            List of LRH dicts.
        """
        k = k if k is not None else self.fanout
        candidates = [
            (nid, info)
            for nid, info in self._table.items()
            if nid != node_id
        ]
        candidates.sort(key=lambda x: x[1].get("reputation", 0), reverse=True)
        return [info for _, info in candidates[:k]]

    def mean_neighbor_availability(self, node_id: int) -> float:
        """
        Compute mean flops_ratio of neighbors.

        Used as state observation for MA-HASO.

        Args:
            node_id: Querying node ID.

        Returns:
            Mean availability in [0, 1].
        """
        neighbors = self.get_neighbors(node_id, k=self.fanout)
        if not neighbors:
            return 0.5
        return sum(n.get("flops_ratio", 0.5) for n in neighbors) / len(neighbors)

    def _current_time(self) -> float:
        """Wall-clock time (mockable for testing)."""
        import time
        return time.time()