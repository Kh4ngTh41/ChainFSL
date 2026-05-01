"""
ComputeNode model for MA-HASO DRL orchestration.

Represents a heterogeneous IoT/Edge compute node with queue management,
task processing, and bandwidth fluctuation capabilities for the
federated split learning orchestration.
"""

from dataclasses import dataclass, field
from typing import List
import math
import random

from ..emulator.node_profile import TIER_CONFIGS as PROFILE_TIER_CONFIGS

# Reference compute power: 1.0 GFLOPS (Tier 1 baseline)
REF_GFLOPS: float = 1.0

# Tier configurations for MA-HASO ComputeNode
TIER_CONFIGS: dict[int, dict] = {
    1: dict(flops_ratio=1.0, ram_mb=8192, bandwidth_mbps=100.0),
    2: dict(flops_ratio=0.3, ram_mb=4096, bandwidth_mbps=50.0),
    3: dict(flops_ratio=0.05, ram_mb=512, bandwidth_mbps=10.0),
    4: dict(flops_ratio=0.005, ram_mb=200, bandwidth_mbps=1.0),
}


@dataclass
class ComputeNode:
    """
    Compute node for MA-HASO orchestration.

    Models a heterogeneous IoT/Edge device with queue-based task scheduling,
    energy management, and network bandwidth fluctuation.

    Attributes:
        node_id: Unique identifier for this node.
        tier: Hardware tier (1-4) determining resource limits.
        flops_ratio: Compute power relative to Tier-1 (1.0 = GPU-class).
        max_memory_mb: Memory limit in MB.
        bandwidth_mbps: Network bandwidth in Mbps.
        current_queue: Number of tasks currently queued.
        processing: Whether node is currently processing a task.
        accepted_layers: List of cut layers (1-4) this node can handle.
        energy_remaining: Remaining energy budget in mAh.
        energy_budget: Total energy budget in mAh.
    """

    node_id: int
    tier: int
    flops_ratio: float
    max_memory_mb: int
    bandwidth_mbps: float
    current_queue: int
    processing: bool
    accepted_layers: List[int]
    energy_remaining: float = 1000.0
    energy_budget: float = 1000.0

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if not 1 <= self.tier <= 4:
            raise ValueError(f"Tier must be 1-4, got {self.tier}")
        if self.flops_ratio <= 0:
            raise ValueError(f"flops_ratio must be positive, got {self.flops_ratio}")
        if self.max_memory_mb <= 0:
            raise ValueError(f"max_memory_mb must be positive, got {self.max_memory_mb}")
        if self.bandwidth_mbps <= 0:
            raise ValueError(f"bandwidth_mbps must be positive, got {self.bandwidth_mbps}")
        if self.current_queue < 0:
            raise ValueError(f"current_queue must be non-negative, got {self.current_queue}")

    @property
    def compute_power_gflops(self) -> float:
        """Actual compute power in GFLOPS."""
        return self.flops_ratio * REF_GFLOPS

    @property
    def memory_gb(self) -> float:
        """RAM in gigabytes."""
        return self.max_memory_mb / 1024.0

    @property
    def queue_utilization(self) -> float:
        """Fraction of max queue capacity used (assuming max 10)."""
        return self.current_queue / 10.0

    def estimated_time(self, cut_layer: int, batch_size: int) -> float:
        """
        Estimate total processing time including queue wait.

        Args:
            cut_layer: Target cut layer index (1-4).
            batch_size: Training batch size.

        Returns:
            Estimated time in seconds.
        """
        # Base FLOPs for forward pass (normalized, 1.0 = 1 GFLOPS)
        # Approximate ResNet-18 FLOPs scaled by cut layer
        base_flops = {
            1: 0.15,
            2: 0.30,
            3: 0.55,
            4: 1.10,
        }.get(cut_layer, 0.5)

        # Adjust for batch size (roughly linear with batch for forward pass)
        batch_multiplier = math.sqrt(batch_size / 32.0)
        effective_flops = base_flops * batch_multiplier

        # Compute time based on this node's compute power
        compute_time = effective_flops / self.compute_power_gflops

        # Queue wait time (assuming 0.1s per queued task)
        queue_wait = self.current_queue * 0.1

        # Communication overhead (data transfer time)
        # SmashData size roughly proportional to batch_size * cut_layer
        smash_size_mb = batch_size * cut_layer * 0.5  # ~0.5MB per layer per sample
        comm_time = (smash_size_mb * 8) / (self.bandwidth_mbps)  # seconds

        return compute_time + queue_wait + comm_time

    def can_process(self, cut_layer: int) -> bool:
        """
        Check if this node can handle the given cut layer.

        Args:
            cut_layer: Target cut layer index (1-4).

        Returns:
            True if node has capacity and accepts this layer.
        """
        if self.processing and self.current_queue > 0:
            return False
        if cut_layer not in self.accepted_layers:
            return False
        # Node should not be overloaded (queue > 5)
        if self.current_queue > 5:
            return False
        return True

    def add_task(self, cut_layer: int) -> bool:
        """
        Add a task to the node's queue.

        Args:
            cut_layer: Layer to process.

        Returns:
            True if task was added, False if rejected.
        """
        if cut_layer not in self.accepted_layers:
            return False
        if self.current_queue >= 10:  # Max queue size
            return False
        self.current_queue += 1
        return True

    def process_task(self) -> int:
        """
        Pop and process the next task in queue.

        Returns:
            The cut_layer being processed, or -1 if queue empty.
        """
        if self.current_queue <= 0:
            return -1
        self.current_queue -= 1
        self.processing = True
        # Simulate processing - caller should call complete_task() after
        return self.accepted_layers[0] if self.accepted_layers else -1

    def complete_task(self) -> None:
        """Mark current task as complete and free the node."""
        self.processing = False

    def update_bandwidth(self, variance: float = 0.1) -> float:
        """
        Simulate bandwidth fluctuation based on network conditions.

        Args:
            variance: Expected variance ratio (0.1 = 10% fluctuation).

        Returns:
            Updated bandwidth in Mbps.
        """
        # Get base bandwidth from tier config
        base_bw = TIER_CONFIGS[self.tier]["bandwidth_mbps"]

        # Gaussian fluctuation around base bandwidth
        fluctuation = random.gauss(0, variance)
        new_bandwidth = base_bw * (1.0 + fluctuation)

        # Clamp to reasonable bounds [10% of base, 150% of base]
        new_bandwidth = max(base_bw * 0.1, min(base_bw * 1.5, new_bandwidth))

        self.bandwidth_mbps = new_bandwidth
        return new_bandwidth

    def energy_consume(self, amount_mah: float) -> bool:
        """
        Deduct energy from budget.

        Args:
            amount_mah: Energy consumed in mAh.

        Returns:
            True if budget sufficient, False if depleted.
        """
        self.energy_remaining = max(0.0, self.energy_remaining - amount_mah)
        return self.energy_remaining > 0

    def __repr__(self) -> str:
        return (
            f"ComputeNode(id={self.node_id}, tier={self.tier}, "
            f"queue={self.current_queue}, processing={self.processing}, "
            f"bw={self.bandwidth_mbps:.1f}Mbps, energy={self.energy_remaining:.0f}mAh)"
        )


def create_compute_node(node_id: int, tier: int, **overrides) -> ComputeNode:
    """
    Factory function to create a ComputeNode with tier defaults.

    Args:
        node_id: Unique node ID.
        tier: Tier level (1-4).
        **overrides: Any field to override defaults.

    Returns:
        ComputeNode instance.
    """
    if tier not in TIER_CONFIGS:
        raise ValueError(f"Unknown tier {tier}")

    cfg = TIER_CONFIGS[tier].copy()
    cfg.update(overrides)

    # Map ram_mb -> max_memory_mb for ComputeNode compatibility
    if "ram_mb" in cfg:
        cfg["max_memory_mb"] = cfg.pop("ram_mb")

    # Set defaults for required fields
    cfg.setdefault("current_queue", 0)
    cfg.setdefault("processing", False)
    cfg.setdefault("accepted_layers", list(range(1, tier + 1)))

    cfg["node_id"] = node_id
    cfg["tier"] = tier

    return ComputeNode(**cfg)