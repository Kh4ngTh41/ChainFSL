"""
Hardware profile modeling for heterogeneous IoT/Edge devices.

This module defines the HardwareProfile dataclass that represents
individual device capabilities in the ChainFSL federation.
"""

from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class HardwareProfile:
    """
    Hardware profile for a single IoT/Edge node.

    Models compute power, memory, bandwidth, and energy characteristics.
    Enforces tier-based constraints for Split Federated Learning.

    Attributes:
        node_id: Unique identifier for this node.
        tier: Hardware tier (1-4) determining resource limits.
        flops_ratio: Compute power relative to Tier 1 (1.0 = GPU-class).
        max_threads: Maximum parallelism threads.
        ram_mb: Available RAM in megabytes.
        bandwidth_mbps: Network bandwidth in Mbps.
        energy_budget: Total energy budget (mAh for battery).
        reputation: Trust score [0,1], updated by GTM.
        stake: Token deposit for TVE committee participation.
    """

    node_id: int
    tier: int
    flops_ratio: float
    max_threads: int
    ram_mb: int
    bandwidth_mbps: float
    energy_budget: float = 1000.0
    energy_remaining: float = field(init=False)
    reputation: float = 0.5
    stake: float = 10.0

    # --- Derived quantities ---
    # Reference compute power: 1.0 GFLOPS (Tier 1 baseline)
    _REF_GFLOPS: float = 1.0

    def __post_init__(self) -> None:
        """Validate fields and initialize derived state."""
        if not 1 <= self.tier <= 4:
            raise ValueError(f"Tier must be 1-4, got {self.tier}")
        if self.flops_ratio <= 0:
            raise ValueError(f"flops_ratio must be positive, got {self.flops_ratio}")
        if self.ram_mb <= 0:
            raise ValueError(f"ram_mb must be positive, got {self.ram_mb}")
        if self.bandwidth_mbps <= 0:
            raise ValueError(f"bandwidth_mbps must be positive, got {self.bandwidth_mbps}")

        self.energy_remaining = self.energy_budget

    @property
    def compute_power_gflops(self) -> float:
        """Actual compute power in GFLOPS."""
        return self.flops_ratio * self._REF_GFLOPS

    @property
    def memory_gb(self) -> float:
        """RAM in gigabytes."""
        return self.ram_mb / 1024.0

    def compute_time(self, base_flops: float) -> float:
        """
        Estimate compute time in seconds.

        Args:
            base_flops: Normalized FLOP count (base=1.0e9 = 1 GFLOPS).

        Returns:
            Estimated time in seconds.
        """
        effective_flops = base_flops * self.flops_ratio
        return effective_flops / (self._REF_GFLOPS * 1e9)

    def comm_time(self, data_size_bytes: float) -> float:
        """
        Estimate communication time in seconds.

        Args:
            data_size_bytes: Data size to transmit.

        Returns:
            Estimated time in seconds.
        """
        bandwidth_bytes_per_sec = self.bandwidth_mbps * 1e6 / 8
        return data_size_bytes / bandwidth_bytes_per_sec

    def can_fit_cut_layer(self, cut_layer: int, model_memory_mb: dict) -> bool:
        """
        Check if a cut layer fits in this node's memory.

        Args:
            cut_layer: Target cut layer index.
            model_memory_mb: Mapping of cut_layer -> required memory (MB).

        Returns:
            True if required memory <= node RAM.
        """
        required = model_memory_mb.get(cut_layer, float("inf"))
        return required <= self.ram_mb

    def energy_consumption(
        self, compute_time_sec: float, power_watts: float = 5.0
    ) -> float:
        """
        Estimate energy consumed for a computation task.

        Args:
            compute_time_sec: Duration of computation in seconds.
            power_watts: Average power draw in watts.

        Returns:
            Energy consumed in mAh, or None if no battery.
        """
        if self.energy_budget is None:
            return None

        # Standard mobile voltage
        voltage = 3.7
        current_ma = (power_watts / voltage) * 1000
        return (current_ma * compute_time_sec) / 3600

    def consume_energy(self, amount_mah: float) -> bool:
        """
        Deduct energy from budget.

        Args:
            amount_mah: Energy consumed in mAh.

        Returns:
            True if budget sufficient, False if depleted.
        """
        if self.energy_remaining is None:
            return True
        self.energy_remaining = max(0.0, self.energy_remaining - amount_mah)
        return self.energy_remaining > 0

    def update_reputation(self, delta: float) -> None:
        """
        Update reputation score via EMA.

        Args:
            delta: Reputation change (positive or negative).
        """
        beta = 0.9
        self.reputation = beta * self.reputation + (1 - beta) * delta
        self.reputation = max(0.0, min(1.0, self.reputation))


# ---------------------------------------------------------------------------
# Tier Configurations
# ---------------------------------------------------------------------------

TIER_CONFIGS: dict[int, dict] = {
    # Tier 1: GPU-enabled edge device (high compute, high memory)
    1: dict(
        flops_ratio=1.0,
        max_threads=8,
        ram_mb=8192,
        bandwidth_mbps=100.0,
        energy_budget=5000.0,
    ),
    # Tier 2: CPU-only mid-range device
    2: dict(
        flops_ratio=0.3,
        max_threads=2,
        ram_mb=4096,
        bandwidth_mbps=50.0,
        energy_budget=3000.0,
    ),
    # Tier 3: Constrained IoT device
    3: dict(
        flops_ratio=0.05,
        max_threads=1,
        ram_mb=512,
        bandwidth_mbps=10.0,
        energy_budget=1000.0,
    ),
    # Tier 4: Minimal resource device
    4: dict(
        flops_ratio=0.005,
        max_threads=1,
        ram_mb=200,
        bandwidth_mbps=1.0,
        energy_budget=500.0,
    ),
}

# Memory requirements for ResNet-18 split points (MB, batch=32)
# Maps cut_layer -> client-side memory requirement
RESNET18_MEMORY_MAP: dict[int, float] = {
    1: 150.0,   # conv1 + layer1
    2: 300.0,   # + layer2
    3: 500.0,   # + layer3
    4: 700.0,   # + layer4 (near-full model)
}


def create_profile(node_id: int, tier: int, **overrides) -> HardwareProfile:
    """
    Factory function to create a HardwareProfile with tier defaults.

    Args:
        node_id: Unique node ID.
        tier: Tier level (1-4).
        **overrides: Any field to override defaults.

    Returns:
        HardwareProfile instance.
    """
    if tier not in TIER_CONFIGS:
        raise ValueError(f"Unknown tier {tier}")
    cfg = TIER_CONFIGS[tier].copy()
    cfg.update(overrides)
    cfg["node_id"] = node_id
    cfg["tier"] = tier
    return HardwareProfile(**cfg)