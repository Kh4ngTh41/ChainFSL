"""
Split model wrappers and SmashData serialization.

Provides the SmashData dataclass for client-server communication,
and re-exports ClientModel/ServerModel for convenience.
"""

from dataclasses import dataclass
from typing import Optional
import torch

from .models import ClientModel, ServerModel, SplittableResNet18

__all__ = ["SmashData", "ClientModel", "ServerModel", "SplittableResNet18"]


@dataclass
class SmashData:
    """
    Serialization unit for client→server smashed data transmission.

    Attributes:
        node_id: Source node ID.
        tensor: Activation tensor at cut layer.
        labels: Ground truth labels (for server-side loss computation).
        round_id: Global round number (for staleness tracking).
        cut_layer: Cut layer index used for this transmission.
        client_state_hash: SHA-256 of client weights at send time (for verification).
        activation_hash: Hash of activation tensor (for TVE proof).
    """

    node_id: int
    tensor: torch.Tensor
    labels: torch.Tensor
    round_id: int
    cut_layer: int
    client_state_hash: Optional[bytes] = None
    activation_hash: Optional[bytes] = None

    def to_device(self, device: torch.device) -> "SmashData":
        """Move tensor and labels to device."""
        return SmashData(
            node_id=self.node_id,
            tensor=self.tensor.to(device),
            labels=self.labels.to(device),
            round_id=self.round_id,
            cut_layer=self.cut_layer,
            client_state_hash=self.client_state_hash,
            activation_hash=self.activation_hash,
        )

    def size_bytes(self) -> int:
        """Approximate size in bytes."""
        return self.tensor.element_size() * self.tensor.numel()

    def with_hashes(self, state_hash: bytes, act_hash: bytes) -> "SmashData":
        """Return a copy with hash fields populated."""
        return SmashData(
            node_id=self.node_id,
            tensor=self.tensor,
            labels=self.labels,
            round_id=self.round_id,
            cut_layer=self.cut_layer,
            client_state_hash=state_hash,
            activation_hash=act_hash,
        )