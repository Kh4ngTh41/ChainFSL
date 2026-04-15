"""
GTM Contribution — ContributionVector and VLI computation.

Implements Eq. 13 from ChainFSL paper:
  v_i = (v_comp, v_data, v_bw, v_rel)

And VLI (Validation-Loss Improvement) for measuring marginal data quality.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict
import torch
import torch.nn as nn
import numpy as np


@dataclass
class ContributionVector:
    """
    Multi-component contribution vector for a node.

    Components:
    - v_comp: Normalized computation (FLOPS / FLOPS_max)
    - v_data: Data contribution (|D_i| * q_i / max_j)
    - v_bw: Bandwidth contribution (b_i / b_max)
    - v_rel: Reliability (1 - failures/rounds)
    """

    node_id: int

    v_comp: float = 0.0  # Computation contribution
    v_data: float = 0.0  # Data quality contribution
    v_bw: float = 0.0    # Bandwidth contribution
    v_rel: float = 0.5   # Reliability contribution

    @property
    def total(self) -> float:
        """Weighted sum of components (equal weights default)."""
        return (self.v_comp + self.v_data + self.v_bw + self.v_rel) / 4.0

    def to_dict(self) -> Dict[str, float]:
        """Serialize to dict."""
        return {
            "node_id": self.node_id,
            "v_comp": self.v_comp,
            "v_data": self.v_data,
            "v_bw": self.v_bw,
            "v_rel": self.v_rel,
            "total": self.total,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ContributionVector":
        return cls(
            node_id=d["node_id"],
            v_comp=d.get("v_comp", 0.0),
            v_data=d.get("v_data", 0.0),
            v_bw=d.get("v_bw", 0.0),
            v_rel=d.get("v_rel", 0.5),
        )


class VLIComputer:
    """
    Validation-Loss Improvement — privacy-preserving proxy for data quality.

    VLI_i = F(w^{t-1}) - F(w^{t-1} + η * grad_i)

    Measures marginal utility of node i's update without revealing raw data.
    Uses uniform (non-staleness-decayed) weighting to ensure fairness for slow nodes.
    """

    def __init__(
        self,
        val_loader,  # Validation DataLoader
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        eta: float = 0.01,
    ):
        """
        Args:
            val_loader: Validation DataLoader for evaluating model quality.
            criterion: Loss function. Defaults to CrossEntropyLoss.
            device: Computation device.
            eta: Learning rate for marginal gradient approximation.
        """
        self.val_loader = val_loader
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eta = eta

    def compute_vli(
        self,
        global_state: Dict[str, torch.Tensor],
        node_update: Dict[str, torch.Tensor],
        model_cls,  # Model class to reconstruct from state
    ) -> float:
        """
        Compute VLI for a single node's update.

        Args:
            global_state: Global model state dict.
            node_update: Node's update dict (state dict delta).
            model_cls: Model class (e.g., SplittableResNet18) to reconstruct.

        Returns:
            VLI value (non-negative).
        """
        # Evaluate loss BEFORE update
        loss_before = self._eval_loss(global_state, model_cls)

        # Apply update with uniform weight (no staleness decay)
        updated_state = self._apply_update(global_state, node_update)

        # Evaluate loss AFTER update
        loss_after = self._eval_loss(updated_state, model_cls)

        return max(0.0, loss_before - loss_after)

    def compute_vli_batch(
        self,
        global_state: Dict[str, torch.Tensor],
        node_updates: List[Dict[str, torch.Tensor]],
        model_cls,
    ) -> Dict[int, float]:
        """
        Compute VLI for multiple node updates.

        Args:
            global_state: Global model state dict.
            node_updates: List of node update dicts.
            model_cls: Model class.

        Returns:
            Dict node_id -> VLI value.
        """
        vli_results = {}
        for node_id, update in enumerate(node_updates):
            vli_results[node_id] = self.compute_vli(global_state, update, model_cls)
        return vli_results

    def _eval_loss(
        self,
        state: Dict[str, torch.Tensor],
        model_cls,
    ) -> float:
        """Evaluate model loss on validation set."""
        model = model_cls()
        model.load_state_dict(state, strict=False)
        model = model.to(self.device)
        model.eval()

        total_loss = 0.0
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = model(x)
                total_loss += self.criterion(output, y).item()

        return total_loss / max(len(self.val_loader), 1)

    def _apply_update(
        self,
        global_state: Dict[str, torch.Tensor],
        node_update: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Apply node update with uniform weight (eta)."""
        updated = {}
        for key in global_state:
            if key in node_update:
                delta = node_update[key].float()
                updated[key] = global_state[key].float() + self.eta * delta
            else:
                updated[key] = global_state[key].float()
        return updated


def compute_contribution_vector(
    node_profile,  # HardwareProfile
    data_size: int,
    max_data_size: int,
    gradient_norm: float,
    max_gradient_norm: float,
    failure_rate: float,  # failures / total_rounds
    max_bw: float = 100.0,
) -> ContributionVector:
    """
    Compute ContributionVector for a node given its profile and behavior.

    Args:
        node_profile: HardwareProfile instance.
        data_size: Number of samples in node's dataset.
        max_data_size: Max data size across all nodes.
        gradient_norm: Observed gradient norm.
        max_gradient_norm: Max gradient norm across all nodes.
        failure_rate: Fraction of rounds where node failed.
        max_bw: Max bandwidth in network (for normalization).

    Returns:
        ContributionVector instance.
    """
    return ContributionVector(
        node_id=node_profile.node_id,
        v_comp=node_profile.flops_ratio,  # Normalized to [0, 1] via tier
        v_data=(data_size / max(max_data_size, 1)),
        v_bw=min(node_profile.bandwidth_mbps / max(max_bw, 1), 1.0),
        v_rel=max(0.0, 1.0 - failure_rate),
    )


def aggregate_contributions(
    contributions: List[ContributionVector],
) -> Dict[str, float]:
    """
    Aggregate per-node contributions into network-level stats.

    Args:
        contributions: List of ContributionVector.

    Returns:
        Dict with aggregated stats (mean, std, min, max per component).
    """
    if not contributions:
        return {}

    component_keys = ["v_comp", "v_data", "v_bw", "v_rel", "total"]
    result = {}

    for key in component_keys:
        values = [getattr(c, key) for c in contributions]
        result[f"{key}_mean"] = np.mean(values)
        result[f"{key}_std"] = np.std(values)
        result[f"{key}_min"] = np.min(values)
        result[f"{key}_max"] = np.max(values)

    return result