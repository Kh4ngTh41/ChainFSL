"""
Asynchronous aggregator with staleness-decayed weighted averaging.

Implements Eq. 3 (layer-wise partial aggregation) and Eq. 9
(staleness-decayed async aggregation) from the ChainFSL paper.
"""

import copy
import torch
from collections import defaultdict
from typing import List, Dict, Optional


class AsyncAggregator:
    """
    Asynchronous aggregator supporting staleness-decayed weighted averaging.

    Implements:
    - Layer-wise partial aggregation (Eq. 3): independent aggregation per layer
    - Staleness-decayed weighting (Eq. 9): alpha_i^(t) = (|D_i|/|D|) * rho^tau_i
    - O(1) update complexity per node (no full model averaging)
    """

    def __init__(self, global_state: Dict[str, torch.Tensor], rho: float = 0.9):
        """
        Args:
            global_state: State dict of the global model.
            rho: Staleness decay constant (0.9 per paper).
        """
        self.global_state = copy.deepcopy(global_state)
        self.rho = rho
        self.current_round = 0

    def aggregate(self, updates: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Aggregate pending updates with staleness-decayed weights.

        Args:
            updates: List of update dicts, each containing:
                - 'node_id': int
                - 'cut_layer': int
                - 'client_state': dict (state_dict of client-side params)
                - 'server_state': dict (state_dict of server-side params)
                - 'data_size': int (|D_i|)
                - 'staleness': int (tau_i, version lag)

        Returns:
            Updated global state dict.
        """
        if not updates:
            return copy.deepcopy(self.global_state)

        total_data = sum(u.get("data_size", 1) for u in updates)

        # Layer-wise accumulation
        layer_updates: Dict[str, torch.Tensor] = {}
        layer_weights: Dict[str, float] = defaultdict(float)

        for update in updates:
            staleness = update.get("staleness", 0)
            data_frac = update.get("data_size", 1) / total_data
            alpha = data_frac * (self.rho ** staleness)

            # Client-side layers (layer index <= cut_layer)
            client_state = update.get("client_state", {})
            for key, param in client_state.items():
                if key not in layer_updates:
                    layer_updates[key] = torch.zeros_like(
                        self.global_state[key], dtype=torch.float32
                    )
                layer_updates[key] += alpha * (param.float() - self.global_state[key].float())
                layer_weights[key] += alpha

            # Server-side layers
            server_state = update.get("server_state", {})
            for key, param in server_state.items():
                if key not in layer_updates:
                    layer_updates[key] = torch.zeros_like(
                        self.global_state[key], dtype=torch.float32
                    )
                layer_updates[key] += alpha * (param.float() - self.global_state[key].float())
                layer_weights[key] += alpha

        # Apply normalized updates
        new_state = copy.deepcopy(self.global_state)
        for key in new_state:
            if key in layer_updates and layer_weights[key] > 0:
                new_state[key] = (
                    self.global_state[key].float() + layer_updates[key] / layer_weights[key]
                ).to(self.global_state[key].dtype)

        self.global_state = new_state
        self.current_round += 1
        return copy.deepcopy(new_state)

    def get_staleness_weights(
        self, node_ids: List[int], staleness_map: Dict[int, int]
    ) -> Dict[int, float]:
        """
        Compute staleness-decayed weights for a set of nodes.

        Args:
            node_ids: List of node IDs.
            staleness_map: Mapping node_id -> staleness (tau).

        Returns:
            Mapping node_id -> weight (alpha_i).
        """
        total_staleness = sum(self.rho ** staleness_map.get(nid, 0) for nid in node_ids)
        if total_staleness == 0:
            return {nid: 0.0 for nid in node_ids}
        return {
            nid: (self.rho ** staleness_map.get(nid, 0)) / total_staleness
            for nid in node_ids
        }

    def reset(self) -> None:
        """Reset aggregator state."""
        self.current_round = 0

    @property
    def round(self) -> int:
        """Current aggregation round."""
        return self.current_round


class FedAvgAggregator:
    """
    Standard FedAvg aggregator (no staleness handling).

    Useful as a baseline comparison.
    """

    def __init__(self, global_state: Dict[str, torch.Tensor]):
        self.global_state = copy.deepcopy(global_state)
        self.current_round = 0

    def aggregate(self, updates: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Standard FedAvg: weighted average by data size.

        Args:
            updates: List of update dicts with 'state' and 'data_size'.

        Returns:
            Updated global state dict.
        """
        if not updates:
            return copy.deepcopy(self.global_state)

        total_data = sum(u.get("data_size", 1) for u in updates)
        new_state = copy.deepcopy(self.global_state)

        # Accumulate weighted sum
        accum = defaultdict(lambda: torch.zeros_like(self.global_state[k]))
        weight_sum = defaultdict(float)

        for update in updates:
            weight = update.get("data_size", 1) / total_data
            state = update.get("state", update.get("client_state", {}))

            for key, param in state.items():
                accum[key] += weight * param.float()
                weight_sum[key] += weight

        # Normalize and apply
        for key in new_state:
            if weight_sum[key] > 0:
                new_state[key] = accum[key].to(new_state[key].dtype)

        self.global_state = new_state
        self.current_round += 1
        return copy.deepcopy(new_state)