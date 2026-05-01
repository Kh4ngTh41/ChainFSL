"""FederatedRLCoordinator — Policy weight sharing between HASO agents."""
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PolicyWeights:
    node_id: int
    weights: np.ndarray
    round_number: int
    shapley_value: float


class FederatedRLCoordinator:
    def __init__(
        self,
        node_id: int,
        cluster_members: List[int],
        cluster_head: bool = False,
        gossip=None,
        share_every_n_rounds: int = 5,
    ):
        self.node_id = node_id
        self.cluster_members = cluster_members
        self.cluster_head = cluster_head
        self.gossip = gossip
        self.share_every_n_rounds = share_every_n_rounds
        self._intra_weights: Dict[int, PolicyWeights] = {}
        self._inter_weights: Dict[int, PolicyWeights] = {}
        self._aggregated_weights: Optional[np.ndarray] = None
        self._last_share_round = -1

    def should_share(self, current_round: int) -> bool:
        if self._last_share_round < 0:
            # First share always allowed
            self._last_share_round = current_round
            return True
        if current_round - self._last_share_round >= self.share_every_n_rounds:
            self._last_share_round = current_round
            return True
        return False

    def federated_aggregate(
        self, agent_weights: Dict[int, np.ndarray], shapley_values: Dict[int, float]
    ) -> np.ndarray:
        total_shapley = sum(shapley_values.values())
        if total_shapley == 0:
            total_shapley = len(agent_weights)
        aggregated = None
        for node_id, weights in agent_weights.items():
            phi = shapley_values.get(node_id, 1.0)
            w = phi / total_shapley
            if aggregated is None:
                aggregated = w * weights
            else:
                aggregated = aggregated + w * weights
        return aggregated if aggregated is not None else np.array([])

    def share_intra_cluster(
        self, local_weights: np.ndarray, shapley_value: float, current_round: int
    ) -> None:
        self._last_share_round = current_round
        if self.cluster_head:
            pw = PolicyWeights(
                node_id=self.node_id,
                weights=local_weights,
                round_number=current_round,
                shapley_value=shapley_value,
            )
            self._intra_weights[self.node_id] = pw
        else:
            if self.gossip is not None and hasattr(self.gossip, "_protocol"):
                cluster_head = self.cluster_members[0] if self.cluster_members else None
                if cluster_head is not None:
                    if cluster_head not in self.gossip._protocol._table:
                        self.gossip._protocol._table[cluster_head] = {}
                    self.gossip._protocol._table[cluster_head][
                        f"intra_weight_{self.node_id}"
                    ] = {
                        "weights": local_weights,
                        "shapley": shapley_value,
                        "round": current_round,
                    }

    def collect_intra_cluster_weights(self) -> Dict[int, np.ndarray]:
        weights_dict = {}
        if hasattr(self.gossip, "_protocol"):
            table = self.gossip._protocol._table
            for member_id in self.cluster_members:
                key = f"intra_weight_{member_id}"
                for nid, info in table.items():
                    if key in info:
                        weights_dict[member_id] = info[key]["weights"]
        return weights_dict

    def share_inter_cluster(
        self, cluster_weights: np.ndarray, shapley_value: float, current_round: int
    ) -> None:
        if not self.cluster_head:
            return
        if self.gossip is not None and hasattr(self.gossip, "_protocol"):
            self.gossip._protocol._table[self.node_id]["inter_cluster_weight"] = {
                "weights": cluster_weights,
                "shapley": shapley_value,
                "round": current_round,
            }

    def collect_inter_cluster_weights(self) -> Dict[int, np.ndarray]:
        weights_dict = {}
        if hasattr(self.gossip, "_protocol"):
            table = self.gossip._protocol._table
            for nid, info in table.items():
                if nid != self.node_id and "inter_cluster_weight" in info:
                    weights_dict[nid] = info["inter_cluster_weight"]["weights"]
        return weights_dict

    def aggregate_inter_cluster(
        self, inter_weights: Dict[int, np.ndarray], inter_shapley: Dict[int, float]
    ) -> np.ndarray:
        if not inter_weights:
            return np.array([])
        all_weights = {self.node_id: self._aggregated_weights}
        all_weights.update(inter_weights)
        all_shapley = {self.node_id: 1.0}
        all_shapley.update(inter_shapley)
        return self.federated_aggregate(all_weights, all_shapley)

    def get_aggregated_weights(self) -> Optional[np.ndarray]:
        return self._aggregated_weights

    def set_aggregated_weights(self, weights: np.ndarray) -> None:
        self._aggregated_weights = weights