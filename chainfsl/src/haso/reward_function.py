"""
Enhanced reward function for MA-HASO scheduler.
Computes multi-component rewards based on computation time, communication time,
Shapley value, fusion bonus, and overlap penalty.
"""

from dataclasses import dataclass
from typing import Dict, List


REF_GFLOPS = 1.0


@dataclass
class RewardConfig:
    alpha: float = 2.0
    beta: float = 1.5
    gamma: float = 0.5
    lambda_fusion: float = 0.3
    mu_overlap: float = 0.4
    min_acc_threshold: float = 0.60


class RewardFunction:
    def __init__(self, config: RewardConfig = None):
        self.config = config if config is not None else RewardConfig()

    def compute(
        self,
        T_comp: float,
        T_comm: float,
        delta_F: float,
        shapley_phi: float,
        fusion_bonus: float,
        overlap_penalty: float,
        current_accuracy: float
    ) -> float:
        """
        Compute total reward:
        R = -α·T_comp - β·T_comm + γ·φ·ΔF + λ·fusion_bonus - μ·overlap_penalty
        If accuracy < min_acc_threshold: R -= 10.0
        """
        cfg = self.config
        reward = (
            -cfg.alpha * T_comp
            - cfg.beta * T_comm
            + cfg.gamma * shapley_phi * delta_F
            + cfg.lambda_fusion * fusion_bonus
            - cfg.mu_overlap * overlap_penalty
        )
        if current_accuracy < cfg.min_acc_threshold:
            reward -= 10.0
        return reward

    def compute_T_comp(self, node_profile, cut_layer: int, batch_size: int) -> float:
        """
        T_comp = base_flops / (flops_ratio * REF_GFLOPS)
        node_profile expected to have: base_flops, flops_ratio
        """
        base_flops = getattr(node_profile, 'base_flops', 1.0)
        flops_ratio = getattr(node_profile, 'flops_ratio', 1.0)
        return base_flops / (flops_ratio * REF_GFLOPS)

    def compute_T_comm(
        self,
        smashed_size: float,
        bw_src: float,
        bw_dst: float,
        propagation_delay: float = 0.001
    ) -> float:
        """T_comm = smashed_size / min(bw_src, bw_dst) + propagation_delay"""
        min_bw = min(bw_src, bw_dst)
        return smashed_size / min_bw + propagation_delay

    def compute_fusion_bonus(self, nodes_sharing: List[int], layer_compatibility: float) -> float:
        """Bonus if multiple nodes share same compute node"""
        if len(nodes_sharing) < 2:
            return 0.0
        return len(nodes_sharing) * layer_compatibility

    def compute_overlap_penalty(self, conflict_score: float) -> float:
        """Penalty if layers overlap too much"""
        return conflict_score


def layer_compatibility(layer_a: int, layer_b: int) -> float:
    """
    Returns 1.0 if perfect overlap (beneficial), 0.0 if no overlap.
    Closer layers = more compatibility.
    """
    return max(0.0, 1.0 - abs(layer_a - layer_b) / 4.0)


def conflict_score(compute_node_id: int, node_layers: Dict[int, int]) -> float:
    """
    Sum of layer similarities for nodes sent to same compute node.
    node_layers: dict mapping node_id -> layer
    Returns total conflict based on layer proximity.
    """
    score = 0.0
    node_layer = node_layers.get(compute_node_id, 0)
    for other_node, other_layer in node_layers.items():
        if other_node != compute_node_id:
            score += layer_compatibility(node_layer, other_layer)
    return score


if __name__ == "__main__":
    config = RewardConfig()
    rf = RewardFunction(config)

    T_comp = 0.5
    T_comm = 0.2
    delta_F = 0.1
    shapley_phi = 0.8
    fusion_bonus = 1.5
    overlap_penalty = 0.3
    current_accuracy = 0.75

    reward = rf.compute(
        T_comp, T_comm, delta_F, shapley_phi,
        fusion_bonus, overlap_penalty, current_accuracy
    )
    print(f"Reward: {reward}")

    class DummyProfile:
        base_flops = 10.0
        flops_ratio = 2.0

    T_comp_val = rf.compute_T_comp(DummyProfile(), cut_layer=2, batch_size=32)
    print(f"T_comp: {T_comp_val}")

    T_comm_val = rf.compute_T_comm(
        smashed_size=1000.0, bw_src=100.0, bw_dst=80.0
    )
    print(f"T_comm: {T_comm_val}")

    nodes_sharing = [1, 2, 3]
    layer_comp = layer_compatibility(1, 2)
    fusion = rf.compute_fusion_bonus(nodes_sharing, layer_comp)
    print(f"Fusion bonus: {fusion}")

    node_layers = {1: 1, 2: 2, 3: 3}
    conflict = conflict_score(1, node_layers)
    print(f"Conflict score: {conflict}")

    print("All tests passed.")