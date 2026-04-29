"""
Distributed PPO coordination via gossip for MA-HASO.

This module provides P2P gossip-based coordination between nodes,
replacing centralized decision-making in HASOOrchestrator.

Key concepts:
- Local PPO: Each node has its own PPO agent (from HaSOAgentPool)
- Gossip coordination: Nodes broadcast decisions to neighbors
- Consensus: Weighted average by reputation
- Configurable: Can toggle between centralized and P2P modes
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class NodeDecision:
    """Represents a node's local decision for a round."""
    node_id: int
    cut_layer: int
    batch_size: int
    H: int
    target_node: int
    reputation: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'cut_layer': self.cut_layer,
            'batch_size': self.batch_size,
            'H': self.H,
            'target_node': self.target_node,
            'reputation': self.reputation,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'NodeDecision':
        return cls(
            node_id=d['node_id'],
            cut_layer=d['cut_layer'],
            batch_size=d['batch_size'],
            H=d['H'],
            target_node=d['target_node'],
            reputation=d.get('reputation', 0.5),
        )


class DistributedPPOCoordinator:
    """
    Coordinates PPO decisions via P2P gossip.

    Replaces centralized HASOOrchestrator by:
    1. Each node makes local decision via its own PPO
    2. Nodes gossip decisions to k-nearest neighbors
    3. Consensus reached via reputation-weighted average

    Usage:
        coordinator = DistributedPPOCoordinator(
            node_id=0,
            cluster_members=[0, 1, 2, 3, 4],
            gossip=gossip_protocol,
        )

        # Each round:
        local_decision = local_ppo.decide(obs)
        coordinator.broadcast_decision(local_decision)
        coordinated = coordinator.get_consensus(node_id)
    """

    # Message types for gossip
    MSG_DECISION = 'haso_decision'
    MSG_GRADIENT = 'haso_gradient'

    def __init__(
        self,
        node_id: int,
        cluster_members: List[int],
        gossip,
        consensus_horizon: int = 3,
        reputation_weight: float = 0.7,
    ):
        """
        Args:
            node_id: This node's ID.
            cluster_members: List of node IDs in the cluster.
            gossip: HASOGossip instance for P2P communication.
            consensus_horizon: How many recent decisions to consider (gossip depth).
            reputation_weight: Weight of reputation in consensus (0-1).
        """
        self.node_id = node_id
        self.cluster_members = cluster_members
        self.gossip = gossip
        self.horizon = consensus_horizon
        self.reputation_weight = reputation_weight

    def broadcast_decision(self, decision: NodeDecision) -> None:
        """
        Broadcast local decision to cluster neighbors via gossip.

        Args:
            decision: Local decision from this node's PPO.
        """
        msg = {
            'type': self.MSG_DECISION,
            'sender_id': self.node_id,
            'decision': decision.to_dict(),
        }
        # Broadcast to intra-cluster neighbors
        self.gossip.broadcast_intra_cluster(
            node_id=self.node_id,
            profile=None,  # Not needed for decision broadcast
            current_load=0.0,
            fanout=len(self.cluster_members),  # Broadcast to all cluster members
        )

    def gather_decisions(self, sender_id: int) -> List[NodeDecision]:
        """
        Gather recent decisions from neighbors.

        Args:
            sender_id: Node that is gathering (to exclude its own decision if needed).

        Returns:
            List of NodeDecision from neighbors.
        """
        decisions = []

        # Get decisions from gossip table
        # In a real implementation, this would query the gossip protocol
        # Here we use the shared table approach
        if hasattr(self.gossip, '_protocol'):
            table = self.gossip._protocol._table
            for nid, info in table.items():
                if nid == sender_id:
                    continue
                if 'last_decision' in info:
                    try:
                        decisions.append(NodeDecision.from_dict(info['last_decision']))
                    except (KeyError, TypeError):
                        pass

        return decisions

    def get_consensus(
        self,
        node_id: int,
        local_decision: NodeDecision,
    ) -> NodeDecision:
        """
        Compute consensus decision for a node.

        Combines local decision with neighbor decisions via
        reputation-weighted voting.

        Args:
            node_id: Node to compute consensus for.
            local_decision: This node's local PPO decision.

        Returns:
            Consensus decision (possibly modified from local).
        """
        neighbors = self.gather_decisions(node_id)

        if not neighbors:
            # No neighbor decisions, use local
            return local_decision

        # Weight by reputation
        total_rep = sum(d.reputation for d in neighbors) + local_decision.reputation
        if total_rep == 0:
            return local_decision

        # Weighted average for cut_layer (discrete, so we round)
        local_weight = local_decision.reputation / total_rep
        neighbor_weights = [d.reputation / total_rep for d in neighbors]

        # For discrete actions, use softmax-like voting
        cut_layer_votes = [local_decision.cut_layer * local_weight]
        cut_layer_votes.extend([
            d.cut_layer * w for d, w in zip(neighbors, neighbor_weights)
        ])

        # Final cut_layer = weighted vote rounded to nearest valid
        consensus_cut = int(round(sum(cut_layer_votes)))
        consensus_cut = max(1, min(4, consensus_cut))  # Clamp to [1, 4]

        # For batch_size and H, use weighted average
        all_batch = [local_decision.batch_size * local_weight]
        all_batch.extend([d.batch_size * w for d, w in zip(neighbors, neighbor_weights)])
        consensus_batch = max(8, int(round(sum(all_batch))))  # Min batch 8

        all_H = [local_decision.H * local_weight]
        all_H.extend([d.H * w for d, w in zip(neighbors, neighbor_weights)])
        consensus_H = max(1, int(round(sum(all_H))))

        return NodeDecision(
            node_id=node_id,
            cut_layer=consensus_cut,
            batch_size=consensus_batch,
            H=consensus_H,
            target_node=local_decision.target_node,
            reputation=local_decision.reputation,
        )

    def store_decision(self, decision: NodeDecision) -> None:
        """
        Store decision in gossip table for others to read.

        Args:
            decision: Decision to store.
        """
        if hasattr(self.gossip, '_protocol'):
            nid = decision.node_id
            if nid not in self.gossip._protocol._table:
                self.gossip._protocol._table[nid] = {}
            self.gossip._protocol._table[nid]['last_decision'] = decision.to_dict()


class HybridPPOManager:
    """
    Manages hybrid PPO architecture with configurable modes.

    Supports:
    - 'centralized': HASOOrchestrator decides for all
    - 'cluster': ClusterHASOAgent decides per cluster
    - 'fully_distributed': Per-node PPO with P2P gossip consensus

    This class provides a unified interface while delegating to the
    appropriate underlying implementation.
    """

    def __init__(
        self,
        mode: str = 'cluster',
        node_id: int = 0,
        cluster_members: List[int] = None,
        gossip=None,
        haso_agent_pool=None,
        cluster_agent_pool=None,
    ):
        """
        Args:
            mode: Architecture mode ('centralized', 'cluster', 'fully_distributed').
            node_id: This node's ID.
            cluster_members: List of cluster member IDs (for cluster/distributed modes).
            gossip: HASOGossip instance.
            haso_agent_pool: HaSOAgentPool instance (for per-node agents).
            cluster_agent_pool: ClusterAgentPool instance (for cluster agents).
        """
        self.mode = mode
        self.node_id = node_id
        self.cluster_members = cluster_members or []

        if mode == 'fully_distributed':
            self._coordinator = DistributedPPOCoordinator(
                node_id=node_id,
                cluster_members=cluster_members,
                gossip=gossip,
            )
        else:
            self._coordinator = None

        self.haso_pool = haso_agent_pool
        self.cluster_pool = cluster_agent_pool

    def decide_all(
        self,
        obs_list: List[np.ndarray],
        deterministic: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get decisions for all nodes based on architecture mode.

        Args:
            obs_list: List of observations per node.
            deterministic: If True, use mean policy.

        Returns:
            List of decision dicts per node.
        """
        if self.mode == 'fully_distributed':
            return self._decide_distributed(obs_list, deterministic)
        elif self.mode == 'cluster':
            return self._decide_cluster(obs_list, deterministic)
        else:
            return self._decide_centralized(obs_list, deterministic)

    def _decide_distributed(
        self,
        obs_list: List[np.ndarray],
        deterministic: bool,
    ) -> List[Dict[str, Any]]:
        """Per-node PPO with P2P gossip consensus."""
        if self.haso_pool is None:
            raise ValueError("haso_agent_pool required for fully_distributed mode")

        # Step 1: Local decisions
        local_decisions = self.haso_pool.decide_all(obs_list, deterministic)

        # Step 2: Convert to NodeDecision and store in gossip
        for nid, dec in enumerate(local_decisions):
            node_dec = NodeDecision(
                node_id=nid,
                cut_layer=dec.get('cut_layer', 2),
                batch_size=dec.get('batch_size', 32),
                H=dec.get('H', 10),
                target_node=dec.get('target_compute_node', 0),
                reputation=self._get_reputation(nid),
            )
            self._coordinator.store_decision(node_dec)

        # Step 3: Gossip broadcast
        # In real implementation, this would trigger async gossip
        # For simulation, we use synchronous shared table

        # Step 4: Consensus per node
        results = []
        for nid, local_dec in enumerate(local_decisions):
            node_dec = NodeDecision(
                node_id=nid,
                cut_layer=local_dec.get('cut_layer', 2),
                batch_size=local_dec.get('batch_size', 32),
                H=local_dec.get('H', 10),
                target_node=local_dec.get('target_compute_node', 0),
                reputation=self._get_reputation(nid),
            )
            consensus = self._coordinator.get_consensus(nid, node_dec)
            results.append({
                'node_id': nid,
                'cut_layer': consensus.cut_layer,
                'batch_size': consensus.batch_size,
                'H': consensus.H,
                'target_compute_node': consensus.target_node,
                'mode': 'distributed_consensus',
            })

        return results

    def _decide_cluster(
        self,
        obs_list: List[np.ndarray],
        deterministic: bool,
    ) -> List[Dict[str, Any]]:
        """Cluster-level decisions via ClusterHASOAgent."""
        if self.cluster_pool is None:
            raise ValueError("cluster_agent_pool required for cluster mode")

        all_decisions = []
        for cid, agent in self.cluster_pool.agents.items():
            cluster_decisions = agent.decide_per_node(
                obs_list[cid], deterministic=deterministic
            )
            all_decisions.extend(cluster_decisions)
        return all_decisions

    def _decide_centralized(
        self,
        obs_list: List[np.ndarray],
        deterministic: bool,
    ) -> List[Dict[str, Any]]:
        """Per-node decisions without gossip coordination."""
        if self.haso_pool is None:
            raise ValueError("haSO_agent_pool required for centralized mode")

        decisions = self.haso_pool.decide_all(obs_list, deterministic)
        for dec in decisions:
            dec['mode'] = 'per_node_local'
        return decisions

    def _get_reputation(self, node_id: int) -> float:
        """Get reputation for a node from gossip table."""
        if hasattr(self.gossip, '_protocol'):
            info = self.gossip._protocol._table.get(node_id, {})
            return info.get('reputation', 0.5)
        return 0.5

    def learn_all(self, total_timesteps: int) -> None:
        """Update all PPO agents based on mode."""
        if self.mode == 'fully_distributed' or self.mode == 'per_node':
            if self.haso_pool:
                self.haso_pool.learn_all(total_timesteps)
        elif self.mode == 'cluster':
            if self.cluster_pool:
                self.cluster_pool.learn_all(total_timesteps)

    def update_shapley_all(self, shapley_dict: Dict[int, float]) -> None:
        """Update Shapley values for all agents."""
        if self.haso_pool:
            self.haso_pool.update_shapley_all(shapley_dict)
        if self.cluster_pool:
            for cid, agent in self.cluster_pool.agents.items():
                cluster_shapley = {
                    nid: shapley_dict[nid]
                    for nid in self.cluster_pool.cluster_manager.get_cluster_members(cid)
                    if nid in shapley_dict
                }
                if cluster_shapley:
                    agent.update_cluster_shapley(cluster_shapley)
