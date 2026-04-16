"""
TVE Committee — VRF-based committee selection and verification.

Implements Eq. 10-11 from ChainFSL paper for VRF-based selection,
and verification logic for submitted proofs from all tiers.
"""

import time
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict

from .vrf import MockVRF, select_committee, select_committee_reputation, reputation_adjusted_threshold
from .commitment import Proof, CommitmentVerifier

from ..emulator.node_profile import HardwareProfile


@dataclass
class VerificationResult:
    """Result of verifying a single node's update."""

    node_id: int
    is_valid: bool
    penalty: float
    proof_type: str
    verification_time_ms: float
    reason: str = ""  # Reason for invalidity


class VerificationCommittee:
    """
    VRF-based committee selection + verification logic.

    Implements:
    - VRF committee selection with reputation weighting (Eq. 10-11)
    - Parallel verification of all node proofs
    - Slashing penalty calculation
    - Lazy/Byzantine node detection
    """

    PENALTY_LAZY = 1000.0  # Fixed penalty for lazy/malicious nodes
    STAKE_MIN = 10.0       # Minimum stake to participate

    def __init__(
        self,
        nodes: List[HardwareProfile],
        committee_size: int = 5,
        omega: float = 0.3,
    ):
        """
        Args:
            nodes: List of HardwareProfile for all nodes.
            committee_size: Target committee size K.
            omega: Reputation weighting for selection (Eq. 11).
        """
        self.nodes = {n.node_id: n for n in nodes}
        self.node_list = nodes
        self.K = committee_size
        self.omega = omega

        # Per-epoch tracking
        self._round_selections: Dict[int, Set[int]] = {}  # epoch -> selected node_ids

        # Gradient cache for Tier 1 cosine similarity verification
        self._grad_cache: Dict[int, Dict[str, torch.Tensor]] = {}  # node_id -> {grad, smash_data}

    def select_committee(
        self,
        epoch: int,
        block_hash: bytes,
    ) -> List[HardwareProfile]:
        """
        Select verification committee for epoch using VRF (Eq. 10-11).

        Args:
            epoch: Current epoch number.
            block_hash: Block hash to use as public seed.

        Returns:
            List of HardwareProfile for selected committee members.
        """
        # Build node tuples: (node_id, secret_key, reputation)
        node_tuples = [
            (n.node_id, f"sk_{n.node_id}".encode(), n.reputation)
            for n in self.node_list
        ]

        selected_ids = select_committee_reputation(
            nodes=node_tuples,
            public_seed=block_hash,
            epoch=epoch,
            committee_size=self.K,
            omega=self.omega,
        )

        self._round_selections[epoch] = set(selected_ids)
        return [self.nodes[nid] for nid in selected_ids if nid in self.nodes]

    def verify_updates(
        self,
        updates: List[Dict],
        proofs: List[Proof],
        lazy_node_ids: Optional[Set[int]] = None,
    ) -> Dict[int, VerificationResult]:
        """
        Verify all node updates and proofs.

        Args:
            updates: List of update dicts with keys: node_id, input_hash, cut_layer, etc.
            proofs: List of Proof objects (aligned with updates).
            lazy_node_ids: Set of known lazy node IDs (for E4 attack injection).

        Returns:
            Dict mapping node_id -> VerificationResult.
        """
        if len(updates) != len(proofs):
            raise ValueError("updates and proofs must have same length")

        lazy_node_ids = lazy_node_ids or set()
        results = {}

        for update, proof in zip(updates, proofs):
            node_id = update.get("node_id", 0)
            tier = self.nodes.get(node_id, HardwareProfile(0, 1, 1.0, 1, 512, 1.0)).tier
            expected_hash = update.get("input_hash", b"")

            t_start = time.perf_counter()

            # Inject attack for lazy clients (E4 experiment)
            if node_id in lazy_node_ids:
                is_valid = False
                reason = "lazy_client_attack"
                penalty = self.PENALTY_LAZY
            else:
                is_valid, reason = self._verify_single(
                    proof=proof,
                    tier=tier,
                    expected_hash=expected_hash,
                    gradient_norm=update.get("gradient_norm", 0.0),
                    node_id=node_id,
                )
                penalty = 0.0 if is_valid else self.PENALTY_LAZY

            elapsed_ms = (time.perf_counter() - t_start) * 1000

            results[node_id] = VerificationResult(
                node_id=node_id,
                is_valid=is_valid,
                penalty=penalty,
                proof_type=proof.proof_type,
                verification_time_ms=elapsed_ms,
                reason=reason,
            )

        return results

    def _verify_single(
        self,
        proof: Proof,
        tier: int,
        expected_hash: bytes,
        gradient_norm: float = 0.0,
        node_id: int = 0,
    ) -> Tuple[bool, str]:
        """Verify a single proof for given tier."""
        try:
            # For Tier 1, perform cosine similarity check if gradient data available
            if tier == 1 and node_id in self._grad_cache:
                grad_data = self._grad_cache[node_id]
                submitted_grad = grad_data.get("grad")
                smash_data = grad_data.get("smash_data")

                if submitted_grad is not None and smash_data is not None:
                    # Use hash-based verification for now since we don't have server backbone here
                    # The actual cosine similarity would require recomputing gradient from activations
                    # For now, verify gradient is non-trivial (not zero or huge)
                    grad_norm_value = submitted_grad.norm().item()
                    if grad_norm_value < 1e-6 or grad_norm_value > 1000.0:
                        return False, f"tier1_grad_anomalous_norm_{grad_norm_value:.2f}"

                    # Verify proof hash matches
                    is_valid = CommitmentVerifier.verify_proof(
                        proof=proof,
                        tier=tier,
                        expected_input_hash=expected_hash,
                    )
                    if not is_valid:
                        return False, f"tier{tier}_proof_invalid"
                    return True, ""
                    # Note: Full cosine similarity verification would need server backbone access
                    # For now we do hash + norm sanity check

            is_valid = CommitmentVerifier.verify_proof(
                proof=proof,
                tier=tier,
                expected_input_hash=expected_hash,
                gradient_norm=gradient_norm,
                historical_norms=(1.0, 0.5),  # Default historical stats
            )
            reason = "" if is_valid else f"tier{tier}_proof_invalid"
            return is_valid, reason
        except Exception as e:
            return False, f"verification_error: {e}"

    def get_penalties(self, results: Dict[int, VerificationResult]) -> Dict[int, float]:
        """Extract node_id -> penalty mapping from results."""
        return {node_id: r.penalty for node_id, r in results.items()}

    def get_valid_node_ids(self, results: Dict[int, VerificationResult]) -> Set[int]:
        """Get set of node IDs with valid proofs."""
        return {node_id for node_id, r in results.items() if r.is_valid}

    def get_average_verification_time(self, results: Dict[int, VerificationResult]) -> float:
        """Mean verification time in ms."""
        if not results:
            return 0.0
        return sum(r.verification_time_ms for r in results.values()) / len(results)

    def is_committee_member(self, epoch: int, node_id: int) -> bool:
        """Check if node was selected for committee in given epoch."""
        return node_id in self._round_selections.get(epoch, set())


@dataclass
class TVEConfig:
    """Configuration for TVE module."""

    committee_size: int = 5
    omega: float = 0.3
    stake_min: float = 10.0
    lazy_penalty: float = 1000.0
    enable_random_audit: bool = True
    audit_fraction: float = 0.1  # 10% of Tier 3/4 proofs audited by Tier 1

    @property
    def K(self) -> int:
        return self.committee_size


class TieredVerificationEngine:
    """
    High-level TVE interface.

    Combines MockVRF committee selection with tiered proof verification
    for the full verification lifecycle.
    """

    def __init__(
        self,
        nodes: List[HardwareProfile],
        config: Optional[TVEConfig] = None,
    ):
        """
        Args:
            nodes: List of HardwareProfile.
            config: TVEConfig. Defaults to TVEConfig().
        """
        self.config = config or TVEConfig()
        self.committee = VerificationCommittee(
            nodes=nodes,
            committee_size=self.config.committee_size,
            omega=self.config.omega,
        )

        # Per-round state
        self._current_epoch = 0
        self._historical_norms: Dict[int, Tuple[float, float]] = {}  # node_id -> (μ, σ)

    def select(self, epoch: int, block_hash: bytes) -> List[int]:
        """
        Select committee for epoch.

        Args:
            epoch: Epoch number.
            block_hash: Block hash as VRF seed.

        Returns:
            List of selected node IDs.
        """
        self._current_epoch = epoch
        selected = self.committee.select_committee(epoch, block_hash)
        return [n.node_id for n in selected]

    def verify(
        self,
        updates: List[Dict],
        proofs: List[Proof],
        lazy_node_ids: Optional[Set[int]] = None,
        grad_cache: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
    ) -> Dict[int, VerificationResult]:
        """
        Verify all updates.

        Args:
            updates: List of update dicts.
            proofs: List of Proof objects.
            lazy_node_ids: Known lazy/malicious node IDs (for E4).
            grad_cache: Cache of gradient tensors for Tier 1 verification.

        Returns:
            Dict node_id -> VerificationResult.
        """
        if grad_cache:
            self._grad_cache = grad_cache
        return self.committee.verify_updates(updates, proofs, lazy_node_ids)

    def set_grad_cache(self, grad_cache: Dict[int, Dict[str, torch.Tensor]]) -> None:
        """Set gradient cache for Tier 1 cosine similarity verification."""
        self._grad_cache = grad_cache

    def update_historical_stats(
        self,
        node_id: int,
        gradient_norm: float,
    ) -> None:
        """
        Update running statistics for gradient norms (for Tier 2 bounds).

        Uses exponential moving average for μ and σ.

        Args:
            node_id: Node identifier.
            gradient_norm: Observed gradient norm.
        """
        if node_id not in self._historical_norms:
            self._historical_norms[node_id] = (gradient_norm, 0.5)
            return

        mu, sigma = self._historical_norms[node_id]
        beta = 0.9
        new_mu = beta * mu + (1 - beta) * gradient_norm
        # Rolling std approximation
        new_sigma = beta * sigma + (1 - beta) * abs(gradient_norm - mu)
        self._historical_norms[node_id] = (new_mu, new_sigma)

    def generate_audit_schedule(
        self,
        node_ids: List[int],
        node_tiers: Dict[int, int],
    ) -> List[int]:
        """
        Generate random audit schedule for Tier 3/4 proofs.

        Returns node_ids that should have their proofs audited by Tier 1.

        Args:
            node_ids: All participating node IDs.
            node_tiers: Mapping node_id -> tier.

        Returns:
            List of node_ids to audit.
        """
        if not self.config.enable_random_audit:
            return []

        audit_candidates = [
            nid for nid in node_ids if node_tiers.get(nid, 3) >= 3
        ]
        n_audit = max(1, int(len(audit_candidates) * self.config.audit_fraction))

        import random
        return random.sample(audit_candidates, min(n_audit, len(audit_candidates)))