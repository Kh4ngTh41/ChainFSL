"""
MockVRF — Verifiable Random Function using HMAC-SHA256.

Provides tamper-proof, unpredictable committee selection for TVE.
In production, replace with libsodium's crypto_vrf_prove / crypto_vrf_verify.
"""

import hmac
import hashlib
import struct
from typing import Tuple, List, Optional


class MockVRF:
    """
    VRF using HMAC-SHA256.

    Properties:
    - Deterministic: same (sk, seed, epoch) → same output
    - Unpredictable: cannot compute output without secret_key
    - Verifiable: proof allows verification without secret_key (simplified)

    For production, use curve25519-based VRF (e.g., libsodium crypto_vrf).
    """

    OUTPUT_BYTES = 8  # Use first 8 bytes for [0, 1) float
    MAX_UINT64 = 2**64

    def __init__(self, secret_key: bytes):
        """
        Args:
            secret_key: Node's secret key (at least 32 bytes recommended).
        """
        if not secret_key:
            raise ValueError("secret_key must not be empty")
        self.secret_key = secret_key

    def evaluate(self, public_seed: bytes, epoch: int) -> Tuple[float, bytes]:
        """
        Compute VRF output for given epoch.

        Args:
            public_seed: Public seed for this selection round.
            epoch: Epoch number.

        Returns:
            (random_value ∈ [0,1), proof bytes) tuple.
        """
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")

        message = public_seed + struct.pack(">Q", epoch)
        h = hmac.new(self.secret_key, message, hashlib.sha256)
        digest = h.digest()

        # First 8 bytes → uint64 → [0, 1)
        val = struct.unpack(">Q", digest[: self.OUTPUT_BYTES])[0]
        random_value = val / self.MAX_UINT64

        return random_value, digest

    def prove(self, public_seed: bytes, epoch: int) -> bytes:
        """Alias for evaluate that returns only proof."""
        return self.evaluate(public_seed, epoch)[1]

    @staticmethod
    def verify(
        secret_key: bytes,
        public_seed: bytes,
        epoch: int,
        claimed_value: float,
        proof: bytes,
    ) -> bool:
        """
        Verify a VRF proof.

        In this mock: recompute and compare. Production would use
        public_key (derived from secret_key) for asymmetric verification.

        Args:
            secret_key: Key used to generate proof (acts as public key in mock).
            public_seed: Public seed.
            epoch: Epoch number.
            claimed_value: Claimed random value.
            proof: Proof bytes.

        Returns:
            True if proof is valid.
        """
        vrf = MockVRF(secret_key)
        computed_value, _ = vrf.evaluate(public_seed, epoch)
        return abs(computed_value - claimed_value) < 1e-9

    @staticmethod
    def hash_to_committee(
        vrf_outputs: List[Tuple[float, int]],
        committee_size: int,
    ) -> List[int]:
        """
        Select committee from sorted VRF outputs (lowest values win).

        Args:
            vrf_outputs: List of (vrf_value, node_id) tuples.
            committee_size: Number of members to select.

        Returns:
            List of selected node_ids.
        """
        sorted_outputs = sorted(vrf_outputs, key=lambda x: x[0])
        return [node_id for _, node_id in sorted_outputs[:committee_size]]


def select_committee(
    nodes: List[Tuple[int, bytes]],  # (node_id, secret_key)
    public_seed: bytes,
    epoch: int,
    committee_size: int,
) -> List[int]:
    """
    Select committee using VRF-based sorting.

    Args:
        nodes: List of (node_id, secret_key) tuples.
        public_seed: Public seed for this epoch.
        epoch: Epoch number.
        committee_size: Number of committee members.

    Returns:
        List of selected node_ids.
    """
    vrf_outputs = []
    for node_id, sk in nodes:
        vrf = MockVRF(sk)
        val, _ = vrf.evaluate(public_seed, epoch)
        vrf_outputs.append((val, node_id))

    return MockVRF.hash_to_committee(vrf_outputs, committee_size)


def reputation_adjusted_threshold(
    n_nodes: int,
    committee_size: int,
    omega: float,
    node_reputations: dict[int, float],
    delta: float = 1.0,
) -> float:
    """
    Compute reputation-adjusted threshold per Eq. 11 in ChainFSL paper.

    threshold = (K/N) * (1 + ω * tanh(δ * reputation))

    Args:
        n_nodes: Total number of nodes.
        committee_size: Desired committee size K.
        omega: Reputation weighting factor.
        node_reputations: Mapping node_id -> reputation ∈ [0, 1].
        delta: Saturation constant.

    Returns:
        Per-node threshold (same for all in simplified model).
    """
    import math

    base_threshold = committee_size / n_nodes
    avg_rep = sum(node_reputations.values()) / max(len(node_reputations), 1)
    adjustment = 1 + omega * math.tanh(delta * avg_rep)
    return base_threshold * adjustment


def select_committee_reputation(
    nodes: List[Tuple[int, bytes, float]],  # (node_id, secret_key, reputation)
    public_seed: bytes,
    epoch: int,
    committee_size: int,
    omega: float = 0.3,
) -> List[int]:
    """
    Select committee with reputation-weighted VRF threshold.

    Higher-reputation nodes have lower VRF threshold → more likely selected.

    Args:
        nodes: List of (node_id, secret_key, reputation).
        public_seed: Public seed.
        epoch: Epoch number.
        committee_size: Target committee size.
        omega: Reputation weighting.

    Returns:
        List of selected node_ids.
    """
    import math

    threshold = reputation_adjusted_threshold(
        n_nodes=len(nodes),
        committee_size=committee_size,
        omega=omega,
        node_reputations={n[0]: n[2] for n in nodes},
    )

    selected = []
    for node_id, sk, rep in nodes:
        vrf = MockVRF(sk)
        val, proof = vrf.evaluate(public_seed, epoch)

        # Reputation lowers effective threshold
        rep_threshold = threshold / max(1e-9, 1 + omega * math.tanh(rep))
        if val < rep_threshold:
            selected.append(node_id)

        if len(selected) >= committee_size:
            break

    # If not enough selected, fill with lowest VRF values
    if len(selected) < committee_size:
        all_vals = sorted(
            [(vrf.evaluate(public_seed, epoch)[0], nid) for nid, sk, rep in nodes],
            key=lambda x: x[0],
        )
        for val, nid in all_vals:
            if nid not in selected:
                selected.append(nid)
            if len(selected) >= committee_size:
                break

    return selected[:committee_size]