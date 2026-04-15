"""
TVE Commitment — Tier-dependent proof generation.

Implements Algorithm 2 from the ChainFSL paper:
- Tier 1: Full gradient verification + zk-SNARK mock
- Tier 2: Gradient norm checks + Merkle proof
- Tier 3: Hash commitment (deferred verification)
- Tier 4: Hash only (lightweight)
"""

import hashlib
import time
from dataclasses import dataclass
from typing import Optional, Union
import torch
import numpy as np


@dataclass
class Proof:
    """Container for verification proof and metadata."""

    proof_type: str  # 'zk_snark_mock', 'merkle', 'hash_commit', 'hash_only'
    proof_data: bytes  # Main proof bytes
    input_hash: bytes  # Hash of input data
    activation_hash: bytes  # Hash of activation tensor
    model_hash: Optional[bytes] = None  # Hash of model weights (Tier 1 only)
    merkle_path: Optional[list[bytes]] = None  # Merkle proof path (Tier 2)
    verification_time_ms: float = 0.0
    is_valid: bool = True  # Default to valid, verification sets this

    def size_bytes(self) -> int:
        """Approximate size of this proof in bytes."""
        base = (
            len(self.proof_data)
            + len(self.input_hash)
            + len(self.activation_hash)
        )
        if self.merkle_path:
            base += sum(len(p) for p in self.merkle_path)
        return base


class CommitmentVerifier:
    """
    Tiered verification per Algorithm 2.

    Proof complexity scales with device tier to balance security and overhead.
    """

    # Thresholds for gradient consistency checks
    COSINE_SIM_THRESHOLD = 0.95  # Tier 1: gradient consistency
    NORM_STD_MULTIPLIER = 3.0    # Tier 2: norm bounds = μ ± 3σ

    @staticmethod
    def commit_input(x: torch.Tensor) -> bytes:
        """h_i = Hash(x_i) — commit before training."""
        data = x.detach().numpy().tobytes()
        return hashlib.sha256(data).digest()

    @staticmethod
    def commit_activation(a: torch.Tensor) -> bytes:
        """Hash of activation tensor."""
        data = a.detach().numpy().tobytes()
        return hashlib.sha256(data).digest()

    @staticmethod
    def commit_model(model_state: dict) -> bytes:
        """Hash of model weights."""
        # Serialize state dict deterministically
        state_bytes = b"".join(
            v.detach().numpy().tobytes() for v in sorted(model_state.values(), key=lambda x: str(x.shape))
        )
        return hashlib.sha256(state_bytes).digest()

    # ------------------------------------------------------------------ #
    # Proof generation by tier
    # ------------------------------------------------------------------ #

    @staticmethod
    def gen_proof_tier1(
        x: torch.Tensor,
        a: torch.Tensor,
        model_state: dict,
    ) -> Proof:
        """
        Tier 1: Full zk-SNARK proof (mock).

        Generates hash-based proof simulating zk-SNARK properties:
        - 200-byte proof size (mock)
        - ~2-5s proving time in production (scaled down for simulation)
        - Proves correct computation of activation given input and model

        Args:
            x: Input tensor.
            a: Activation tensor at cut layer.
            model_state: Client model state dict.

        Returns:
            Proof with zk_snark_mock type.
        """
        proof_start = time.perf_counter()

        # Mock proving delay (scaled down)
        time.sleep(0.05)  # 50ms mock vs real 2-5s

        model_hash = CommitmentVerifier.commit_model(model_state)
        input_hash = CommitmentVerifier.commit_input(x)
        act_hash = CommitmentVerifier.commit_activation(a)

        # Mock zk-SNARK: combine hashes in structured way
        combined = act_hash + input_hash + model_hash
        proof_data = hashlib.sha256(combined).digest() * 4  # ~64 bytes

        return Proof(
            proof_type="zk_snark_mock",
            proof_data=proof_data,
            input_hash=input_hash,
            activation_hash=act_hash,
            model_hash=model_hash,
            verification_time_ms=0.0,
        )

    @staticmethod
    def gen_proof_tier2(
        x: torch.Tensor,
        a: torch.Tensor,
        gradient: torch.Tensor,
        historical_norms: tuple[float, float] = (1.0, 0.5),
    ) -> Proof:
        """
        Tier 2: Gradient norm check + Merkle proof of activations.

        Args:
            x: Input tensor.
            a: Activation tensor.
            gradient: Gradient tensor.
            historical_norms: (mean, std) of gradient norms for bounds check.

        Returns:
            Proof with merkle type.
        """
        proof_start = time.perf_counter()

        # Merkle proof for activations
        act_hash = CommitmentVerifier.commit_activation(a)
        merkle_path = CommitmentVerifier._build_merkle_path(act_hash)

        # Gradient norm check
        grad_norm = gradient.norm().item()
        mean_norm, std_norm = historical_norms
        norm_bound_low = max(0.0, mean_norm - CommitmentVerifier.NORM_STD_MULTIPLIER * std_norm)
        norm_bound_high = mean_norm + CommitmentVerifier.NORM_STD_MULTIPLIER * std_norm

        # Encode norm bounds in proof
        norm_bytes = struct.pack(">ff", norm_bound_low, norm_bound_high)
        proof_data = merkle_path[0] if merkle_path else b""
        proof_data += norm_bytes + struct.pack(">f", grad_norm)

        return Proof(
            proof_type="merkle",
            proof_data=proof_data,
            input_hash=CommitmentVerifier.commit_input(x),
            activation_hash=act_hash,
            merkle_path=merkle_path,
            verification_time_ms=0.0,
        )

    @staticmethod
    def gen_proof_tier3(x: torch.Tensor, a: torch.Tensor) -> Proof:
        """
        Tier 3: Hash commitment (deferred verification).

        Lightweight: just hashes. Verification done asynchronously by Tier 1 auditors.

        Args:
            x: Input tensor.
            a: Activation tensor.

        Returns:
            Proof with hash_commit type.
        """
        return Proof(
            proof_type="hash_commit",
            proof_data=b"",  # No extra data needed
            input_hash=CommitmentVerifier.commit_input(x),
            activation_hash=CommitmentVerifier.commit_activation(a),
            verification_time_ms=0.0,
        )

    @staticmethod
    def gen_proof_tier4(x: torch.Tensor) -> Proof:
        """
        Tier 4: Hash only — minimal overhead.

        Args:
            x: Input tensor only (no activation needed).

        Returns:
            Proof with hash_only type.
        """
        return Proof(
            proof_type="hash_only",
            proof_data=b"",
            input_hash=CommitmentVerifier.commit_input(x),
            activation_hash=b"",  # Not required for Tier 4
            verification_time_ms=0.0,
        )

    @staticmethod
    def _build_merkle_path(leaf_hash: bytes, depth: int = 4) -> list[bytes]:
        """Build mock Merkle proof path."""
        path = []
        current = leaf_hash
        for _ in range(depth):
            # Mock sibling hash
            sibling = hashlib.sha256(current + b"sibling").digest()
            path.append(sibling)
            current = hashlib.sha256(current + sibling).digest()
        return path

    # ------------------------------------------------------------------ #
    # Proof verification
    # ------------------------------------------------------------------ #

    @staticmethod
    def verify_proof_tier1(proof: Proof, expected_input_hash: bytes) -> bool:
        """
        Verify Tier 1 zk-SNARK mock proof.

        Checks:
        1. Input hash matches
        2. Proof has expected structure (mock: always valid if hash matches)
        """
        if proof.input_hash != expected_input_hash:
            return False
        # Mock: trust proof if hashes match (production would verify zk-SNARK)
        return proof.is_valid

    @staticmethod
    def verify_proof_tier2(
        proof: Proof,
        expected_input_hash: bytes,
        gradient_norm: float,
        historical_norms: tuple[float, float] = (1.0, 0.5),
    ) -> bool:
        """
        Verify Tier 2 Merkle + norm proof.

        Checks:
        1. Input hash matches
        2. Gradient norm within μ ± 3σ bounds
        3. Merkle path validates (mock: structure check only)
        """
        if proof.input_hash != expected_input_hash:
            return False

        mean_norm, std_norm = historical_norms
        low = max(0.0, mean_norm - CommitmentVerifier.NORM_STD_MULTIPLIER * std_norm)
        high = mean_norm + CommitmentVerifier.NORM_STD_MULTIPLIER * std_norm

        if not (low <= gradient_norm <= high):
            return False

        # Merkle path structural check
        if proof.merkle_path is None or len(proof.merkle_path) < 2:
            return False

        return True

    @staticmethod
    def verify_proof_tier3(proof: Proof, expected_input_hash: bytes) -> bool:
        """Verify Tier 3 hash commitment (deferred)."""
        return proof.input_hash == expected_input_hash

    @staticmethod
    def verify_proof_tier4(proof: Proof) -> bool:
        """Verify Tier 4 hash only — always passes if hash present."""
        return len(proof.input_hash) == 32  # SHA-256 length

    @staticmethod
    def verify_proof(
        proof: Proof,
        tier: int,
        expected_input_hash: bytes,
        **kwargs,
    ) -> bool:
        """
        Unified verification dispatch.

        Args:
            proof: Proof to verify.
            tier: Device tier (1-4).
            expected_input_hash: Expected input hash.
            **kwargs: Additional args (gradient_norm, historical_norms for Tier 2).

        Returns:
            True if proof is valid.
        """
        if tier == 1:
            return CommitmentVerifier.verify_proof_tier1(proof, expected_input_hash)
        elif tier == 2:
            return CommitmentVerifier.verify_proof_tier2(
                proof,
                expected_input_hash,
                gradient_norm=kwargs.get("gradient_norm", 0.0),
                historical_norms=kwargs.get("historical_norms", (1.0, 0.5)),
            )
        elif tier == 3:
            return CommitmentVerifier.verify_proof_tier3(proof, expected_input_hash)
        elif tier == 4:
            return CommitmentVerifier.verify_proof_tier4(proof)
        return False


# Helper for struct packing
import struct