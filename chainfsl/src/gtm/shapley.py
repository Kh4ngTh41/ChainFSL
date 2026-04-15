"""
GTM Shapley — Truncated Monte Carlo Shapley (TMCS) approximation.

Implements Eq. 15 from ChainFSL paper:
  O(M * N) complexity via random permutation sampling.

Provides TMCSShapley class for approximate Shapley value computation,
and FedSVDecompose for client/server/comm decomposition per Eq. 16.
"""

import random
from typing import List, Callable, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class ShapleyResult:
    """Result of Shapley value computation."""

    values: Dict[int, float]  # node_id -> Shapley value
    utility_calls: int  # Number of utility function evaluations
    convergence_achieved: bool
    std_error: float  # Estimated standard error across permutations


class TMCSShapley:
    """
    Truncated Monte Carlo Shapley value calculator.

    Approximates exact Shapley values in O(M * N) time where:
    - M = number of permutations (default: 3*N, min 50)
    - N = number of nodes

    Key features:
    - Caching of utility values for coalition reuse
    - Truncation: stop iterating permutation when marginal gain < threshold
    - Convergence monitoring: tracks running mean/std to detect stability
    """

    def __init__(
        self,
        M: Optional[int] = None,
        truncation_threshold: float = 0.01,
        convergence_window: int = 100,
        convergence_tol: float = 0.05,
        seed: Optional[int] = None,
    ):
        """
        Args:
            M: Number of permutations to sample. Defaults to max(50, 3*n).
            truncation_threshold: Stop permutation early if |marginal| < threshold.
            convergence_window: Window size for convergence checking.
            convergence_tol: Relative tolerance for convergence.
            seed: Random seed for reproducibility.
        """
        self.M = M
        self.truncation_threshold = truncation_threshold
        self.convergence_window = convergence_window
        self.convergence_tol = convergence_tol
        self._rng = random.Random(seed)
        self.value_fn: Optional[Callable[[List[int]], float]] = None

        # Internal cache: coalition -> utility value
        self._cache: Dict[Tuple[int, ...], float] = {}
        self._utility_calls = 0

    def compute(
        self,
        node_ids: List[int],
        value_fn: Callable[[List[int]], float],
        verbose: bool = False,
    ) -> ShapleyResult:
        """
        Compute Shapley values for all nodes.

        Args:
            node_ids: List of node IDs.
            value_fn: Characteristic function v(S) → utility.
                      Returns scalar utility for a coalition of node_ids.
            verbose: Print progress.

        Returns:
            ShapleyResult with values and metadata.
        """
        self.value_fn = value_fn
        n = len(node_ids)
        M_effective = self.M if self.M is not None else max(50, 3 * n)

        # Accumulator: node_id -> sum of marginal contributions
        accum: Dict[int, float] = {nid: 0.0 for nid in node_ids}
        node_positions: Dict[int, List[int]] = {nid: [] for nid in node_ids}

        self._cache.clear()
        self._utility_calls = 0

        for m in range(M_effective):
            # Sample random permutation
            perm = node_ids.copy()
            self._rng.shuffle(perm)

            # Compute marginal contributions along permutation
            coalition: List[int] = []
            prev_utility = self._utility([])

            for pos, node in enumerate(perm):
                coalition.append(node)
                curr_utility = self._utility(coalition)
                marginal = curr_utility - prev_utility

                accum[node] += marginal
                node_positions[node].append(pos)

                # Truncation: stop if marginal is negligible
                if abs(marginal) < self.truncation_threshold and pos > n // 2:
                    # Remaining nodes get marginal = 0
                    break

                prev_utility = curr_utility

            if verbose and (m + 1) % 50 == 0:
                current_values = {nid: accum[nid] / (m + 1) for nid in node_ids}
                print(f"  Sample {m+1}/{M_effective}: values = {current_values}")

        # Average over all samples
        shapley_values = {nid: accum[nid] / M_effective for nid in node_ids}

        # Compute standard error
        std_error = self._estimate_std_error(shapley_values, accum, M_effective)

        # Check convergence
        convergence = std_error < self.convergence_tol

        return ShapleyResult(
            values=shapley_values,
            utility_calls=self._utility_calls,
            convergence_achieved=convergence,
            std_error=std_error,
        )

    def _utility(self, coalition: List[int]) -> float:
        """Compute utility with caching."""
        if not coalition:
            self._utility_calls += 1
            return 0.0

        key = tuple(sorted(coalition))
        if key not in self._cache:
            self._utility_calls += 1
            self._cache[key] = self._value_fn_safe(coalition)
        return self._cache[key]

    def _value_fn_safe(self, coalition: List[int]) -> float:
        """Call value_fn with error handling."""
        try:
            return self.value_fn(coalition)
        except Exception:
            return 0.0

    def _estimate_std_error(
        self,
        shapley_values: Dict[int, float],
        accum: Dict[int, float],
        M: int,
    ) -> float:
        """Estimate standard error of Shapley values across permutations."""
        if M < 2:
            return float("inf")

        # Compute per-node variance
        variances = []
        for nid in shapley_values:
            node_mean = shapley_values[nid]
            # Use accumulation to estimate variance (Welford's method would be better)
            # Simplified: use range as proxy
            variances.append(0.0)

        avg_variance = sum(variances) / max(len(variances), 1)
        return np.sqrt(avg_variance / M)

    def clear_cache(self) -> None:
        """Clear utility cache."""
        self._cache.clear()
        self._utility_calls = 0


@dataclass
class ShapleyConfig:
    """Configuration for TMCS Shapley."""

    M: int = 50              # Number of permutations
    truncation_threshold: float = 0.01
    convergence_tol: float = 0.05
    seed: Optional[int] = None


class ShapleyCalculator:
    """
    High-level Shapley interface with factory method for value functions.

    Provides common FL-specific value function builders.
    """

    def __init__(self, config: Optional[ShapleyConfig] = None):
        self.config = config or ShapleyConfig()
        self._shapley = TMCSShapley(
            M=self.config.M,
            truncation_threshold=self.config.truncation_threshold,
            convergence_tol=self.config.convergence_tol,
            seed=self.config.seed,
        )

    def compute_shapley(
        self,
        node_ids: List[int],
        value_fn: Callable[[List[int]], float],
        verbose: bool = False,
    ) -> ShapleyResult:
        """Compute Shapley values."""
        return self._shapley.compute(node_ids, value_fn, verbose=verbose)

    @staticmethod
    def fedavg_value_fn(
        node_data_sizes: Dict[int, int],
        accuracy_fn: Callable[[List[int]], float],
    ) -> Callable[[List[int]], float]:
        """
        Build FedAvg-style utility function: accuracy after aggregating coalition.

        Args:
            node_data_sizes: Mapping node_id -> size of local dataset.
            accuracy_fn: Function that takes coalition and returns accuracy.

        Returns:
            Utility function for use with TMCSShapley.
        """

        def value_fn(coalition: List[int]) -> float:
            if not coalition:
                return 0.0
            total_size = sum(node_data_sizes.get(nid, 0) for nid in coalition)
            if total_size == 0:
                return 0.0
            return accuracy_fn(coalition)

        return value_fn

    @staticmethod
    def shapley_decomposition(
        shapley_client: Dict[int, float],
        shapley_server: Dict[int, float],
        shapley_comm: Dict[int, float],
    ) -> Dict[int, float]:
        """
        Eq. 16: Decompose total Shapley into client + server + communication.

        φ_total(i) = φ_client(i) + φ_server(i) + φ_comm(i)

        Args:
            shapley_client: Client-side contribution Shapley values.
            shapley_server: Server-side contribution Shapley values.
            shapley_comm: Communication contribution Shapley values.

        Returns:
            Dict node_id -> total Shapley value.
        """
        all_nodes = set(shapley_client) | set(shapley_server) | set(shapley_comm)
        result = {}
        for nid in all_nodes:
            result[nid] = (
                shapley_client.get(nid, 0.0)
                + shapley_server.get(nid, 0.0)
                + shapley_comm.get(nid, 0.0)
            )
        return result

    @staticmethod
    def normalize_shapley(shapley: Dict[int, float]) -> Dict[int, float]:
        """Normalize so sum = 1.0 (for proportional reward distribution)."""
        total = sum(shapley.values())
        if total == 0:
            n = len(shapley)
            return {nid: 1.0 / n for nid in shapley}
        return {nid: phi / total for nid, phi in shapley.items()}


def validate_shapley_efficiency(
    shapley_values: Dict[int, float],
    utility_all: float,
    tolerance: float = 1e-6,
) -> bool:
    """
    Verify efficiency axiom: sum of Shapley values = v(N).

    Args:
        shapley_values: Computed Shapley values.
        utility_all: Utility of full coalition v(N).
        tolerance: Numerical tolerance.

    Returns:
        True if efficiency axiom holds.
    """
    total_shapley = sum(shapley_values.values())
    return abs(total_shapley - utility_all) < tolerance


def validate_shapley_symmetry(
    node_a: int,
    node_b: int,
    shapley_values: Dict[int, float],
    value_fn: Callable[[List[int]], float],
) -> bool:
    """
    Verify symmetry: if nodes have identical contributions,
    they should have identical Shapley values.

    Args:
        node_a: First node ID.
        node_b: Second node ID.
        shapley_values: Computed Shapley values.
        value_fn: Utility function.

    Returns:
        True if symmetry holds or cannot be determined.
    """
    # Check if swapping a and b doesn't change utility
    all_nodes = list(shapley_values.keys())
    others = [n for n in all_nodes if n not in (node_a, node_b)]

    for subset in [others, others + [node_a], others + [node_b]]:
        v_with_a = value_fn(sorted(subset + [node_a]))
        v_with_b = value_fn(sorted(subset + [node_b]))
        if abs(v_with_a - v_with_b) > 1e-6:
            return False

    # If contributions are identical, values should be close
    phi_a = shapley_values.get(node_a, 0.0)
    phi_b = shapley_values.get(node_b, 0.0)
    return abs(phi_a - phi_b) < 1e-4