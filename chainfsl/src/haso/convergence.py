"""
Convergence analysis for MA-HASO federated learning.
Provides empirical convergence tracking and theoretical bounds.
"""

from typing import Any, Dict, List, Optional, Tuple


class ConvergenceAnalyzer:
    """Analyze convergence from experiment data."""

    def __init__(self):
        self.rounds_data: List[Dict[str, float]] = []

    def add_round(self, round_idx: int, accuracy: float, loss: float, latency: float):
        """Add round data."""
        self.rounds_data.append({
            "round": round_idx,
            "accuracy": accuracy,
            "loss": loss,
            "latency": latency,
        })

    def compute_convergence_rate(self) -> float:
        """Δaccuracy / Δround averaged over all rounds."""
        if len(self.rounds_data) < 2:
            return 0.0

        total_rate = 0.0
        count = 0
        for i in range(1, len(self.rounds_data)):
            delta_acc = self.rounds_data[i]["accuracy"] - self.rounds_data[i - 1]["accuracy"]
            delta_round = self.rounds_data[i]["round"] - self.rounds_data[i - 1]["round"]
            if delta_round > 0:
                total_rate += delta_acc / delta_round
                count += 1

        return total_rate / count if count > 0 else 0.0

    def time_to_accuracy(self, threshold: float) -> Optional[int]:
        """Return first round where accuracy >= threshold, or None."""
        for entry in self.rounds_data:
            if entry["accuracy"] >= threshold:
                return entry["round"]
        return None

    def time_to_accuracy_curve(
        self, thresholds: List[float]
    ) -> Dict[float, int]:
        """Return {threshold: round} for multiple thresholds."""
        return {t: self.time_to_accuracy(t) for t in thresholds}

    def accuracy_curve_data(self) -> Tuple[List[int], List[float]]:
        """Return (rounds, accuracies) for plotting."""
        rounds = [e["round"] for e in self.rounds_data]
        accuracies = [e["accuracy"] for e in self.rounds_data]
        return rounds, accuracies

    def loss_curve_data(self) -> Tuple[List[int], List[float]]:
        """Return (rounds, losses) for plotting."""
        rounds = [e["round"] for e in self.rounds_data]
        losses = [e["loss"] for e in self.rounds_data]
        return rounds, losses

    def compute_regret(self, optimal_accuracy: float) -> List[float]:
        """Regret = optimal_accuracy - actual_accuracy per round."""
        return [optimal_accuracy - e["accuracy"] for e in self.rounds_data]


def convergence_bound(
    T: int,
    L_0: float,
    mu: float,
    sigma_sq: float,
    rho: float,
) -> float:
    """
    Formal bound: E[L_T] <= L_0/(μT) + σ²/(μT) + ρ·L_0/T

    Returns upper bound on expected loss after T rounds.

    Args:
        T: Number of rounds
        L_0: Initial loss
        mu: Smoothness parameter
        sigma_sq: Gradient variance bound
        rho: Compression ratio (smashed/raw data)

    Returns:
        Upper bound on expected loss
    """
    return L_0 / (mu * T) + sigma_sq / (mu * T) + rho * L_0 / T


def convergence_rate_theorem(
    accuracy_0: float,
    T: int,
    rho: float = 0.1,
    mu: float = 0.1,
    sigma_sq: float = 0.01,
) -> Dict[str, float]:
    """
    Returns analysis dict with:
    - bound: theoretical upper bound
    - rate: convergence rate (O(1/T))
    - compression_benefit: how much rho helps

    Args:
        accuracy_0: Initial accuracy (used to derive initial loss)
        T: Number of rounds
        rho: Compression ratio
        mu: Smoothness parameter
        sigma_sq: Gradient variance bound

    Returns:
        Dictionary with bound, rate, and compression_benefit
    """
    # Convert accuracy to loss proxy (assuming cross-entropy-like relationship)
    L_0 = 1.0 - accuracy_0

    bound = convergence_bound(T, L_0, mu, sigma_sq, rho)
    rate = 1.0 / T
    compression_benefit = rho * L_0 / T

    return {
        "bound": bound,
        "rate": rate,
        "compression_benefit": compression_benefit,
    }


def compare_convergence(
    data_a: List[Dict[str, float]],
    data_b: List[Dict[str, float]],
    threshold: float = 0.70,
) -> Dict[str, Any]:
    """
    Compare two convergence curves.

    Args:
        data_a: First dataset [{round, accuracy, loss}, ...]
        data_b: Second dataset [{round, accuracy, loss}, ...]
        threshold: Accuracy threshold for time-to-accuracy comparison

    Returns:
        Dictionary with:
        - time_to_acc_a: rounds to reach threshold for A
        - time_to_acc_b: rounds to reach threshold for B
        - speedup: ratio
        - final_acc_a, final_acc_b: final accuracies
    """
    # Extract time-to-accuracy for A
    time_a = None
    for entry in data_a:
        if entry.get("accuracy", 0) >= threshold:
            time_a = entry.get("round")
            break

    # Extract time-to-accuracy for B
    time_b = None
    for entry in data_b:
        if entry.get("accuracy", 0) >= threshold:
            time_b = entry.get("round")
            break

    final_a = data_a[-1].get("accuracy") if data_a else None
    final_b = data_b[-1].get("accuracy") if data_b else None

    speedup = None
    if time_a is not None and time_b is not None and time_a > 0:
        speedup = time_b / time_a

    return {
        "time_to_acc_a": time_a,
        "time_to_acc_b": time_b,
        "speedup": speedup,
        "final_acc_a": final_a,
        "final_acc_b": final_b,
    }


if __name__ == "__main__":
    # Example test
    analyzer = ConvergenceAnalyzer()
    for i in range(50):
        analyzer.add_round(i, 0.1 + 0.02 * i, 2.5 - 0.03 * i, 5.0 - 0.05 * i)

    rate = analyzer.compute_convergence_rate()
    tt70 = analyzer.time_to_accuracy(0.70)
    print(f"Rate: {rate:.4f}, Time to 70%: round {tt70}")

    # Test theoretical bound
    result = convergence_rate_theorem(accuracy_0=0.1, T=100)
    print(f"Bound: {result['bound']:.4f}, Rate: {result['rate']:.4f}")

    # Test comparison
    data_a = [{"round": i, "accuracy": 0.1 + 0.02 * i} for i in range(50)]
    data_b = [{"round": i, "accuracy": 0.1 + 0.025 * i} for i in range(50)]
    comparison = compare_convergence(data_a, data_b, threshold=0.70)
    print(f"Speedup: {comparison['speedup']:.2f}x")