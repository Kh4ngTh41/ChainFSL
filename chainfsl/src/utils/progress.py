"""
Progress tracking for ChainFSL experiments.

Provides:
- Global progress tracking (per-round, ETA)
- Per-node progress tracking (epochs, batches)
- Real-time console display
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, List


@dataclass
class NodeProgressInfo:
    """Progress info for a single node."""
    node_id: int
    current_round: int = 0
    epochs_trained: int = 0
    total_epochs_expected: int = 0
    batch_index: int = 0
    total_batches: int = 0
    status: str = "idle"  # idle, training, verified, aggregated, done, excluded, error
    last_loss: float = 0.0
    last_reward: float = 0.0
    cut_layer: int = 0
    batch_size: int = 0
    excluded_reason: str = ""

    @property
    def epoch_progress_pct(self) -> float:
        if self.total_batches == 0:
            return 0.0
        return min(100.0, 100.0 * self.batch_index / self.total_batches)

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "current_round": self.current_round,
            "epochs_trained": self.epochs_trained,
            "total_epochs_expected": self.total_epochs_expected,
            "batch_index": self.batch_index,
            "total_batches": self.total_batches,
            "epoch_progress_pct": self.epoch_progress_pct,
            "status": self.status,
            "last_loss": self.last_loss,
            "last_reward": self.last_reward,
            "cut_layer": self.cut_layer,
            "batch_size": self.batch_size,
            "excluded_reason": self.excluded_reason,
        }


@dataclass
class GlobalProgressInfo:
    """Global experiment progress."""
    current_round: int
    total_rounds: int
    status: str = "running"  # running, evaluating, checkpointing, done, error
    error_message: Optional[str] = None
    mean_round_time_s: float = 0.0
    _round_times: deque = field(default_factory=lambda: deque(maxlen=10))

    @property
    def overall_progress_pct(self) -> float:
        if self.total_rounds == 0:
            return 0.0
        return min(100.0, 100.0 * self.current_round / self.total_rounds)

    @property
    def eta_seconds(self) -> float:
        if self.mean_round_time_s == 0 or self.current_round >= self.total_rounds:
            return 0.0
        remaining = self.total_rounds - self.current_round
        return remaining * self.mean_round_time_s

    def update_round_time(self, elapsed_s: float) -> None:
        self._round_times.append(elapsed_s)
        self.mean_round_time_s = sum(self._round_times) / len(self._round_times)

    def to_dict(self) -> Dict:
        return {
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "overall_progress_pct": self.overall_progress_pct,
            "eta_seconds": self.eta_seconds,
            "rounds_complete": self.current_round,
            "rounds_remaining": self.total_rounds - self.current_round,
            "mean_round_time_s": self.mean_round_time_s,
            "status": self.status,
            "error_message": self.error_message,
        }


class ProgressTracker:
    """
    Central progress tracking with console display.

    Usage:
        tracker = ProgressTracker(total_rounds=100, n_nodes=10)

        for round_num in range(1, 101):
            tracker.start_round(round_num)
            # ... training ...
            tracker.end_round(
                round_num=round_num,
                train_metrics={"loss": 1.23, "accuracy": 0.65},
                per_node_states={0: NodeProgressInfo(...), ...},
            )
    """

    def __init__(
        self,
        total_rounds: int,
        n_nodes: int,
        eval_every: int = 10,
        checkpoint_every: int = 10,
    ):
        self.total_rounds = total_rounds
        self.n_nodes = n_nodes
        self.eval_every = eval_every
        self.checkpoint_every = checkpoint_every

        self.global_info = GlobalProgressInfo(
            current_round=0,
            total_rounds=total_rounds,
        )

        self.node_info: Dict[int, NodeProgressInfo] = {
            i: NodeProgressInfo(node_id=i) for i in range(n_nodes)
        }

        self._round_start_time: Optional[float] = None
        self._last_print_time: float = 0.0

    def start_round(self, round_num: int) -> None:
        """Mark round as started."""
        self.global_info.current_round = round_num
        self._round_start_time = time.perf_counter()

    def end_round(
        self,
        round_num: int,
        train_metrics: Optional[Dict[str, float]] = None,
        test_metrics: Optional[Dict[str, float]] = None,
        per_node_states: Optional[Dict[int, NodeProgressInfo]] = None,
        system_metrics: Optional[Dict[str, float]] = None,
        fairness_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Mark round as completed and print progress."""
        elapsed = time.perf_counter() - self._round_start_time
        self.global_info.update_round_time(elapsed)

        if per_node_states:
            self.node_info.update(per_node_states)

        self._print_progress(
            round_num, train_metrics, test_metrics,
            system_metrics, fairness_metrics
        )

    def _print_progress(
        self,
        round_num: int,
        train_metrics: Optional[Dict[str, float]],
        test_metrics: Optional[Dict[str, float]],
        system_metrics: Optional[Dict[str, float]],
        fairness_metrics: Optional[Dict[str, float]],
    ) -> None:
        """Print progress bar and metrics to console."""
        g = self.global_info
        eta_str = self._format_eta(g.eta_seconds)

        # Progress bar
        bar_len = 40
        filled = int(bar_len * g.current_round / max(g.total_rounds, 1))
        bar = "█" * filled + "░" * (bar_len - filled)

        # Header
        print(f"\r[{bar}] {g.overall_progress_pct:.1f}% | Round {round_num}/{g.total_rounds} | ETA: {eta_str}")

        # Train metrics
        if train_metrics:
            loss = train_metrics.get("loss", 0)
            acc = train_metrics.get("accuracy", 0)
            f1 = train_metrics.get("f1", 0)
            print(f"  Train: loss={loss:.4f} acc={acc:.2%} F1={f1:.3f}")

        # Test metrics
        if test_metrics:
            loss = test_metrics.get("loss", 0)
            acc = test_metrics.get("accuracy", 0)
            f1 = test_metrics.get("f1", 0)
            print(f"  Test:  loss={loss:.4f} acc={acc:.2%} F1={f1:.3f}")

        # System metrics
        if system_metrics:
            latency = system_metrics.get("round_latency_s", 0)
            ledger = system_metrics.get("ledger_size_kb", 0)
            verif_ms = system_metrics.get("verification_ms", 0)
            print(f"  System: latency={latency:.2f}s verif={verif_ms:.1f}ms ledger={ledger:.1f}KB")

        # Fairness
        if fairness_metrics:
            jains = fairness_metrics.get("jains", 0)
            gini = fairness_metrics.get("gini", 0)
            print(f"  Fairness: Jains={jains:.3f} Gini={gini:.3f}")

        # Per-node mini status
        node_statuses = []
        for nid, info in self.node_info.items():
            if info.status not in ("idle", "done"):
                pct = info.epoch_progress_pct
                status_char = {"training": "T", "verified": "V", "aggregated": "A", "excluded": "X", "error": "!"}.get(info.status, "?")
                node_statuses.append(f"n{nid}:{pct:.0f}%{status_char}")

        if node_statuses:
            print(f"  Nodes: {', '.join(node_statuses[:5])}" + (" ..." if len(node_statuses) > 5 else ""))

    def _format_eta(self, seconds: float) -> str:
        """Format ETA in human-readable form."""
        if seconds <= 0:
            return "--:--"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m {secs}s"

    def needs_checkpoint(self, round_num: int) -> bool:
        return round_num > 0 and round_num % self.checkpoint_every == 0

    def needs_eval(self, round_num: int) -> bool:
        return round_num > 0 and round_num % self.eval_every == 0

    def get_global_info(self) -> GlobalProgressInfo:
        return self.global_info

    def get_node_info(self, node_id: int) -> Optional[NodeProgressInfo]:
        return self.node_info.get(node_id)

    def all_nodes_done(self) -> bool:
        return all(info.status in ("done", "excluded") for info in self.node_info.values())

    def format_full_summary(self) -> str:
        """Format full experiment summary as string."""
        lines = [
            "=" * 70,
            "EXPERIMENT SUMMARY",
            "=" * 70,
            f"Rounds: {self.global_info.current_round}/{self.global_info.total_rounds}",
            f"Progress: {self.global_info.overall_progress_pct:.1f}%",
            f"Mean round time: {self.global_info.mean_round_time_s:.2f}s",
            f"ETA: {self._format_eta(self.global_info.eta_seconds)}",
            "=" * 70,
        ]
        return "\n".join(lines)
