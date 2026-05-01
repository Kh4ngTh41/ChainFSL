"""
Structured logging framework for ChainFSL experiments.

Provides hierarchical loggers for different components with
consistent formatting and log levels.
"""

import logging
import sys
from typing import Optional
from enum import IntEnum


class ChainFSLLogLevel(IntEnum):
    """Log levels matching Python logging."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class ChainFSLLogger:
    """Hierarchical logger for ChainFSL components."""

    _registry: dict[str, logging.Logger] = {}
    _default_level = logging.INFO

    @classmethod
    def get_logger(cls, name: str, level: Optional[int] = None) -> logging.Logger:
        """
        Get or create a logger with the given name.

        Args:
            name: Logger name (e.g., 'haso.agent', 'sfl.trainer').
            level: Optional log level override.

        Returns:
            Configured logger instance.
        """
        if name in cls._registry:
            return cls._registry[name]

        logger = logging.getLogger(name)
        logger.setLevel(level or cls._default_level)

        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(level or cls._default_level)
            formatter = logging.Formatter(
                fmt="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
                datefmt="%H:%M:%S"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        cls._registry[name] = logger
        return logger

    @classmethod
    def set_default_level(cls, level: int) -> None:
        """Set default level for all new loggers."""
        cls._default_level = level


class ExperimentLogger:
    """High-level logger for experiment tracking."""

    def __init__(self, name: str = "experiment", log_dir: Optional[str] = None):
        self.logger = ChainFSLLogger.get_logger(f"chainfsl.{name}")
        self.log_dir = log_dir
        self._metrics_buffer: list[dict] = []

    def log_round_start(self, round_id: int, n_nodes: int) -> None:
        self.logger.info(f"[Round {round_id}] Starting with {n_nodes} nodes")

    def log_round_end(self, round_id: int, metrics: dict) -> None:
        loss = metrics.get("loss", -1)
        acc = metrics.get("accuracy", -1)
        self.logger.info(f"[Round {round_id}] Completed | loss={loss:.4f} acc={acc:.4f}")

    def log_node_selection(self, round_id: int, selected_nodes: list[int]) -> None:
        self.logger.debug(f"[Round {round_id}] Selected nodes: {selected_nodes}")

    def log_training_start(self, node_id: int, cut_layer: int, batch_size: int) -> None:
        self.logger.debug(f"[Node {node_id}] Training | cut={cut_layer} batch={batch_size}")

    def log_training_end(self, node_id: int, duration: float, loss: float) -> None:
        self.logger.debug(f"[Node {node_id}] Training done | dur={duration:.2f}s loss={loss:.4f}")

    def log_shapley_computed(self, node_id: int, phi: float) -> None:
        self.logger.debug(f"[Node {node_id}] Shapley value: {phi:.6f}")

    def log_aggregation(self, round_id: int, method: str, duration: float) -> None:
        self.logger.info(f"[Round {round_id}] Aggregated ({method}) in {duration:.2f}s")

    def log_error(self, component: str, error: str) -> None:
        self.logger.error(f"[{component}] {error}")

    def log_metrics(self, round_id: int, node_id: int, metrics: dict) -> None:
        self._metrics_buffer.append({
            "round": round_id,
            "node": node_id,
            **metrics
        })

    def flush_metrics(self) -> list[dict]:
        """Flush and return buffered metrics."""
        metrics = self._metrics_buffer.copy()
        self._metrics_buffer.clear()
        return metrics


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Convenience function to get a logger."""
    return ChainFSLLogger.get_logger(f"chainfsl.{name}", level)


def set_level(level: int) -> None:
    """Set default log level for all ChainFSL loggers."""
    ChainFSLLogger.set_default_level(level)


def test():
    """Test logging framework."""
    print("=== Logging Framework Test ===")

    logger = get_logger("test")
    logger.info("Info message")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")

    exp_logger = ExperimentLogger("test_exp")
    exp_logger.log_round_start(1, 4)
    exp_logger.log_training_start(0, 2, 32)
    exp_logger.log_training_end(0, 1.5, 0.45)
    exp_logger.log_round_end(1, {"loss": 0.45, "accuracy": 0.87})

    print("\nAll logging tests passed!")


if __name__ == "__main__":
    test()