"""
Checkpoint save/load for ChainFSL experiments.

Provides:
- Full protocol state serialization
- RNG state preservation (Python, NumPy, PyTorch)
- Round-level checkpointing for crash recovery
"""

import pickle
import random
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch


CHECKPOINT_VERSION = 1


def save_checkpoint(
    path: str,
    round_num: int,
    model_state: Dict[str, torch.Tensor],
    node_states: Dict[int, Dict[str, Any]],
    metrics_history: list,
    config: Dict[str, Any],
    node_progress: Optional[Dict[int, Dict[str, Any]]] = None,
    aggregator_state: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a checkpoint to disk.

    Args:
        path: Path to save checkpoint file.
        round_num: Current round number.
        model_state: Global model state dict.
        node_states: Per-node state (node_id -> state dict).
        metrics_history: List of RoundMetrics.to_dict() for each completed round.
        config: Config dict.
        node_progress: Optional NodeProgress.to_dict() per node.
        aggregator_state: Optional aggregator state.
    """
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "round": round_num,
        "timestamp": _timestamp(),
        "model_state": {k: v.clone() for k, v in model_state.items()},
        "node_states": node_states,
        "metrics_history": metrics_history,
        "config": config,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "node_progress": node_progress or {},
        "aggregator_state": aggregator_state,
    }

    # Atomic write: write to temp then rename
    path = Path(path)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(checkpoint, f)
    tmp_path.rename(path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    """
    Load a checkpoint from disk.

    Args:
        path: Path to checkpoint file.

    Returns:
        Checkpoint dict with keys: round, model_state, node_states, etc.
    """
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)

    # Verify version
    version = checkpoint.get("checkpoint_version", 0)
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"Checkpoint version mismatch: expected {CHECKPOINT_VERSION}, got {version}"
        )

    # Restore RNG state
    rng = checkpoint["rng_state"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(rng["torch_cuda"])

    return checkpoint


def restore_from_checkpoint(
    checkpoint: Dict[str, Any],
) -> tuple:
    """
    Restore all state from a checkpoint dict.

    Returns:
        (round_num, model_state, node_states, metrics_history, node_progress)
    """
    return (
        checkpoint["round"],
        checkpoint["model_state"],
        checkpoint["node_states"],
        checkpoint["metrics_history"],
        checkpoint.get("node_progress", {}),
    )


def checkpoint_exists(path: str) -> bool:
    """Check if a checkpoint file exists."""
    return Path(path).exists()


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the most recent checkpoint in a directory.

    Args:
        checkpoint_dir: Directory to search.

    Returns:
        Path to latest checkpoint, or None.
    """
    p = Path(checkpoint_dir)
    if not p.exists():
        return None
    checkpoints = list(p.glob("checkpoint_*.pkl"))
    if not checkpoints:
        return None
    return str(max(checkpoints, key=lambda x: x.stat().st_mtime))


def _timestamp() -> str:
    """Get ISO timestamp string."""
    from datetime import datetime
    return datetime.now().isoformat()
