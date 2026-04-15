# ExperimentAgent Specification

## Purpose
Ensure experiments run crash-free with comprehensive metrics tracking and real-time progress visibility.

## Metrics Specification

### Per-Round Metrics (RoundMetricsExtended)

```python
@dataclass
class RoundMetricsExtended:
    # Identity
    round: int
    total_rounds: int

    # Progress
    round_progress_pct: float  # 0.0 to 100.0
    eta_seconds: float

    # Training metrics (per-batch aggregated)
    train_loss: float
    train_accuracy: float
    train_precision: float
    train_recall: float
    train_f1: float

    # Test metrics (computed at eval_every)
    test_loss: float
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float
    per_class_accuracy: Dict[int, float]  # class_id -> accuracy

    # Per-node metrics
    per_node_loss: Dict[int, float]
    per_node_accuracy: Dict[int, float]
    per_node_progress_pct: Dict[int, float]  # epochs_done / total_epochs
    per_node_cut_layer: Dict[int, int]
    per_node_batch_size: Dict[int, int]
    per_node_excluded: Dict[int, bool]  # True if OOM

    # System metrics
    round_latency_s: float
    verification_overhead_ms: float
    ledger_size_kb: float
    memory_peak_mb: float

    # Fairness
    jains_fairness: float
    gini_coefficient: float

    # TVE metrics
    attack_detection_rate: float
    false_positive_rate: float
    n_valid_updates: int
    n_participants: int

    # GTM metrics
    total_rewards_distributed: float
    shapley_variance: float
    shapley_mean: float
    lazy_nodes_detected: int

    # HASO metrics
    mean_cut_layer: float
    mean_batch_size: float
    haso_decisions_valid: float  # fraction of valid actions
```

### Per-Epoch/Step Progress

```python
# Real-time progress tracking
per_node_progress = {
    node_id: {
        "current_round": 45,
        "total_rounds": 100,
        "round_progress_pct": 67.5,  # 67.5% through round 45
        "epochs_trained": 45,
        "total_epochs_expected": 100,
        "epoch_progress_pct": 100.0,  # completed
        "batch_index": 156,
        "total_batches": 156,
        "batch_progress_pct": 100.0,
        "status": "training" | "verified" | "aggregated" | "done" | "excluded",
        "last_loss": 1.234,
        "last_reward": 12.5,
    }
}

global_progress = {
    "current_round": 45,
    "total_rounds": 100,
    "overall_progress_pct": 45.0,
    "eta_seconds": 3600.0,
    "rounds_complete": 44,
    "rounds_remaining": 55,
    "mean_round_time_s": 12.5,
    "status": "running" | "evaluating" | "checkpointing" | "done" | "error",
    "error_message": None,
}
```

## Progress Tracking Implementation

### Real-time Console Output Format

```
================================================================================
Experiment E1 | Round 45/100 [45.0%] | ETA: 1h 2m 14s
================================================================================
Train Loss: 1.234 | Train Acc: 65.2% | Train F1: 0.642
Test  Loss: 1.456 | Test  Acc: 58.3% | Test  F1: 0.578
--------------------------------------------------------------------------------
Per-Node Progress:
  Node 0: 100% | loss=1.20 | acc=67% | cut=3 | bs=32 | status=done
  Node 1:  80% | loss=1.35 | acc=63% | cut=2 | bs=32 | status=training
  Node 2: 100% | loss=1.15 | acc=68% | cut=4 | bs=32 | status=verified
  Node 3: 100% | loss=1.42 | acc=61% | cut=2 | bs=16 | status=excluded
  ...
================================================================================
System: latency=12.5s | verification=234ms | ledger=456KB
Fairness: Jains=0.85 | Gini=0.15
TVE: valid=47/50 | detection=95.0% | FP=5.0%
GTM: rewards=$500 | shapley_var=0.12
HASO: mean_cut=2.3 | mean_bs=28.5 | valid_actions=98.5%
================================================================================
```

## Crash-Proof Implementation

### Requirements

1. **Per-phase timeouts** (30s default, configurable)
2. **OOM handling** with graceful node exclusion
3. **Try/catch around every phase**
4. **KeyboardInterrupt handling** with state save
5. **Automatic checkpoint save** every N rounds
6. **Resume from checkpoint** capability

### Checkpoint Format

```json
{
  "checkpoint_version": 1,
  "round": 45,
  "timestamp": "2026-04-16T10:30:00Z",
  "model_state": {...},
  "node_states": {
    "0": {"cut_layer": 3, "batch_size": 32, "last_loss": 1.234},
    "1": {"cut_layer": 2, "batch_size": 32, "last_loss": 1.345},
    ...
  },
  "metrics_history": [...],
  "config": {...},
  "rng_state": {...}
}
```

### Crash Recovery Flow

```
1. On crash/error:
   a. Save checkpoint immediately
   b. Log error with full traceback
   c. Print "Saved checkpoint at round X"
   d. Exit with exit code != 0

2. On resume:
   a. Load checkpoint
   b. Restore model state
   c. Restore RNG state
   d. Continue from round + 1
```

## Metrics Collection Points

### Phase-by-Phase Collection

| Phase | Metrics Collected |
|-------|-----------------|
| HASO decision | cut_layer distribution, action validity |
| Training (per batch) | loss, accuracy, grad_norm |
| Training (per epoch) | avg_loss, avg_accuracy |
| Training (per node) | node_loss, node_accuracy, T_comp, T_comm |
| Verification | proof_type, verification_time, is_valid |
| Aggregation | n_valid, n_excluded, staleness |
| GTM | shapley_values, rewards, lazy_detected |
| Blockchain | ledger_size, commit_time |
| HASO update | ppo_timesteps, learning_rate |

## F1/Precision/Recall Implementation

```python
def compute_metrics(predictions: np.ndarray, targets: np.ndarray, n_classes: int) -> Dict[str, float]:
    """Compute accuracy, precision, recall, F1 per-class and macro."""
    metrics = {}

    # Per-class
    for c in range(n_classes):
        tp = ((predictions == c) & (targets == c)).sum()
        fp = ((predictions == c) & (targets != c)).sum()
        fn = ((predictions != c) & (targets == c)).sum()
        tn = ((predictions != c) & (targets != c)).sum()

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-10)

        metrics[f"class_{c}_precision"] = precision
        metrics[f"class_{c}_recall"] = recall
        metrics[f"class_{c}_f1"] = f1
        metrics[f"class_{c}_accuracy"] = accuracy

    # Macro averages
    metrics["precision_macro"] = np.mean([metrics[f"class_{c}_precision"] for c in range(n_classes)])
    metrics["recall_macro"] = np.mean([metrics[f"class_{c}_recall"] for c in range(n_classes)])
    metrics["f1_macro"] = np.mean([metrics[f"class_{c}_f1"] for c in range(n_classes)])

    # Weighted (by support)
    supports = [(targets == c).sum() for c in range(n_classes)]
    total_support = sum(supports)
    metrics["precision_weighted"] = sum(
        metrics[f"class_{c}_precision"] * supports[c] / total_support
        for c in range(n_classes)
    )
    metrics["recall_weighted"] = sum(
        metrics[f"class_{c}_recall"] * supports[c] / total_support
        for c in range(n_classes)
    )
    metrics["f1_weighted"] = sum(
        metrics[f"class_{c}_f1"] * supports[c] / total_support
        for c in range(n_classes)
    )

    return metrics
```

## Checkpoint Implementation

### Protocol Extension

```python
class ChainFSLProtocol:
    def save_checkpoint(self, path: str) -> None:
        """Save complete protocol state to checkpoint."""
        import pickle
        checkpoint = {
            "round": self.current_round,
            "model_state": {k: v.clone() for k, v in self.global_state.items()},
            "node_staleness": dict(self.node_staleness),
            "node_losses": dict(self.node_losses),
            "node_progress": {k: v.to_dict() for k, v in self.node_progress.items()},
            "metrics_history": [m.to_dict() for m in self.metrics_history],
            "config": self.cfg,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
        }
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self, path: str) -> None:
        """Restore protocol state from checkpoint."""
        import pickle
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        self.current_round = checkpoint["round"]
        self.global_state = checkpoint["model_state"]
        self.model.load_state_dict(self.global_state)
        self.node_staleness = checkpoint["node_staleness"]
        self.node_losses = checkpoint["node_losses"]
        self.node_progress = {
            k: NodeProgress(**v) for k, v in checkpoint["node_progress"].items()
        }
        self.metrics_history = [
            RoundMetrics(**m) for m in checkpoint["metrics_history"]
        ]

        random.setstate(checkpoint["rng_state"]["python"])
        np.random.set_state(checkpoint["rng_state"]["numpy"])
        torch.set_rng_state(checkpoint["rng_state"]["torch"])
```

## Integration with ChainFSLProtocol

### Required Changes to chainfsl.py

1. Add `save_checkpoint()` and `load_checkpoint()` methods
2. Wrap each phase in try/catch with timeout
3. Add progress callback system
4. Enhance `_collect_metrics()` with all ExtendedMetrics fields
5. Add F1/precision/recall computation in `_evaluate()`
6. Add per-node progress tracking in `NodeProgress`

### Phase Timeout Implementation

```python
import signal

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Phase timed out")

def run_with_timeout(fn, timeout_seconds, default=None):
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        result = fn()
        signal.alarm(0)
        return result
    except TimeoutError:
        print(f"WARNING: Phase timed out after {timeout_seconds}s")
        return default
    except Exception as e:
        print(f"ERROR: Phase failed with {e}")
        return default
```

## Files to Create/Modify

| File | Change | Agent |
|------|--------|-------|
| `src/protocol/chainfsl.py` | Add checkpoint, timeout, progress | ExperimentAgent |
| `src/protocol/chainfsl.py` | Enhance `_evaluate()` with F1 | ExperimentAgent |
| `src/sfl/trainer.py` | Add per-batch metrics return | ExperimentAgent |
| `src/protocol/chainfsl.py` | Enhance `NodeProgress` | ExperimentAgent |
| `experiments/utils.py` | Add metrics computation helpers | ExperimentAgent |
| `experiments/run_experiment.py` | Add resume from checkpoint | ExperimentAgent |
| New: `src/utils/metrics.py` | F1/precision/recall computation | ExperimentAgent |
| New: `src/utils/checkpoint.py` | Checkpoint save/load | ExperimentAgent |
| New: `src/utils/progress.py` | Progress tracking | ExperimentAgent |
