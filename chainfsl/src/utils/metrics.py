"""
Metrics computation for ChainFSL experiments.

Provides:
- Per-class and macro-averaged precision, recall, F1
- Accuracy, confusion matrix
- Jains fairness index, Gini coefficient
"""

import numpy as np
from typing import Dict, Any, List, Optional


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_classes: int,
) -> Dict[str, float]:
    """
    Compute accuracy, precision, recall, F1 per-class and macro-averaged.

    Args:
        predictions: Predicted class indices (N,)
        targets: Ground truth class indices (N,)
        n_classes: Number of classes.

    Returns:
        Dict with per_class_* metrics and *_macro, *_weighted averages.
    """
    metrics: Dict[str, float] = {}

    # Per-class metrics
    for c in range(n_classes):
        tp = ((predictions == c) & (targets == c)).sum()
        fp = ((predictions == c) & (targets != c)).sum()
        fn = ((predictions != c) & (targets == c)).sum()
        tn = ((predictions != c) & (targets != c)).sum()

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-10)

        metrics[f"class_{c}_precision"] = float(precision)
        metrics[f"class_{c}_recall"] = float(recall)
        metrics[f"class_{c}_f1"] = float(f1)
        metrics[f"class_{c}_accuracy"] = float(accuracy)

    # Macro averages
    precisions = [metrics[f"class_{c}_precision"] for c in range(n_classes)]
    recalls = [metrics[f"class_{c}_recall"] for c in range(n_classes)]
    f1s = [metrics[f"class_{c}_f1"] for c in range(n_classes)]

    metrics["precision_macro"] = float(np.mean(precisions))
    metrics["recall_macro"] = float(np.mean(recalls))
    metrics["f1_macro"] = float(np.mean(f1s))

    # Weighted averages (by support)
    supports = [(targets == c).sum() for c in range(n_classes)]
    total_support = sum(supports) + 1e-10

    metrics["precision_weighted"] = float(sum(
        metrics[f"class_{c}_precision"] * supports[c] / total_support
        for c in range(n_classes)
    ))
    metrics["recall_weighted"] = float(sum(
        metrics[f"class_{c}_recall"] * supports[c] / total_support
        for c in range(n_classes)
    ))
    metrics["f1_weighted"] = float(sum(
        metrics[f"class_{c}_f1"] * supports[c] / total_support
        for c in range(n_classes)
    ))

    # Overall accuracy
    metrics["accuracy"] = float((predictions == targets).mean())

    return metrics


def compute_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        predictions: Predicted class indices (N,)
        targets: Ground truth class indices (N,)
        n_classes: Number of classes.

    Returns:
        Confusion matrix (n_classes, n_classes).
    """
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    for p, t in zip(predictions, targets):
        cm[t, p] += 1
    return cm


def jains_fairness(values: List[float]) -> float:
    """
    Jain's fairness index: (sum x_i)^2 / (n * sum x_i^2).

    Args:
        values: List of values (e.g., rewards or Shapley values).

    Returns:
        Fairness index in [0, 1]. 1 = perfect equality.
    """
    if not values or sum(values) == 0:
        return 0.0
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    return float((arr.sum() ** 2) / (n * (arr ** 2).sum()))


def gini_coefficient(values: List[float]) -> float:
    """
    Gini coefficient: 0 = perfect equality, 1 = maximum inequality.

    Args:
        values: List of values.

    Returns:
        Gini coefficient in [0, 1].
    """
    if not values:
        return 0.0
    arr = np.array(sorted(values), dtype=np.float64)
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    cumsum = np.cumsum(arr)
    return float((n + 1 - 2 * cumsum.sum() / (cumsum[-1] + 1e-10)) / n)


def format_metrics_table(metrics: Dict[str, float], prefix: str = "") -> str:
    """
    Format metrics dict as a readable table string.

    Args:
        metrics: Metrics dict.
        prefix: Optional prefix filter (e.g., "train_", "test_").

    Returns:
        Formatted string.
    """
    rows = []
    for key, val in sorted(metrics.items()):
        if prefix and not key.startswith(prefix):
            continue
        if "class_" in key:
            continue  # Skip per-class for summary
        rows.append(f"  {key}: {val:.4f}")
    return "\n".join(rows) if rows else "  (no metrics)"
