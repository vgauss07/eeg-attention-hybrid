"""
Evaluation metrics for EEG classification.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)
        y_prob: Predicted probabilities (N, n_classes), optional

    Returns:
        Dict with accuracy, balanced_accuracy, roc_auc, confusion_matrix
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }

    # ROC-AUC (one-vs-rest, macro)
    if y_prob is not None:
        try:
            n_classes = y_prob.shape[1]
            if n_classes == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                )
        except (ValueError, IndexError):
            metrics["roc_auc"] = 0.0
    else:
        metrics["roc_auc"] = 0.0

    # Confusion matrix (as nested list for JSON serialization)
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    return metrics


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names=None):
    """Pretty print sklearn classification report."""
    if class_names is None:
        class_names = [f"Class {i}" for i in sorted(np.unique(y_true))]
    print(classification_report(y_true, y_pred, target_names=class_names))
