"""Evaluation metrics for classification tasks."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
    confusion_matrix,
)


def compute_all_metrics(y_true, y_pred, y_proba=None):
    """Compute all classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (for ROC-AUC).

    Returns:
        Dictionary of metric values.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
    }

    if y_proba is not None:
        try:
            if y_proba.ndim == 2:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = float("nan")

    return metrics


def evaluate_model(model, K_test, y_test):
    """Evaluate a fitted QSVM model.

    Args:
        model: Fitted QSVM instance.
        K_test: Test kernel matrix.
        y_test: True test labels.

    Returns:
        Dictionary of metrics.
    """
    y_pred = model.predict(K_test)
    try:
        y_proba = model.predict_proba(K_test)
    except Exception:
        y_proba = None

    return compute_all_metrics(y_test, y_pred, y_proba)


def print_results(metrics, title="Results"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name:>15s}: {value:.4f}")
        else:
            print(f"  {name:>15s}: {value}")
    print(f"{'='*50}\n")
