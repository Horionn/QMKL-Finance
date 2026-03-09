"""Visualization utilities for QMKL experiments."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_kernel_heatmap(K, title="Kernel Matrix", save_path=None):
    """Plot a heatmap of a kernel matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(K, cmap="viridis", ax=ax, square=True, cbar_kws={"label": "Kernel value"})
    ax.set_title(title)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Sample index")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_curve(y_true, y_scores, labels=None, title="ROC Curve", save_path=None):
    """Plot ROC curves for one or more models.

    Args:
        y_true: True labels.
        y_scores: Single array or list of arrays of predicted scores.
        labels: List of model names.
    """
    from sklearn.metrics import roc_curve, auc

    if not isinstance(y_scores, list):
        y_scores = [y_scores]
        labels = labels or ["Model"]

    fig, ax = plt.subplots(figsize=(8, 6))

    for scores, label in zip(y_scores, labels):
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_concentration(kernel_stats_by_dim, title="Kernel Concentration Analysis", save_path=None):
    """Plot mean and variance of kernel elements vs feature dimension.

    Useful for diagnosing exponential concentration.

    Args:
        kernel_stats_by_dim: Dict mapping dimension -> kernel_statistics dict.
    """
    dims = sorted(kernel_stats_by_dim.keys())
    means = [kernel_stats_by_dim[d]["mean"] for d in dims]
    variances = [kernel_stats_by_dim[d]["variance"] for d in dims]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(dims, means, "o-", color="tab:blue")
    ax1.set_xlabel("Feature Dimension (n_qubits)")
    ax1.set_ylabel("Mean of off-diagonal elements")
    ax1.set_title("Kernel Mean vs Dimension")
    ax1.grid(True, alpha=0.3)

    ax2.plot(dims, variances, "s-", color="tab:orange")
    ax2.set_xlabel("Feature Dimension (n_qubits)")
    ax2.set_ylabel("Variance of off-diagonal elements")
    ax2.set_title("Kernel Variance vs Dimension")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_weights(weights, kernel_names=None, title="Kernel Weights", save_path=None):
    """Bar plot of kernel combination weights."""
    n = len(weights)
    if kernel_names is None:
        kernel_names = [f"K_{i}" for i in range(n)]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, n))
    bars = ax.bar(range(n), weights, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels(kernel_names, rotation=45, ha="right")
    ax.set_ylabel("Weight")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_comparison(results_dict, metric="roc_auc", title=None, save_path=None):
    """Compare multiple models on a metric.

    Args:
        results_dict: Dict mapping model_name -> metrics dict.
        metric: Which metric to compare.
    """
    names = list(results_dict.keys())
    values = [results_dict[n].get(metric, 0) for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    ax.barh(range(len(names)), values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel(metric.upper().replace("_", " "))
    ax.set_title(title or f"Model Comparison - {metric.upper()}")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
