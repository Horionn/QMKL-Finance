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


def _kernel_off_diagonal_stats(K):
    """Compute off-diagonal statistics of a kernel matrix.

    Used by ablation_n_qubits to measure concentration at each qubit count.

    Returns:
        Dict with 'mean', 'std', 'variance', 'min', 'max'.
    """
    n = K.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off_diag = K[mask]
    return {
        "mean":     float(np.mean(off_diag)),
        "std":      float(np.std(off_diag)),
        "variance": float(np.var(off_diag)),
        "min":      float(np.min(off_diag)),
        "max":      float(np.max(off_diag)),
    }


# ──────────────────────────────────────────────
# ADVANCED VISUALIZATIONS (Notebook 06)
# ──────────────────────────────────────────────

def plot_method_comparison_grouped(results_by_dataset, metric="mean",
                                   title=None, save_path=None):
    """Grouped bar chart: methods × datasets.

    Args:
        results_by_dataset: Dict {dataset_name: {method_name: {'mean': float, 'std': float}}}.
        metric: Key to plot from the inner dict ('mean').
    """
    datasets = list(results_by_dataset.keys())
    methods = list(results_by_dataset[datasets[0]].keys())
    n_d, n_m = len(datasets), len(methods)

    x = np.arange(n_m)
    width = 0.8 / n_d
    colors = plt.cm.Set2(np.linspace(0, 1, n_d))

    fig, ax = plt.subplots(figsize=(max(12, n_m * 1.2), 6))
    for i, ds in enumerate(datasets):
        vals = [results_by_dataset[ds][m][metric] for m in methods]
        errs = [results_by_dataset[ds][m].get("std", 0) for m in methods]
        offset = (i - n_d / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, yerr=errs, label=ds,
               color=colors[i], edgecolor="black", linewidth=0.5, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.set_title(title or "Comparaison des méthodes MKL par dataset", fontsize=13)
    ax.legend(title="Dataset", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_bo_convergence(histories, title=None, save_path=None):
    """Plot BO convergence curves (best_so_far vs iteration).

    Args:
        histories: Dict {label: convergence_history_dict} from
                   BayesianKernelOptimizer.get_convergence_history().
    """
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-iteration scores
    ax = axes[0]
    for (label, h), c in zip(histories.items(), colors):
        ax.scatter(range(len(h["scores"])), h["scores"],
                   alpha=0.3, color=c, s=15)
        ax.plot(h["best_so_far"], "-", color=c, linewidth=2, label=label)
    ax.set_xlabel("Itération BO")
    ax.set_ylabel("Score CV")
    ax.set_title("Convergence BO — meilleur score cumulé")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: zoom on last 50% of iterations
    ax = axes[1]
    for (label, h), c in zip(histories.items(), colors):
        mid = len(h["best_so_far"]) // 2
        ax.plot(range(mid, len(h["best_so_far"])), h["best_so_far"][mid:],
                "o-", color=c, linewidth=2, markersize=4, label=label)
    ax.set_xlabel("Itération BO")
    ax.set_ylabel("Meilleur score cumulé")
    ax.set_title("Convergence BO — zoom 2ème moitié")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(title or "Analyse de convergence — Bayesian Optimization", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_weight_heatmap(weights_dict, kernel_names, title=None, save_path=None):
    """Heatmap of weights: methods (rows) × kernels (columns).

    Args:
        weights_dict: Dict {method_name: np.array of weights}.
        kernel_names: List of kernel labels.
    """
    methods = list(weights_dict.keys())
    W = np.array([weights_dict[m] for m in methods])

    fig, ax = plt.subplots(figsize=(max(12, len(kernel_names) * 0.7), max(4, len(methods) * 0.6)))
    im = sns.heatmap(W, annot=True, fmt=".2f", cmap="YlOrRd",
                     xticklabels=kernel_names, yticklabels=methods,
                     ax=ax, cbar_kws={"label": "Poids"},
                     linewidths=0.5, linecolor="white",
                     annot_kws={"fontsize": 7})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_title(title or "Poids MKL par méthode et kernel", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_scaling_curve(scaling_results, title=None, save_path=None):
    """n_qubits scaling curve with CI95 band for multiple methods.

    Args:
        scaling_results: Dict {method_name: {n_qubits: {'mean', 'std', 'scores'}}}.
    """
    colors = {"BO": "#e74c3c", "Centered": "#2ecc71", "Single-Best": "#3498db",
              "Average": "#9b59b6", "SDP": "#f39c12", "Projection": "#1abc9c"}
    markers = {"BO": "o", "Centered": "s", "Single-Best": "^",
               "Average": "D", "SDP": "v", "Projection": "p"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, data in scaling_results.items():
        nqs = sorted(data.keys())
        means = [data[nq]["mean"] for nq in nqs]
        stds = [data[nq]["std"] for nq in nqs]
        n_eval = len(data[nqs[0]].get("scores", [1, 2, 3]))
        ci95 = [1.96 * s / np.sqrt(max(n_eval, 1)) for s in stds]

        c = colors.get(method, "gray")
        m = markers.get(method, "o")
        ax.plot(nqs, means, f"{m}-", color=c, linewidth=2, markersize=8, label=method)
        ax.fill_between(nqs, [mu - ci for mu, ci in zip(means, ci95)],
                        [mu + ci for mu, ci in zip(means, ci95)],
                        alpha=0.15, color=c)

    ax.set_xlabel("Nombre de qubits", fontsize=12)
    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.set_title(title or "Scaling — Performance vs nombre de qubits", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted(list(scaling_results.values())[0].keys()))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_concentration_scatter(kernel_stats, kernel_aucs, kernel_names,
                               title=None, save_path=None):
    """Scatter: kernel concentration (off-diag std) vs single-kernel AUC.

    Args:
        kernel_stats: List of dicts from _kernel_off_diagonal_stats.
        kernel_aucs: List of float AUC scores.
        kernel_names: List of labels.
    """
    stds = [s["std"] for s in kernel_stats]

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(stds, kernel_aucs, c=kernel_aucs, cmap="RdYlGn",
                    s=120, edgecolors="black", linewidth=0.8, zorder=3)
    for i, name in enumerate(kernel_names):
        ax.annotate(name[:12], (stds[i], kernel_aucs[i]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, alpha=0.8)
    plt.colorbar(sc, label="ROC-AUC")
    ax.set_xlabel("Écart-type hors-diagonale (pouvoir discriminant)", fontsize=11)
    ax.set_ylabel("ROC-AUC (kernel seul)", fontsize=11)
    ax.set_title(title or "Concentration vs Performance individuelle", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_radar_chart(metrics_by_method, title=None, save_path=None):
    """Radar (spider) chart comparing methods on multiple metrics.

    Args:
        metrics_by_method: Dict {method: {metric_name: value}}.
    """
    methods = list(metrics_by_method.keys())
    metric_names = list(metrics_by_method[methods[0]].keys())
    n_metrics = len(metric_names)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, method in enumerate(methods):
        values = [metrics_by_method[method][m] for m in metric_names]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=method, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(title or "Comparaison multi-métriques", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
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
