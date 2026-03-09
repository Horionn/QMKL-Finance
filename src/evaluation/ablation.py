"""Ablation study tools for QMKL experiments.

Systematically measures the impact of each design choice:
- Number of qubits (feature dimension)
- Bandwidth parameter α
- Number of kernels in the combination
- Kernel type (fidelity vs projected)

Key improvements over naive holdout:
- Stratified K-fold CV for stable variance estimates
- Consensus kernel ranking across folds
- Weight analysis utilities (which kernels get zero weight and why)
"""

import warnings

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score


# ──────────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────────

def _frobenius_alignment(K1, K2):
    """Frobenius alignment between two kernel matrices."""
    num = np.sum(K1 * K2)
    denom = np.sqrt(np.sum(K1 * K1) * np.sum(K2 * K2))
    return float(num / denom) if denom > 0 else 0.0


def _evaluate_svm(K_tr, K_te, y_tr, y_te, C=1.0, scoring="roc_auc"):
    """Train SVM on precomputed kernel and return score."""
    # Ensure PSD
    min_eig = np.min(np.linalg.eigvalsh(K_tr))
    if min_eig < 0:
        K_tr = K_tr + (abs(min_eig) + 1e-10) * np.eye(K_tr.shape[0])

    svm = SVC(kernel="precomputed", C=C, probability=(scoring == "roc_auc"))
    svm.fit(K_tr, y_tr)

    if scoring == "roc_auc":
        return roc_auc_score(y_te, svm.predict_proba(K_te)[:, 1])
    else:
        return (svm.predict(K_te) == y_te).mean()


def _combine_kernels(K_list, weights):
    """Weighted sum of kernel matrices."""
    return sum(w * K for w, K in zip(weights, K_list))


# ──────────────────────────────────────────────
# 1. ABLATION: NUMBER OF QUBITS
# ──────────────────────────────────────────────

def ablation_n_qubits(
    X_raw,
    y,
    qubit_range,
    weight_fn,
    build_kernels_fn,
    preprocessing_fn,
    test_size=0.33,
    C=1.0,
    n_runs=5,
    scoring="roc_auc",
):
    """Study performance vs number of qubits.

    For each n_qubits value, applies PCA to that dimension,
    computes all kernels, runs MKL, evaluates.

    Args:
        X_raw: Raw features (before preprocessing).
        y: Labels.
        qubit_range: List of qubit counts to test, e.g. [2, 4, 6, 8].
        weight_fn: Callable(K_list_train, y_train) -> weights.
        build_kernels_fn: Callable(X_train, n_qubits) -> list of K_train matrices.
        preprocessing_fn: Callable(X, n_qubits) -> X_processed.
        test_size, C, n_runs, scoring: Standard params.

    Returns:
        Dict {n_qubits: {'mean': float, 'std': float, 'scores': list}}.
    """
    from .visualization import _kernel_off_diagonal_stats

    results = {}

    for n_qubits in qubit_range:
        print(f"  Testing n_qubits={n_qubits}...")
        scores = []
        kernel_stats_runs = []

        for seed in range(n_runs):
            X_proc = preprocessing_fn(X_raw, n_qubits)

            idx = np.arange(len(y))
            idx_tr, idx_te = train_test_split(
                idx, test_size=test_size, random_state=seed, stratify=y
            )
            X_tr, X_te = X_proc[idx_tr], X_proc[idx_te]
            y_tr, y_te = y[idx_tr], y[idx_te]

            K_list_train = build_kernels_fn(X_tr, n_qubits)
            K_list_test = build_kernels_fn(X_te, n_qubits, X_tr)

            run_stats = [_kernel_off_diagonal_stats(K) for K in K_list_train]
            kernel_stats_runs.append(run_stats)

            weights = weight_fn(K_list_train, y_tr)
            weights = np.array(weights)
            if weights.sum() > 0:
                weights = weights / weights.sum()

            K_tr = _combine_kernels(K_list_train, weights)
            K_te = _combine_kernels(K_list_test, weights)

            score = _evaluate_svm(K_tr, K_te, y_tr, y_te, C=C, scoring=scoring)
            scores.append(score)

        results[n_qubits] = {
            "mean": np.mean(scores),
            "std": np.std(scores, ddof=1),
            "scores": scores,
            "kernel_stats_runs": kernel_stats_runs,
        }

    return results


# ──────────────────────────────────────────────
# 2. ABLATION: NUMBER OF KERNELS (robust version)
# ──────────────────────────────────────────────

def _consensus_ranking(K_list_full, y, n_folds=5, random_state=42):
    """Rank kernels by alignment, averaged over multiple stratified folds.

    Instead of ranking on a single arbitrary train split (fragile),
    this computes alignment on each fold's training set and averages.

    Returns:
        ranked_indices: Kernel indices sorted by decreasing mean alignment.
        mean_alignments: Mean alignment score per kernel (same order as K_list_full).
    """
    n_kernels = len(K_list_full)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    alignment_matrix = np.zeros((n_folds, n_kernels))

    for fold_idx, (idx_tr, _) in enumerate(skf.split(np.zeros(len(y)), y)):
        y_tr = y[idx_tr]
        K_target_tr = np.outer(y_tr, y_tr).astype(float)

        for k_idx, K in enumerate(K_list_full):
            K_tr = K[np.ix_(idx_tr, idx_tr)]
            alignment_matrix[fold_idx, k_idx] = _frobenius_alignment(K_tr, K_target_tr)

    mean_alignments = np.mean(alignment_matrix, axis=0)
    ranked_indices = list(np.argsort(mean_alignments)[::-1])

    return ranked_indices, mean_alignments


def ablation_n_kernels(
    K_list_full,
    y,
    weight_fn,
    kernel_names,
    n_folds=5,
    C=1.0,
    scoring="roc_auc",
    random_state=42,
):
    """Study performance vs number of kernels using stratified K-fold CV.

    Improvements over the naive version:
    1. Consensus ranking: kernel importance averaged over all folds (not 1 seed)
    2. Stratified K-fold CV: every sample used in test exactly once per round
    3. Multiple CV rounds with different shuffles for stable estimates

    Args:
        K_list_full: List of precomputed kernel matrices on full dataset.
        y: Full label array.
        weight_fn: Callable(K_list_train, y_train) -> weights.
        kernel_names: List of kernel names (for reporting).
        n_folds: Number of CV folds (default 5).
        C: SVM regularization.
        scoring: 'roc_auc' or 'accuracy'.
        random_state: Base random seed.

    Returns:
        results: Dict {n_kernels: {'mean', 'std', 'scores', 'added_kernel', 'weights_per_fold'}}.
        ranking_info: Dict with ranking details.
    """
    n_full = len(y)
    n_kernels_total = len(K_list_full)

    # ── Consensus ranking ──────────────────────────────────────
    print("  Computing consensus kernel ranking (across folds)...")
    ranked_indices, mean_alignments = _consensus_ranking(
        K_list_full, y, n_folds=n_folds, random_state=random_state
    )
    ranked_names = [kernel_names[i] for i in ranked_indices]

    for rank, idx in enumerate(ranked_indices):
        print(f"    #{rank+1}: {kernel_names[idx]:20s} "
              f"(alignment = {mean_alignments[idx]:.4f})")

    # ── N_ROUNDS of shuffled K-fold CV for stability ───────────
    N_ROUNDS = 3  # 3 rounds × 5 folds = 15 evaluations per n_k
    results = {}

    for n_k in range(1, n_kernels_total + 1):
        subset_idx = ranked_indices[:n_k]
        K_subset = [K_list_full[i] for i in subset_idx]

        all_scores = []
        all_weights = []

        for round_idx in range(N_ROUNDS):
            seed = random_state + round_idx * 100
            skf = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=seed
            )

            for idx_tr, idx_te in skf.split(np.zeros(n_full), y):
                y_tr, y_te = y[idx_tr], y[idx_te]

                K_sub_tr = [K[np.ix_(idx_tr, idx_tr)] for K in K_subset]
                K_sub_te = [K[np.ix_(idx_te, idx_tr)] for K in K_subset]

                if n_k == 1:
                    weights = np.array([1.0])
                else:
                    weights = weight_fn(K_sub_tr, y_tr)
                    weights = np.array(weights)
                    if weights.sum() > 0:
                        weights = weights / weights.sum()
                    else:
                        weights = np.ones(n_k) / n_k

                all_weights.append(weights.copy())

                K_tr = _combine_kernels(K_sub_tr, weights)
                K_te = _combine_kernels(K_sub_te, weights)

                score = _evaluate_svm(K_tr, K_te, y_tr, y_te, C=C, scoring=scoring)
                all_scores.append(score)

        all_scores = np.array(all_scores)
        all_weights = np.array(all_weights)

        results[n_k] = {
            "mean": float(np.mean(all_scores)),
            "std": float(np.std(all_scores, ddof=1)),
            "ci95": float(1.96 * np.std(all_scores, ddof=1) / np.sqrt(len(all_scores))),
            "scores": all_scores.tolist(),
            "added_kernel": ranked_names[n_k - 1],
            "mean_weights": np.mean(all_weights, axis=0).tolist(),
            "n_evals": len(all_scores),
        }
        print(f"  n_kernels={n_k:2d} (+{ranked_names[n_k-1]:20s}): "
              f"{results[n_k]['mean']:.4f} ± {results[n_k]['std']:.4f}  "
              f"(CI95: ±{results[n_k]['ci95']:.4f}, n={results[n_k]['n_evals']})")

    ranking_info = {
        "ranked_indices": ranked_indices,
        "ranked_names": ranked_names,
        "mean_alignments": {kernel_names[i]: float(mean_alignments[i])
                            for i in range(n_kernels_total)},
    }

    return results, ranking_info


# ──────────────────────────────────────────────
# 3. ABLATION: BANDWIDTH ALPHA
# ──────────────────────────────────────────────

def ablation_alpha(
    X_processed,
    y,
    fm_name,
    alpha_range,
    kernel_type="fidelity",
    n_folds=5,
    C=1.0,
    scoring="roc_auc",
    random_state=42,
):
    """Study the impact of bandwidth α on a single kernel type.

    Uses stratified K-fold CV for stable estimates.

    Args:
        X_processed: Preprocessed feature array (n_samples, n_qubits).
        y: Labels.
        fm_name: Feature map name (e.g., 'ZZ').
        alpha_range: List of α values to test.
        kernel_type: 'fidelity' or 'projected'.

    Returns:
        Dict {alpha: {'mean': float, 'std': float, 'ci95': float}}.
    """
    from ..kernels.feature_maps import build_feature_map
    from ..kernels.quantum_kernel import build_quantum_kernel
    from ..kernels.kernel_matrix import compute_kernel_matrix

    n_qubits = X_processed.shape[1]
    results = {}

    for alpha in alpha_range:
        print(f"  Testing alpha={alpha}...")
        fm = build_feature_map(fm_name, n_qubits, alpha=alpha, reps=1)
        qk = build_quantum_kernel(fm, kernel_type=kernel_type)
        K_full = compute_kernel_matrix(qk, X_processed)

        scores = []
        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=random_state
        )
        for idx_tr, idx_te in skf.split(np.zeros(len(y)), y):
            y_tr, y_te = y[idx_tr], y[idx_te]
            K_tr = K_full[np.ix_(idx_tr, idx_tr)]
            K_te = K_full[np.ix_(idx_te, idx_tr)]

            score = _evaluate_svm(K_tr, K_te, y_tr, y_te, C=C, scoring=scoring)
            scores.append(score)

        scores = np.array(scores)
        results[alpha] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores, ddof=1)),
            "ci95": float(1.96 * np.std(scores, ddof=1) / np.sqrt(len(scores))),
        }
        print(f"    ROC-AUC: {results[alpha]['mean']:.4f} "
              f"± {results[alpha]['std']:.4f}")

    return results


# ──────────────────────────────────────────────
# 4. WEIGHT ANALYSIS
# ──────────────────────────────────────────────

def weight_analysis(
    K_list_full,
    y,
    weight_fn,
    kernel_names,
    n_folds=5,
    n_rounds=3,
    random_state=42,
):
    """Analyze which kernels get non-zero weight and why.

    Returns per-kernel statistics across all folds/rounds:
    - Mean weight and std
    - Fraction of times the kernel got non-zero weight (> 0.01)
    - Individual alignment score
    - Off-diagonal variance (concentration measure)

    Args:
        K_list_full: List of precomputed kernel matrices.
        y: Labels.
        weight_fn: MKL weight function.
        kernel_names: List of names.

    Returns:
        analysis: Dict {kernel_name: {mean_weight, std_weight, nonzero_rate, alignment, concentration}}.
        weights_matrix: (n_total_evals, n_kernels) array.
    """
    from .visualization import _kernel_off_diagonal_stats

    n_kernels = len(K_list_full)
    n_full = len(y)

    # Collect weights across folds
    all_weights = []

    for round_idx in range(n_rounds):
        seed = random_state + round_idx * 100
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for idx_tr, _ in skf.split(np.zeros(n_full), y):
            y_tr = y[idx_tr]
            K_list_tr = [K[np.ix_(idx_tr, idx_tr)] for K in K_list_full]

            w = weight_fn(K_list_tr, y_tr)
            w = np.array(w)
            if w.sum() > 0:
                w = w / w.sum()
            else:
                w = np.ones(n_kernels) / n_kernels
            all_weights.append(w)

    weights_matrix = np.array(all_weights)

    # Compute alignment and concentration on full dataset
    K_target = np.outer(y, y).astype(float)
    alignments = [_frobenius_alignment(K, K_target) for K in K_list_full]
    concentrations = [_kernel_off_diagonal_stats(K) for K in K_list_full]

    analysis = {}
    for i, name in enumerate(kernel_names):
        ws = weights_matrix[:, i]
        analysis[name] = {
            "mean_weight": float(np.mean(ws)),
            "std_weight": float(np.std(ws, ddof=1)),
            "nonzero_rate": float(np.mean(ws > 0.01)),
            "alignment": float(alignments[i]),
            "off_diag_std": float(concentrations[i]["std"]),
            "off_diag_mean": float(concentrations[i]["mean"]),
        }

    return analysis, weights_matrix
