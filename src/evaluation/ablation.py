"""Ablation study tools for QMKL experiments.

Systematically measures the impact of each design choice:
- Number of qubits (feature dimension)
- Bandwidth parameter α
- Number of kernels in the combination
- Kernel type (fidelity vs projected)
- Entanglement pattern
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score


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
        qubit_range: List of qubit counts to test, e.g. [4, 6, 8, 10, 12].
        weight_fn: Callable(K_list_train, y_train) -> weights.
        build_kernels_fn: Callable(X_train, n_qubits) -> list of K_train matrices.
        preprocessing_fn: Callable(X, n_qubits) -> X_processed.
        test_size, C, n_runs, scoring: Standard params.

    Returns:
        Dict {n_qubits: {'mean': float, 'std': float, 'kernel_stats': list}}.
    """
    results = {}

    for n_qubits in qubit_range:
        print(f"  Testing n_qubits={n_qubits}...")
        scores = []
        kernel_stats_runs = []

        for seed in range(n_runs):
            # Preprocess with n_qubits components
            X_proc = preprocessing_fn(X_raw, n_qubits)

            idx = np.arange(len(y))
            idx_tr, idx_te = train_test_split(
                idx, test_size=test_size, random_state=seed, stratify=y
            )
            X_tr, X_te = X_proc[idx_tr], X_proc[idx_te]
            y_tr, y_te = y[idx_tr], y[idx_te]

            # Compute kernels
            K_list_train = build_kernels_fn(X_tr, n_qubits)
            K_list_test  = build_kernels_fn(X_te, n_qubits, X_tr)

            # Kernel statistics (for concentration analysis)
            from .visualization import _kernel_off_diagonal_stats
            run_stats = [_kernel_off_diagonal_stats(K) for K in K_list_train]
            kernel_stats_runs.append(run_stats)

            # MKL
            weights = weight_fn(K_list_train, y_tr)
            weights = np.array(weights)
            if weights.sum() > 0:
                weights = weights / weights.sum()

            K_tr = sum(w * K for w, K in zip(weights, K_list_train))
            K_te = sum(w * K for w, K in zip(weights, K_list_test))

            # Ensure PSD
            min_eig = np.min(np.linalg.eigvalsh(K_tr))
            if min_eig < 0:
                K_tr += (abs(min_eig) + 1e-10) * np.eye(K_tr.shape[0])

            # SVM
            svm = SVC(kernel="precomputed", C=C, probability=(scoring == "roc_auc"))
            svm.fit(K_tr, y_tr)

            if scoring == "roc_auc":
                score = roc_auc_score(y_te, svm.predict_proba(K_te)[:, 1])
            else:
                score = (svm.predict(K_te) == y_te).mean()

            scores.append(score)

        results[n_qubits] = {
            "mean": np.mean(scores),
            "std":  np.std(scores, ddof=1),
            "scores": scores,
            "kernel_stats_runs": kernel_stats_runs,
        }

    return results


def ablation_n_kernels(
    K_list_full,
    y,
    weight_fn,
    kernel_names,
    test_size=0.33,
    C=1.0,
    n_runs=10,
    scoring="roc_auc",
):
    """Study performance vs number of kernels in the combination.

    Starts with the best single kernel, adds kernels one by one
    in order of their individual alignment score.

    Args:
        K_list_full: List of all kernel matrices (on full dataset).
        y: Labels.
        weight_fn: Callable to optimize weights.
        kernel_names: List of kernel names.

    Returns:
        Dict {n_kernels: {'mean': float, 'std': float}}.
    """
    n_full = len(y)

    # First, rank kernels by individual alignment with target
    K_target = np.outer(y, y).astype(float)
    from .statistical_analysis import _frobenius_alignment

    # Rank kernels
    alignments = []
    idx_full = np.arange(n_full)
    idx_tr, _ = train_test_split(idx_full, test_size=test_size, random_state=0, stratify=y)
    for i, K in enumerate(K_list_full):
        K_tr = K[np.ix_(idx_tr, idx_tr)]
        Ky_tr = K_target[np.ix_(idx_tr, idx_tr)]
        a = _frobenius_alignment(K_tr, Ky_tr)
        alignments.append((a, i))

    ranked_indices = [i for _, i in sorted(alignments, reverse=True)]
    ranked_names = [kernel_names[i] for i in ranked_indices]

    results = {}
    for n_k in range(1, len(K_list_full) + 1):
        subset_idx = ranked_indices[:n_k]
        K_subset = [K_list_full[i] for i in subset_idx]

        scores = []
        for seed in range(n_runs):
            idx = np.arange(n_full)
            idx_tr, idx_te = train_test_split(
                idx, test_size=test_size, random_state=seed, stratify=y
            )
            y_tr, y_te = y[idx_tr], y[idx_te]

            K_sub_tr = [K[np.ix_(idx_tr, idx_tr)] for K in K_subset]
            K_sub_te = [K[np.ix_(idx_te, idx_tr)] for K in K_subset]

            if n_k == 1:
                weights = [1.0]
            else:
                weights = weight_fn(K_sub_tr, y_tr)
                weights = np.array(weights)
                if weights.sum() > 0:
                    weights = weights / weights.sum()

            K_tr = sum(w * K for w, K in zip(weights, K_sub_tr))
            K_te = sum(w * K for w, K in zip(weights, K_sub_te))

            min_eig = np.min(np.linalg.eigvalsh(K_tr))
            if min_eig < 0:
                K_tr += (abs(min_eig) + 1e-10) * np.eye(K_tr.shape[0])

            svm = SVC(kernel="precomputed", C=C, probability=(scoring == "roc_auc"))
            svm.fit(K_tr, y_tr)

            if scoring == "roc_auc":
                score = roc_auc_score(y_te, svm.predict_proba(K_te)[:, 1])
            else:
                score = (svm.predict(K_te) == y_te).mean()

            scores.append(score)

        results[n_k] = {
            "mean": np.mean(scores),
            "std": np.std(scores, ddof=1),
            "added_kernel": ranked_names[n_k - 1],
        }
        print(f"  n_kernels={n_k} (+{ranked_names[n_k-1]}): "
              f"{results[n_k]['mean']:.4f} ± {results[n_k]['std']:.4f}")

    return results, ranked_names


def ablation_alpha(
    X_processed,
    y,
    fm_name,
    alpha_range,
    kernel_type="fidelity",
    test_size=0.33,
    C=1.0,
    n_runs=5,
    scoring="roc_auc",
):
    """Study the impact of bandwidth α on a single kernel type.

    Args:
        X_processed: Preprocessed feature array (n_samples, n_qubits).
        y: Labels.
        fm_name: Feature map name (e.g., 'ZZ').
        alpha_range: List of α values to test.
        kernel_type: 'fidelity' or 'projected'.

    Returns:
        Dict {alpha: {'mean': float, 'std': float}}.
    """
    from ..kernels.feature_maps import build_feature_map
    from ..kernels.quantum_kernel import build_quantum_kernel
    from ..kernels.kernel_matrix import compute_kernel_matrix

    n_qubits = X_processed.shape[1]
    n_full = len(y)
    results = {}

    for alpha in alpha_range:
        print(f"  Testing alpha={alpha}...")
        # Build kernel once on full dataset
        fm = build_feature_map(fm_name, n_qubits, alpha=alpha, reps=1)
        qk = build_quantum_kernel(fm, kernel_type=kernel_type)
        K_full = compute_kernel_matrix(qk, X_processed)

        scores = []
        for seed in range(n_runs):
            idx = np.arange(n_full)
            idx_tr, idx_te = train_test_split(
                idx, test_size=test_size, random_state=seed, stratify=y
            )
            y_tr, y_te = y[idx_tr], y[idx_te]

            K_tr = K_full[np.ix_(idx_tr, idx_tr)]
            K_te = K_full[np.ix_(idx_te, idx_tr)]

            min_eig = np.min(np.linalg.eigvalsh(K_tr))
            if min_eig < 0:
                K_tr += (abs(min_eig) + 1e-10) * np.eye(K_tr.shape[0])

            svm = SVC(kernel="precomputed", C=C, probability=(scoring == "roc_auc"))
            svm.fit(K_tr, y_tr)

            if scoring == "roc_auc":
                score = roc_auc_score(y_te, svm.predict_proba(K_te)[:, 1])
            else:
                score = (svm.predict(K_te) == y_te).mean()

            scores.append(score)

        results[alpha] = {
            "mean": np.mean(scores),
            "std": np.std(scores, ddof=1),
        }
        print(f"    ROC-AUC: {results[alpha]['mean']:.4f} ± {results[alpha]['std']:.4f}")

    return results


def _frobenius_alignment(K1, K2):
    """Simple Frobenius alignment score (used internally)."""
    num = np.sum(K1 * K2)
    denom = np.sqrt(np.sum(K1 * K1) * np.sum(K2 * K2))
    return num / denom if denom > 0 else 0.0
