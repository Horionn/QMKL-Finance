"""Statistical analysis tools for QMKL experiments.

Implements:
- Multi-run evaluation with random seeds (as in IBM paper: 20 draws)
- Bootstrap confidence intervals
- Paired statistical tests (t-test, Wilcoxon)
- Effect size (Cohen's d)
- Win/tie/loss tables (as in IBM paper tables II-IV)
"""

import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


# ──────────────────────────────────────────────
# 1. MULTI-RUN EVALUATION
# ──────────────────────────────────────────────

def multi_run_evaluation(
    kernel_matrices_full,   # List of K matrices on FULL dataset
    y_full,                 # Full labels
    methods,                # Dict: {name: combiner_or_func}
    n_runs=20,
    test_size=0.33,
    C=1.0,
    scoring="roc_auc",
    random_seeds=None,
):
    """Evaluate multiple MKL methods over N random train/test splits.

    Replicates the IBM paper methodology: 20 random draws of 400 points,
    each split 67/33 train/test. Reports mean ± std for each method.

    Args:
        kernel_matrices_full: List of precomputed kernel matrices (n_full, n_full).
        y_full: Full label array.
        methods: Dict {method_name: callable(K_list_train, y_train) -> weights}.
        n_runs: Number of random splits (20 in IBM paper).
        test_size: Fraction for test split.
        C: SVM regularization.
        scoring: 'roc_auc', 'accuracy', or 'f1'.
        random_seeds: List of seeds. If None, uses range(n_runs).

    Returns:
        results: Dict {method_name: list of scores (length n_runs)}.
    """
    if random_seeds is None:
        random_seeds = list(range(n_runs))

    n_full = len(y_full)
    results = {name: [] for name in methods}

    for seed in random_seeds:
        # Random train/test split
        idx = np.arange(n_full)
        idx_train, idx_test = train_test_split(
            idx, test_size=test_size, random_state=seed, stratify=y_full
        )

        y_train = y_full[idx_train]
        y_test  = y_full[idx_test]

        # Slice kernel matrices
        K_list_train = [K[np.ix_(idx_train, idx_train)] for K in kernel_matrices_full]
        K_list_test  = [K[np.ix_(idx_test,  idx_train)] for K in kernel_matrices_full]

        for name, weight_fn in methods.items():
            # Compute weights using only train data
            weights = weight_fn(K_list_train, y_train)
            weights = np.array(weights)
            if weights.sum() > 0:
                weights = weights / weights.sum()

            # Combine kernels
            K_tr = sum(w * K for w, K in zip(weights, K_list_train))
            K_te = sum(w * K for w, K in zip(weights, K_list_test))

            # Ensure PSD
            min_eig = np.min(np.linalg.eigvalsh(K_tr))
            if min_eig < 0:
                K_tr += (abs(min_eig) + 1e-10) * np.eye(K_tr.shape[0])

            # Train and evaluate SVM
            svm = SVC(kernel="precomputed", C=C, probability=(scoring == "roc_auc"))
            svm.fit(K_tr, y_train)

            if scoring == "roc_auc":
                score = roc_auc_score(y_test, svm.predict_proba(K_te)[:, 1])
            elif scoring == "accuracy":
                score = accuracy_score(y_test, svm.predict(K_te))
            elif scoring == "f1":
                score = f1_score(y_test, svm.predict(K_te), zero_division=0)
            else:
                raise ValueError(f"Unknown scoring: {scoring}")

            results[name].append(score)

    return results


def summarize_multi_run(results):
    """Compute mean, std, min, max for each method.

    Args:
        results: Output of multi_run_evaluation.

    Returns:
        Dict {method_name: {'mean', 'std', 'min', 'max', 'n'}}.
    """
    summary = {}
    for name, scores in results.items():
        arr = np.array(scores)
        summary[name] = {
            "mean":   np.mean(arr),
            "std":    np.std(arr, ddof=1),
            "min":    np.min(arr),
            "max":    np.max(arr),
            "median": np.median(arr),
            "n":      len(arr),
        }
    return summary


# ──────────────────────────────────────────────
# 2. WIN / TIE / LOSS TABLE (IBM paper style)
# ──────────────────────────────────────────────

def win_tie_loss_table(results, baseline_method, threshold=0.0):
    """Count how many times each method beats the baseline.

    Replicates IBM paper Tables II-IV format.

    Args:
        results: Dict {method_name: list of scores}.
        baseline_method: Name of the baseline method to compare against.
        threshold: Minimum difference to count as a win (default: any positive diff).

    Returns:
        DataFrame-ready dict with wins, ties, losses for each method.
    """
    baseline_scores = np.array(results[baseline_method])
    table = {}

    for name, scores in results.items():
        if name == baseline_method:
            continue
        arr = np.array(scores)
        diff = arr - baseline_scores
        wins   = int(np.sum(diff >  threshold))
        ties   = int(np.sum(np.abs(diff) <= threshold))
        losses = int(np.sum(diff < -threshold))
        table[name] = {"wins": wins, "ties": ties, "losses": losses,
                       "win_rate": wins / len(scores)}

    return table


# ──────────────────────────────────────────────
# 3. BOOTSTRAP CONFIDENCE INTERVALS
# ──────────────────────────────────────────────

def bootstrap_ci(scores, n_bootstrap=2000, ci=0.95, random_state=42):
    """Compute bootstrap confidence interval for the mean.

    Args:
        scores: Array of observed scores (one per run).
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (0.95 = 95% CI).
        random_state: Random seed.

    Returns:
        Dict with 'mean', 'ci_low', 'ci_high', 'ci_width'.
    """
    rng = np.random.RandomState(random_state)
    arr = np.array(scores)
    boot_means = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means[i] = np.mean(sample)

    alpha = (1 - ci) / 2
    ci_low  = np.percentile(boot_means, 100 * alpha)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha))

    return {
        "mean":     np.mean(arr),
        "ci_low":   ci_low,
        "ci_high":  ci_high,
        "ci_width": ci_high - ci_low,
    }


def bootstrap_ci_all(results, **kwargs):
    """Apply bootstrap CI to all methods in a results dict."""
    return {name: bootstrap_ci(scores, **kwargs) for name, scores in results.items()}


# ──────────────────────────────────────────────
# 4. STATISTICAL TESTS
# ──────────────────────────────────────────────

def pairwise_ttest(results, alternative="greater"):
    """Paired t-test between all pairs of methods.

    Paired because each run uses the same data split.

    Args:
        results: Dict {method_name: list of scores}.
        alternative: 'greater' (A better than B?), 'two-sided', 'less'.

    Returns:
        Dict of dicts: pvalue[A][B] = p-value of "A > B".
    """
    names = list(results.keys())
    pvalues = {n: {} for n in names}

    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            if i == j:
                pvalues[name_a][name_b] = 1.0
                continue
            a = np.array(results[name_a])
            b = np.array(results[name_b])
            _, p = stats.ttest_rel(a, b, alternative=alternative)
            pvalues[name_a][name_b] = p

    return pvalues


def pairwise_wilcoxon(results, alternative="greater"):
    """Paired Wilcoxon signed-rank test (non-parametric alternative to t-test).

    More robust when distributions are not normal (often the case for AUC scores).
    """
    names = list(results.keys())
    pvalues = {n: {} for n in names}

    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            if i == j:
                pvalues[name_a][name_b] = 1.0
                continue
            a = np.array(results[name_a])
            b = np.array(results[name_b])
            diff = a - b
            if np.all(diff == 0):
                pvalues[name_a][name_b] = 1.0
            else:
                _, p = stats.wilcoxon(diff, alternative=alternative)
                pvalues[name_a][name_b] = p

    return pvalues


def cohens_d(scores_a, scores_b):
    """Compute Cohen's d effect size between two sets of scores.

    d = (mean_A - mean_B) / pooled_std

    Interpretation:
        |d| < 0.2  : negligible
        |d| < 0.5  : small
        |d| < 0.8  : medium
        |d| >= 0.8 : large
    """
    a, b = np.array(scores_a), np.array(scores_b)
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


# ──────────────────────────────────────────────
# 5. KERNEL WEIGHT STABILITY ANALYSIS
# ──────────────────────────────────────────────

def weight_stability_analysis(
    kernel_matrices_full,
    y_full,
    weight_fn,
    kernel_names,
    n_runs=20,
    test_size=0.33,
):
    """Analyze how stable kernel weights are across random splits.

    If weights vary a lot between runs, the selection is unreliable.
    If weights are stable, the selected kernels are genuinely important.

    Args:
        weight_fn: Callable(K_list_train, y_train) -> weights array.
        kernel_names: List of kernel names.
        n_runs, test_size: Same as multi_run_evaluation.

    Returns:
        weights_matrix: (n_runs, n_kernels) array.
        summary: Mean ± std for each kernel weight.
    """
    n_kernels = len(kernel_matrices_full)
    n_full = len(y_full)
    weights_matrix = np.zeros((n_runs, n_kernels))

    for run in range(n_runs):
        idx = np.arange(n_full)
        idx_train, _ = train_test_split(
            idx, test_size=test_size, random_state=run, stratify=y_full
        )
        y_train = y_full[idx_train]
        K_list_train = [K[np.ix_(idx_train, idx_train)] for K in kernel_matrices_full]

        w = weight_fn(K_list_train, y_train)
        w = np.array(w)
        if w.sum() > 0:
            w = w / w.sum()
        weights_matrix[run] = w

    summary = {
        name: {
            "mean": np.mean(weights_matrix[:, i]),
            "std":  np.std(weights_matrix[:, i], ddof=1),
            "max":  np.max(weights_matrix[:, i]),
            "nonzero_rate": np.mean(weights_matrix[:, i] > 0.01),
        }
        for i, name in enumerate(kernel_names)
    }

    return weights_matrix, summary
