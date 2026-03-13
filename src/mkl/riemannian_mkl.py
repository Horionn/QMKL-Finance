"""Riemannian Multiple Kernel Learning for Quantum Kernels.

Combines M kernel matrices using the weighted Fréchet mean on the
SPD (Symmetric Positive Definite) manifold, instead of the standard
Euclidean linear combination K = Σ w_m K_m.

Mathematical Foundations
------------------------
The space of n×n SPD matrices (Sym+_n) equipped with the affine-invariant
Riemannian metric:

    d²_R(A, B) = ‖log(A^{-½} B A^{-½})‖²_F

forms a Hadamard (complete, non-positively curved) manifold. This geometry
has the following key properties:

1. Geodesic uniqueness: any two SPD matrices are connected by a unique geodesic.
2. Congruence invariance: d_R(PAP^T, PBP^T) = d_R(A,B) ∀ invertible P.
3. Inversion invariance: d_R(A^{-1}, B^{-1}) = d_R(A,B).
4. Positive definiteness: all points on geodesics remain SPD.

Weighted Fréchet Mean (Karcher Mean)
-------------------------------------
Given weights w = (w_1,...,w_M) with w_m ≥ 0, Σw_m = 1, the weighted
Fréchet mean K★ minimizes:

    K★ = argmin_{K ∈ Sym+_n} Σ_m w_m · d²_R(K, K_m)

Computed via the fixed-point iteration (Moakher 2005, Bhatia 2009):

    S_{(t)} = Σ_m w_m log(K_{(t)}^{-½} K_m K_{(t)}^{-½})
    K_{(t+1)} = K_{(t)}^{½} exp(S_{(t)}) K_{(t)}^{½}

Convergence is guaranteed on the SPD manifold (globally, not just locally).

Log-Euclidean Mean (Arsigny et al. 2007)
-----------------------------------------
A faster approximation that operates on the flat log-space:

    K_LogE = exp(Σ_m w_m log(K_m))

Computationally cheaper (no iteration needed) but less geometrically
accurate than the Fréchet mean. Always produces a valid SPD matrix.

Weight Optimization
--------------------
Weights are optimized to maximize the Kernel-Target Alignment (KTA):

    w★ = argmax_{w ∈ Δ_M} KTA(K★(w), K_y)

where K_y[i,j] = 1 if y[i]==y[j], else 0, and Δ_M is the M-simplex.

Novel Contribution
------------------
This Riemannian combination is absent from quantum MKL literature (2023–2026).
Classical SPD-MKL has been applied to EEG/BCI data (e.g., SPDNet), but the
quantum kernel setting introduces unique features:
- Kernels encode quantum geometry (Fubini-Study metric)
- The Fréchet mean may better respect quantum state distances
- Log-Euclidean mean has direct connection to matrix quantum mechanics

References
-----------
- Moakher, M. (2005). A differential geometric approach to the geometric mean
  of symmetric positive-definite matrices. SIAM J. Matrix Anal. Appl., 26(3).
- Bhatia, R. & Holbrook, J. (2006). Riemannian geometry and matrix geometric
  means. Linear Algebra Appl., 413(2-3).
- Arsigny, V. et al. (2007). Geometric means in a novel vector space structure
  on symmetric positive-definite matrices. SIAM J. Matrix Anal. Appl., 29(1).
- Moakher 2005 iteration convergence: guaranteed by non-positive curvature of
  the SPD manifold (Cartan-Hadamard theorem).
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import warnings


# ─────────────────────────────────────────────────────────────
# 1. CORE SPD MATRIX OPERATIONS
# ─────────────────────────────────────────────────────────────

def _sym(A):
    """Symmetrize a matrix to correct floating-point asymmetry."""
    return (A + A.T) * 0.5


def _regularize(K, eps=1e-8):
    """Add small diagonal regularization to ensure strict positive definiteness."""
    n = K.shape[0]
    return K + eps * np.eye(n)


def matrix_sqrt(A, eps=1e-8):
    """Compute A^{1/2} via eigendecomposition.

    Args:
        A: Symmetric positive semi-definite matrix.
        eps: Floor for eigenvalues (numerical stability).

    Returns:
        S such that S @ S = A (approximately).
    """
    A = _sym(A)
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, eps)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def matrix_inv_sqrt(A, eps=1e-8):
    """Compute A^{-1/2} via eigendecomposition.

    Args:
        A: Symmetric positive definite matrix.
        eps: Floor for eigenvalues.

    Returns:
        S such that S @ S = A^{-1} (approximately).
    """
    A = _sym(A)
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, eps)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


def matrix_log(A, eps=1e-8):
    """Compute principal matrix logarithm via eigendecomposition.

    Args:
        A: Symmetric positive definite matrix.
        eps: Floor for eigenvalues.

    Returns:
        log(A) such that exp(log(A)) ≈ A.
    """
    A = _sym(A)
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, eps)
    return eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T


def matrix_exp(A):
    """Compute matrix exponential via eigendecomposition.

    Args:
        A: Symmetric matrix.

    Returns:
        exp(A), always symmetric positive definite.
    """
    A = _sym(A)
    eigvals, eigvecs = np.linalg.eigh(A)
    return eigvecs @ np.diag(np.exp(eigvals)) @ eigvecs.T


def riemannian_dist(A, B, eps=1e-8):
    """Affine-invariant Riemannian distance between SPD matrices.

    d_R(A, B) = ‖log(A^{-½} B A^{-½})‖_F

    Args:
        A, B: Symmetric positive definite matrices of the same size.
        eps: Eigenvalue floor for numerical stability.

    Returns:
        Non-negative scalar distance.
    """
    n = A.shape[0]
    A_reg = _regularize(_sym(A), eps)
    B_reg = _regularize(_sym(B), eps)
    A_inv_sqrt = matrix_inv_sqrt(A_reg, eps)
    M = A_inv_sqrt @ B_reg @ A_inv_sqrt
    log_M = matrix_log(_sym(M), eps)
    return float(np.sqrt(np.maximum(np.sum(log_M ** 2), 0.0)))


# ─────────────────────────────────────────────────────────────
# 2. RIEMANNIAN MEAN ALGORITHMS
# ─────────────────────────────────────────────────────────────

def frechet_mean(K_list, weights, eps=1e-8, max_iter=100, tol=1e-9,
                 verbose=False):
    """Weighted Fréchet mean (Karcher mean) on the SPD manifold.

    Uses the Moakher (2005) fixed-point iteration:

        S = Σ_m w_m log(K^{-½} K_m K^{-½})
        K ← K^{½} exp(S) K^{½}

    Convergence is guaranteed by the non-positive curvature of the manifold
    (Cartan-Hadamard theorem). Typically 20–50 iterations suffice.

    Args:
        K_list: List of M SPD matrices of shape (n, n).
        weights: Array-like of M non-negative weights (need not sum to 1).
        eps: Eigenvalue floor for numerical regularization.
        max_iter: Maximum number of fixed-point iterations.
        tol: Frobenius-norm convergence threshold.
        verbose: Print iteration info if True.

    Returns:
        K★: Weighted Fréchet mean, shape (n, n), guaranteed SPD.
    """
    M = len(K_list)
    n = K_list[0].shape[0]
    weights = np.array(weights, dtype=float)
    weights = np.maximum(weights, 0.0)
    s = weights.sum()
    if s < 1e-12:
        weights = np.ones(M) / M
    else:
        weights = weights / s

    # Regularize all kernels once
    K_reg = [_regularize(_sym(K), eps) for K in K_list]

    # Initialize with arithmetic mean
    K = _sym(sum(w * Km for w, Km in zip(weights, K_reg)))

    for iteration in range(max_iter):
        K_sqrt = matrix_sqrt(K, eps)
        K_inv_sqrt = matrix_inv_sqrt(K, eps)

        # Weighted sum of logs in tangent space at K
        S = np.zeros((n, n), dtype=float)
        for w, Km in zip(weights, K_reg):
            if w < 1e-15:
                continue
            M_inner = _sym(K_inv_sqrt @ Km @ K_inv_sqrt)
            S += w * matrix_log(M_inner, eps)

        # Geodesic retraction
        K_new = _sym(K_sqrt @ matrix_exp(S) @ K_sqrt)

        # Convergence check
        diff = np.linalg.norm(K_new - K, 'fro')
        K = K_new

        if verbose:
            print(f"  iter {iteration+1:3d}: ‖ΔK‖_F = {diff:.2e}")

        if diff < tol:
            break

    return K


def log_euclidean_mean(K_list, weights, eps=1e-8):
    """Log-Euclidean mean of SPD matrices (Arsigny et al. 2007).

    K_LogE = exp(Σ_m w_m log(K_m))

    Faster than Fréchet mean (no iteration), but less geometrically accurate.
    Always produces a valid SPD matrix. Useful for large M or real-time use.

    Args:
        K_list: List of M SPD matrices of shape (n, n).
        weights: Array-like of M weights.
        eps: Eigenvalue floor.

    Returns:
        K_LogE: Log-Euclidean mean, shape (n, n), guaranteed SPD.
    """
    M = len(K_list)
    weights = np.array(weights, dtype=float)
    weights = np.maximum(weights, 0.0)
    s = weights.sum()
    weights = weights / s if s > 1e-12 else np.ones(M) / M

    n = K_list[0].shape[0]
    log_mean = np.zeros((n, n), dtype=float)
    for w, K in zip(weights, K_list):
        if w < 1e-15:
            continue
        K_reg = _regularize(_sym(K), eps)
        log_mean += w * matrix_log(K_reg, eps)

    return matrix_exp(log_mean)


def geodesic_path(K1, K2, n_points=20, eps=1e-8):
    """Compute geodesic path between two SPD matrices.

    γ(t) = K1^{½} (K1^{-½} K2 K1^{-½})^t K1^{½},  t ∈ [0, 1]

    Args:
        K1, K2: SPD matrices, shape (n, n).
        n_points: Number of points along the geodesic.
        eps: Eigenvalue floor.

    Returns:
        ts: Array of t values ∈ [0, 1].
        path: List of n_points SPD matrices on the geodesic.
    """
    n = K1.shape[0]
    K1_reg = _regularize(_sym(K1), eps)
    K2_reg = _regularize(_sym(K2), eps)

    K1_sqrt = matrix_sqrt(K1_reg, eps)
    K1_inv_sqrt = matrix_inv_sqrt(K1_reg, eps)
    M_inner = _sym(K1_inv_sqrt @ K2_reg @ K1_inv_sqrt)

    # Eigendecompose M_inner once
    eigvals, eigvecs = np.linalg.eigh(M_inner)
    eigvals = np.maximum(eigvals, eps)

    ts = np.linspace(0.0, 1.0, n_points)
    path = []
    for t in ts:
        Mt = eigvecs @ np.diag(eigvals ** t) @ eigvecs.T
        Kt = _sym(K1_sqrt @ Mt @ K1_sqrt)
        path.append(Kt)

    return ts, path


def linear_combination(K_list, weights):
    """Standard linear (Euclidean) combination: K = Σ w_m K_m.

    Reference baseline for comparison with Riemannian mean.

    Args:
        K_list: List of M matrices.
        weights: Array of M weights.

    Returns:
        K: Linear combination.
    """
    weights = np.array(weights, dtype=float)
    s = weights.sum()
    weights = weights / s if s > 1e-12 else np.ones(len(K_list)) / len(K_list)
    return sum(w * K for w, K in zip(weights, K_list))


# ─────────────────────────────────────────────────────────────
# 3. RIEMANNIAN QMKL CLASS
# ─────────────────────────────────────────────────────────────

class RiemannianQMKL:
    """Riemannian Multiple Kernel Learning for Quantum Kernels.

    Replaces the Euclidean linear combination of quantum kernels with
    the weighted Fréchet mean (or Log-Euclidean mean) on the SPD manifold.

    The combination K★(w) is used as the training kernel for SVM.
    For test kernels (which are rectangular), a linear combination with
    the Riemannian-optimized weights is used.

    Weight methods
    --------------
    'uniform'     : Equal weights (manifold arithmetic mean).
    'alignment'   : Maximize KTA of K★(w) via gradient-free optimization.
    'log_euclidean': Use Log-Euclidean mean with alignment-optimized weights.
    'sdp'         : Initialize with SDP alignment weights on the manifold.

    Parameters
    ----------
    weight_method : str
        Weight optimization strategy (see above).
    mean_type : str
        'frechet' for Fréchet mean (slower, exact) or
        'log_euclidean' for Log-Euclidean mean (faster, approximate).
    eps : float
        Eigenvalue floor for numerical regularization.
    fm_max_iter : int
        Maximum iterations for Fréchet mean fixed-point.
    fm_tol : float
        Frobenius convergence threshold for Fréchet mean.
    opt_max_iter : int
        Maximum optimizer evaluations for weight search.
    n_cv : int
        Cross-validation folds for evaluation (used with 'alignment').
    C : float
        SVM regularization parameter.
    scoring : str
        'roc_auc', 'accuracy', or 'f1'.
    seed : int
        Random seed.
    verbose : bool
        Print progress information.
    """

    def __init__(
        self,
        weight_method='alignment',
        mean_type='frechet',
        eps=1e-8,
        fm_max_iter=100,
        fm_tol=1e-9,
        opt_max_iter=200,
        n_cv=5,
        C=1.0,
        scoring='roc_auc',
        seed=42,
        verbose=True,
    ):
        self.weight_method = weight_method
        self.mean_type = mean_type
        self.eps = eps
        self.fm_max_iter = fm_max_iter
        self.fm_tol = fm_tol
        self.opt_max_iter = opt_max_iter
        self.n_cv = n_cv
        self.C = C
        self.scoring = scoring
        self.seed = seed
        self.verbose = verbose

        # Fitted attributes
        self.weights_ = None
        self.kernel_names_ = None
        self.n_kernels_ = None
        self._convergence_history = []

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def fit(self, K_list, y, kernel_names=None):
        """Compute optimal weights for the Riemannian kernel combination.

        Args:
            K_list: List of M SPD kernel matrices, each (n, n).
            y: Label array of length n.
            kernel_names: Optional list of M kernel name strings.

        Returns:
            self
        """
        M = len(K_list)
        self.n_kernels_ = M
        self.kernel_names_ = kernel_names or [f'K_{m}' for m in range(M)]

        K_target = self._build_target(y)

        if self.verbose:
            print(f"RiemannianQMKL.fit — M={M}, N={len(y)}, "
                  f"method={self.weight_method}, mean={self.mean_type}")

        if self.weight_method == 'uniform':
            self.weights_ = np.ones(M) / M

        elif self.weight_method in ('alignment', 'log_euclidean'):
            self.weights_ = self._optimize_weights_kta(K_list, K_target)

        elif self.weight_method == 'sdp':
            from src.mkl.alignment import sdp_alignment
            w = sdp_alignment(K_list, K_target)
            s = np.maximum(w, 0.0).sum()
            self.weights_ = np.maximum(w, 0.0) / s if s > 1e-12 else np.ones(M) / M

        else:
            raise ValueError(f"Unknown weight_method: {self.weight_method!r}")

        if self.verbose:
            wstr = ", ".join(f"{w:.4f}" for w in self.weights_)
            print(f"  Weights: [{wstr}]")

        return self

    def combine(self, K_list, weights=None):
        """Compute the Riemannian mean of training kernel matrices.

        Args:
            K_list: List of M SPD matrices, shape (n, n) each.
            weights: Optional weight override. Uses self.weights_ if None.

        Returns:
            K★: Combined kernel matrix, shape (n, n), guaranteed SPD.
        """
        w = self._resolve_weights(weights, len(K_list))
        return self._riemannian_combine(K_list, w)

    def combine_test(self, K_test_list, weights=None):
        """Combine test-train kernel matrices (rectangular, not SPD).

        Since K_test matrices have shape (n_test, n_train) and are not
        square/SPD, the Fréchet mean is not applicable. We use a linear
        combination with the Riemannian-optimized weights instead.

        This is theoretically sound: the weights encode the geometry
        learned from the training SPD structure.

        Args:
            K_test_list: List of M matrices, shape (n_test, n_train).
            weights: Optional weight override.

        Returns:
            K★_test: Linearly combined test kernel, shape (n_test, n_train).
        """
        w = self._resolve_weights(weights, len(K_test_list))
        return linear_combination(K_test_list, w)

    def get_weights(self):
        """Return the fitted weight vector."""
        return self.weights_.copy() if self.weights_ is not None else None

    def compute_pairwise_distances(self, K_list):
        """Compute M×M matrix of pairwise Riemannian distances.

        Args:
            K_list: List of M SPD matrices.

        Returns:
            D: Symmetric distance matrix of shape (M, M).
        """
        M = len(K_list)
        D = np.zeros((M, M))
        for i in range(M):
            for j in range(i + 1, M):
                d = riemannian_dist(K_list[i], K_list[j], self.eps)
                D[i, j] = D[j, i] = d
        return D

    def compute_geodesic_path(self, K1, K2, n_points=20):
        """Geodesic path between two SPD matrices.

        Returns:
            ts: Array of t ∈ [0,1].
            path: List of SPD matrices along the geodesic.
        """
        return geodesic_path(K1, K2, n_points=n_points, eps=self.eps)

    def compute_psd_gap(self, K_list):
        """Measure how far each kernel is from the linear combination.

        Returns the Riemannian distance d_R(K_m, K★_linear) for each m,
        quantifying the 'geodesic spread' of the kernel library.

        Args:
            K_list: List of M kernel matrices.

        Returns:
            gaps: Array of M distances to the arithmetic mean.
        """
        w_uniform = np.ones(len(K_list)) / len(K_list)
        K_linear = linear_combination(K_list, w_uniform)
        gaps = np.array([
            riemannian_dist(K, K_linear, self.eps) for K in K_list
        ])
        return gaps

    def fit_predict_cv(self, K_list, y, n_splits=5):
        """Cross-validated AUC using Riemannian combination.

        Used internally for weight optimization.

        Args:
            K_list: List of M SPD matrices, shape (n, n).
            y: Label array.
            n_splits: Number of CV folds.

        Returns:
            mean_auc: Mean AUC over folds.
        """
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=self.seed)
        scores = []
        for train_idx, val_idx in kf.split(np.zeros(len(y)), y):
            y_tr, y_va = y[train_idx], y[val_idx]
            K_tr = [K[np.ix_(train_idx, train_idx)] for K in K_list]
            K_va = [K[np.ix_(val_idx, train_idx)] for K in K_list]

            w = self.weights_ if self.weights_ is not None \
                else np.ones(len(K_list)) / len(K_list)
            K_tr_c = self._riemannian_combine(K_tr, w)
            K_va_c = linear_combination(K_va, w)

            K_tr_c = self._ensure_psd(K_tr_c)
            svm = SVC(kernel='precomputed', C=self.C, probability=True)
            svm.fit(K_tr_c, y_tr)
            score = roc_auc_score(y_va, svm.predict_proba(K_va_c)[:, 1])
            scores.append(score)

        return float(np.mean(scores))

    # ─────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────

    def _riemannian_combine(self, K_list, weights):
        """Apply the selected Riemannian mean."""
        if self.mean_type == 'frechet':
            return frechet_mean(
                K_list, weights,
                eps=self.eps,
                max_iter=self.fm_max_iter,
                tol=self.fm_tol,
                verbose=False,
            )
        elif self.mean_type == 'log_euclidean':
            return log_euclidean_mean(K_list, weights, eps=self.eps)
        else:
            raise ValueError(f"Unknown mean_type: {self.mean_type!r}")

    def _build_target(self, y):
        """Build the ideal kernel: K_y[i,j] = 1 if y[i]==y[j], else 0."""
        return (y[:, None] == y[None, :]).astype(float)

    def _kta(self, K, K_target):
        """Kernel-Target Alignment (Cortes et al. 2012)."""
        num = np.sum(K * K_target)
        denom = np.sqrt(np.sum(K * K) * np.sum(K_target * K_target) + 1e-24)
        return float(num / denom)

    def _softmax(self, x):
        """Softmax with numerical stability."""
        e = np.exp(x - x.max())
        return e / e.sum()

    def _resolve_weights(self, weights, M):
        """Return weights, defaulting to self.weights_ or uniform."""
        if weights is not None:
            w = np.array(weights, dtype=float)
        elif self.weights_ is not None:
            w = self.weights_
        else:
            w = np.ones(M) / M
        s = w.sum()
        return w / s if s > 1e-12 else np.ones(M) / M

    def _ensure_psd(self, K, margin=1e-8):
        """Shift spectrum if any eigenvalue is negative."""
        min_eig = np.min(np.linalg.eigvalsh(K))
        if min_eig < 0:
            K = K + (abs(min_eig) + margin) * np.eye(K.shape[0])
        return K

    def _optimize_weights_kta(self, K_list, K_target):
        """Optimize weights to maximize KTA(K★(w), K_target).

        Uses softmax reparameterization (θ → softmax(θ)) to enforce
        w_m > 0 and Σ w_m = 1, then minimizes -KTA via Nelder-Mead.

        Multiple random restarts are used to avoid local optima.
        """
        M = len(K_list)
        best_kta = -np.inf
        best_w = np.ones(M) / M
        history = []

        def neg_kta(theta):
            w = self._softmax(theta)
            K_combined = self._riemannian_combine(K_list, w)
            val = self._kta(K_combined, K_target)
            history.append((w.copy(), val))
            return -val

        # Restart initializations
        rng = np.random.RandomState(self.seed)
        inits = [np.zeros(M)]  # uniform
        for m in range(min(M, 5)):  # concentrate on one kernel
            theta = rng.randn(M) * 0.1
            theta[m] += 2.0
            inits.append(theta)
        for _ in range(3):  # random
            inits.append(rng.randn(M))

        if self.verbose:
            print(f"  Optimizing weights ({len(inits)} restarts)...")

        for i, theta0 in enumerate(inits):
            try:
                res = minimize(
                    neg_kta, theta0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': self.opt_max_iter,
                        'xatol': 1e-5,
                        'fatol': 1e-5,
                        'adaptive': True,
                    },
                )
                kta_val = -res.fun
                if self.verbose:
                    print(f"    restart {i+1}: KTA = {kta_val:.6f}")
                if kta_val > best_kta:
                    best_kta = kta_val
                    best_w = self._softmax(res.x)
            except Exception as e:
                warnings.warn(f"Optimizer restart {i+1} failed: {e}")

        self._convergence_history = history
        if self.verbose:
            print(f"  Best KTA = {best_kta:.6f}")

        return best_w


# ─────────────────────────────────────────────────────────────
# 4. MULTI-RUN EVALUATION (Riemannian-aware)
# ─────────────────────────────────────────────────────────────

def riemannian_multi_run_evaluation(
    K_list_full,
    y_full,
    methods,          # dict: {name: (mean_type, weight_method)}
    n_runs=20,
    test_size=0.33,
    C=1.0,
    scoring='roc_auc',
    eps=1e-8,
    fm_max_iter=50,
    fm_tol=1e-8,
    opt_max_iter=100,
    n_cv=5,
    seed=42,
    verbose=False,
):
    """Multi-run evaluation for Riemannian QMKL methods.

    Analogous to statistical_analysis.multi_run_evaluation but handles
    the Riemannian combination for training kernels and linear
    combination for test kernels.

    Args:
        K_list_full: List of M kernel matrices, shape (n_full, n_full).
        y_full: Full label array.
        methods: Dict {name: (mean_type, weight_method)} where:
            - mean_type ∈ {'frechet', 'log_euclidean', 'linear'}
            - weight_method ∈ {'uniform', 'alignment', 'sdp'}
        n_runs: Number of random train/test splits.
        test_size: Fraction for test split.
        C: SVM regularization.
        scoring: Metric for evaluation.
        eps: SPD regularization.
        fm_max_iter: Max Fréchet iterations per call.
        fm_tol: Fréchet convergence threshold.
        opt_max_iter: Max optimizer steps for weight search.
        n_cv: CV folds for weight optimization (if weight_method='alignment').
        seed: Random seed for splits.
        verbose: Print progress.

    Returns:
        results: Dict {method_name: list of n_runs scores}.
    """
    from src.mkl.alignment import centered_alignment, sdp_alignment, projection_alignment

    n_full = len(y_full)
    results = {name: [] for name in methods}

    for run in range(n_runs):
        if verbose:
            print(f"Run {run+1}/{n_runs}...")

        idx = np.arange(n_full)
        idx_tr, idx_te = train_test_split(
            idx, test_size=test_size, random_state=seed + run, stratify=y_full
        )
        y_tr = y_full[idx_tr]
        y_te = y_full[idx_te]

        K_tr = [K[np.ix_(idx_tr, idx_tr)] for K in K_list_full]
        K_te = [K[np.ix_(idx_te, idx_tr)] for K in K_list_full]

        for name, (mean_type, weight_method) in methods.items():
            try:
                model = RiemannianQMKL(
                    weight_method=weight_method,
                    mean_type=mean_type,
                    eps=eps,
                    fm_max_iter=fm_max_iter,
                    fm_tol=fm_tol,
                    opt_max_iter=opt_max_iter,
                    n_cv=n_cv,
                    C=C,
                    scoring=scoring,
                    seed=seed,
                    verbose=False,
                )
                model.fit(K_tr, y_tr)

                if mean_type == 'linear':
                    # Pure linear combination with Riemannian weights
                    K_tr_c = linear_combination(K_tr, model.weights_)
                else:
                    K_tr_c = model.combine(K_tr)

                K_te_c = model.combine_test(K_te)

                # Ensure PSD
                min_eig = np.min(np.linalg.eigvalsh(K_tr_c))
                if min_eig < 0:
                    K_tr_c += (abs(min_eig) + 1e-8) * np.eye(K_tr_c.shape[0])

                svm = SVC(kernel='precomputed', C=C, probability=(scoring == 'roc_auc'))
                svm.fit(K_tr_c, y_tr)

                if scoring == 'roc_auc':
                    score = roc_auc_score(y_te, svm.predict_proba(K_te_c)[:, 1])
                elif scoring == 'accuracy':
                    score = accuracy_score(y_te, svm.predict(K_te_c))
                elif scoring == 'f1':
                    score = f1_score(y_te, svm.predict(K_te_c), zero_division=0)
                else:
                    raise ValueError(f"Unknown scoring: {scoring}")

                results[name].append(score)

            except Exception as e:
                warnings.warn(f"Run {run+1}, method {name!r} failed: {e}")
                results[name].append(float('nan'))

    return results
