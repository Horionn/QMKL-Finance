"""Kernel-target alignment optimization methods.

Implements three alignment strategies from the IBM QMKL paper:
1. SDP-based alignment
2. Centered alignment (Cortes et al., 2012)
3. Iterative projection-based alignment

Robustness features:
- Multi-solver cascade: CLARABEL → ECOS → SCS → closed-form fallback
- Automatic concentration detection: returns uniform weights for degenerate kernels
- Adaptive regularization scaled by matrix norm
"""

import warnings

import numpy as np


# ──────────────────────────────────────────────
# INTERNAL UTILITIES
# ──────────────────────────────────────────────

def _frobenius_inner(K1, K2):
    """Frobenius inner product between two matrices."""
    return float(np.sum(K1 * K2))


def _center_kernel(K):
    """Center a kernel matrix in feature space.

    K_c = (I - 11^T/m) K (I - 11^T/m)
    """
    m = K.shape[0]
    ones = np.ones((m, m)) / m
    centering = np.eye(m) - ones
    return centering @ K @ centering


def _is_concentrated(K, threshold=1e-6):
    """Return True if the kernel is exponentially concentrated.

    A concentrated kernel has near-zero off-diagonal variance — all values
    are approximately constant. This makes alignment optimization degenerate.
    """
    n = K.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off_diag = K[mask]
    if len(off_diag) == 0:
        return True
    return float(np.std(off_diag)) < threshold


def _uniform_weights(n):
    """Return uniform (equal) weights summing to 1."""
    return np.ones(n) / n


def _solve_qp_cvxpy(M, a, n_kernels):
    """Solve min v^T M v - 2 v^T a, s.t. v >= 0, using CVXPY.

    Tries solvers in order: CLARABEL → ECOS → SCS.
    Returns None if all solvers fail.
    """
    try:
        import cvxpy as cp
    except ImportError:
        return None

    v = cp.Variable(n_kernels)
    objective = cp.Minimize(cp.quad_form(v, cp.psd_wrap(M)) - 2 * a @ v)
    constraints = [v >= 0]
    prob = cp.Problem(objective, constraints)

    # Try solvers in order of robustness / speed
    for solver in [cp.CLARABEL, cp.ECOS, cp.SCS]:
        try:
            prob.solve(solver=solver, verbose=False)
            if v.value is not None and not np.any(np.isnan(v.value)):
                return np.maximum(v.value, 0.0)
        except Exception:
            continue

    return None  # all solvers failed


def _solve_sdp_cvxpy(q, S, n_kernels):
    """Solve max w^T q  s.t. w^T S w <= 1, w >= 0, using CVXPY.

    Tries solvers in order: CLARABEL → ECOS → SCS.
    Returns None if all solvers fail.
    """
    try:
        import cvxpy as cp
    except ImportError:
        return None

    w = cp.Variable(n_kernels)
    objective = cp.Maximize(w @ q)
    constraints = [
        cp.quad_form(w, cp.psd_wrap(S)) <= 1,
        w >= 0,
    ]
    prob = cp.Problem(objective, constraints)

    for solver in [cp.CLARABEL, cp.ECOS, cp.SCS]:
        try:
            prob.solve(solver=solver, verbose=False)
            if w.value is not None and not np.any(np.isnan(w.value)):
                return np.maximum(w.value, 0.0)
        except Exception:
            continue

    return None


def _closed_form_alignment(M, a):
    """Closed-form fallback: solve M v = a with non-negative clipping."""
    try:
        v = np.linalg.solve(M, a)
        return np.maximum(v, 0.0)
    except np.linalg.LinAlgError:
        # If singular, use least-squares
        v, _, _, _ = np.linalg.lstsq(M, a, rcond=None)
        return np.maximum(v, 0.0)


def _regularize_matrix(M, scale_factor=1e-6):
    """Add adaptive Tikhonov regularization scaled by the matrix norm."""
    norm = np.linalg.norm(M, "fro")
    reg = max(scale_factor * norm, 1e-10)
    return M + reg * np.eye(M.shape[0])


# ──────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────

def kernel_target_alignment(K, K_target):
    """Compute the kernel-target alignment score.

    A(K1, K2) = <K1, K2>_F / sqrt(<K1, K1>_F * <K2, K2>_F)
    """
    num = _frobenius_inner(K, K_target)
    denom = float(np.sqrt(_frobenius_inner(K, K) * _frobenius_inner(K_target, K_target)))
    if denom == 0.0:
        return 0.0
    return num / denom


# ── Strategy 1: SDP-based alignment ──────────────────────────────────────────

def sdp_alignment(K_list, K_target):
    """Optimize kernel weights using semidefinite programming.

    Solves: max w^T q  s.t. w^T S w <= 1, w >= 0
    where q_i = <K_i, K_target>_F and S_ij = <K_i, K_j>_F

    Falls back to uniform weights if all kernels are concentrated or
    if the solver fails.
    """
    n_kernels = len(K_list)

    # Concentration check: if all kernels are degenerate → uniform weights
    if all(_is_concentrated(K) for K in K_list):
        warnings.warn(
            "SDP alignment: all kernels are exponentially concentrated "
            "(near-constant values). Returning uniform weights.",
            RuntimeWarning, stacklevel=2,
        )
        return _uniform_weights(n_kernels)

    # Build q vector and S matrix
    q = np.array([_frobenius_inner(K, K_target) for K in K_list])
    S = np.zeros((n_kernels, n_kernels))
    for i in range(n_kernels):
        for j in range(n_kernels):
            S[i, j] = _frobenius_inner(K_list[i], K_list[j])

    S = _regularize_matrix(S)

    # Try CVXPY
    weights = _solve_sdp_cvxpy(q, S, n_kernels)

    # Fallback: closed-form
    if weights is None:
        warnings.warn(
            "SDP alignment: all CVXPY solvers failed. Using closed-form fallback.",
            RuntimeWarning, stacklevel=2,
        )
        weights = _closed_form_alignment(S, q)

    # Normalize
    total = weights.sum()
    if total > 0:
        weights = weights / total
    else:
        weights = _uniform_weights(n_kernels)

    return weights


# ── Strategy 2: Centered alignment ───────────────────────────────────────────

def centered_alignment(K_list, K_target):
    """Optimize kernel weights using centered kernel alignment.

    Centers all kernels before computing alignment, which better
    correlates with generalization performance (Cortes et al., 2012).

    Solves: min v^T M v - 2 v^T a  s.t. v >= 0
    where a_i = <K_i^c, K_target^c>_F and M_ij = <K_i^c, K_j^c>_F

    Falls back gracefully when kernels are concentrated or solvers fail.
    """
    n_kernels = len(K_list)

    # Center all kernels
    K_list_c = [_center_kernel(K) for K in K_list]
    K_target_c = _center_kernel(K_target)

    # Concentration check on centered kernels
    if all(_is_concentrated(Kc) for Kc in K_list_c):
        warnings.warn(
            "Centered alignment: all centered kernels are near-zero "
            "(exponential concentration). Returning uniform weights.",
            RuntimeWarning, stacklevel=2,
        )
        return _uniform_weights(n_kernels)

    # Build a vector and M matrix
    a = np.array([_frobenius_inner(Kc, K_target_c) for Kc in K_list_c])
    M = np.zeros((n_kernels, n_kernels))
    for i in range(n_kernels):
        for j in range(n_kernels):
            M[i, j] = _frobenius_inner(K_list_c[i], K_list_c[j])

    M = _regularize_matrix(M)

    # Try CVXPY (cascade: CLARABEL → ECOS → SCS)
    weights = _solve_qp_cvxpy(M, a, n_kernels)

    # Fallback: closed-form least-squares
    if weights is None:
        warnings.warn(
            "Centered alignment: all CVXPY solvers failed. Using closed-form fallback.",
            RuntimeWarning, stacklevel=2,
        )
        weights = _closed_form_alignment(M, a)

    # Normalize by L2 norm (as in Cortes et al.)
    norm = np.linalg.norm(weights)
    if norm > 0:
        weights = weights / norm
    else:
        weights = _uniform_weights(n_kernels)

    return weights


# ── Strategy 3: Iterative projection ─────────────────────────────────────────

def projection_alignment(K_list, K_target, threshold=1e-6):
    """Optimize kernel weights using iterative projection-based alignment.

    Custom method from the IBM QMKL paper:
    1. Start with K_target
    2. Find the kernel K closest to the residual
    3. Subtract its projection, update weights
    4. Repeat until residual stops decreasing

    This method is purely analytical — no solver required, always succeeds.
    """
    n_kernels = len(K_list)
    weights = np.zeros(n_kernels)

    K_target_norm = np.linalg.norm(K_target, "fro")
    if K_target_norm == 0:
        return _uniform_weights(n_kernels)

    K_y = K_target / K_target_norm
    residual = K_y.copy()
    prev_norm = np.linalg.norm(residual, "fro")

    used = set()

    for _ in range(n_kernels):
        best_idx = -1
        best_dist = np.inf

        for i in range(n_kernels):
            if i in used:
                continue
            K_norm = np.linalg.norm(K_list[i], "fro")
            if K_norm == 0:
                continue
            K_i_hat = K_list[i] / K_norm
            dist = np.linalg.norm(K_i_hat - residual, "fro")
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx == -1:
            break

        used.add(best_idx)

        K_i = K_list[best_idx]
        K_i_norm = np.linalg.norm(K_i, "fro")
        if K_i_norm == 0:
            continue

        K_i_hat = K_i / K_i_norm
        projection = _frobenius_inner(K_i_hat, residual)
        residual = residual - projection * K_i_hat

        current_norm = np.linalg.norm(residual, "fro")
        weights[best_idx] = abs(projection)

        if current_norm >= prev_norm or current_norm < threshold:
            break

        prev_norm = current_norm

    # Normalize
    total = weights.sum()
    if total > 0:
        weights = weights / total
    else:
        weights = _uniform_weights(n_kernels)

    return weights


# ── Internal alias used by statistical_analysis.py ───────────────────────────

def _frobenius_alignment(K1, K2):
    """Alias for kernel_target_alignment (used by ablation module)."""
    return kernel_target_alignment(K1, K2)
