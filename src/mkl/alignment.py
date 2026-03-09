"""Kernel-target alignment optimization methods.

Implements three alignment strategies from the IBM QMKL paper:
1. SDP-based alignment
2. Centered alignment (Cortes et al., 2012)
3. Iterative projection-based alignment
"""

import numpy as np


def _frobenius_inner(K1, K2):
    """Frobenius inner product between two matrices."""
    return np.sum(K1 * K2)


def _center_kernel(K):
    """Center a kernel matrix in feature space.

    K_c = (I - 11^T/m) K (I - 11^T/m)
    """
    m = K.shape[0]
    ones = np.ones((m, m)) / m
    I = np.eye(m)
    centering = I - ones
    return centering @ K @ centering


def kernel_target_alignment(K, K_target):
    """Compute the kernel-target alignment score.

    A(K1, K2) = <K1, K2>_F / sqrt(<K1, K1>_F * <K2, K2>_F)
    """
    num = _frobenius_inner(K, K_target)
    denom = np.sqrt(_frobenius_inner(K, K) * _frobenius_inner(K_target, K_target))
    if denom == 0:
        return 0.0
    return num / denom


# --- Strategy 1: SDP-based alignment ---

def sdp_alignment(K_list, K_target):
    """Optimize kernel weights using semidefinite programming.

    Solves: max w^T q  s.t. w^T S w <= 1, w >= 0
    where q_i = <K_i, K_target>_F and S_ij = <K_i, K_j>_F
    """
    try:
        import cvxpy as cp
    except ImportError:
        raise ImportError("cvxpy is required for SDP alignment. Install with: pip install cvxpy")

    n_kernels = len(K_list)

    # Build q vector and S matrix
    q = np.array([_frobenius_inner(K, K_target) for K in K_list])
    S = np.zeros((n_kernels, n_kernels))
    for i in range(n_kernels):
        for j in range(n_kernels):
            S[i, j] = _frobenius_inner(K_list[i], K_list[j])

    # Add small regularization for numerical stability
    S += 1e-8 * np.eye(n_kernels)

    # Solve QCQP
    w = cp.Variable(n_kernels)
    objective = cp.Maximize(w @ q)
    constraints = [
        cp.quad_form(w, S) <= 1,
        w >= 0,
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)

    weights = np.maximum(w.value, 0)
    return weights


# --- Strategy 2: Centered alignment ---

def centered_alignment(K_list, K_target):
    """Optimize kernel weights using centered kernel alignment.

    Centers all kernels before computing alignment, which better
    correlates with generalization performance (Cortes et al., 2012).

    Solves: min v^T M v - 2 v^T a  s.t. v >= 0
    where a_i = <K_i^c, K_target^c>_F and M_ij = <K_i^c, K_j^c>_F
    """
    n_kernels = len(K_list)

    # Center all kernels
    K_list_c = [_center_kernel(K) for K in K_list]
    K_target_c = _center_kernel(K_target)

    # Build a vector and M matrix
    a = np.array([_frobenius_inner(Kc, K_target_c) for Kc in K_list_c])
    M = np.zeros((n_kernels, n_kernels))
    for i in range(n_kernels):
        for j in range(n_kernels):
            M[i, j] = _frobenius_inner(K_list_c[i], K_list_c[j])

    # Add regularization
    M += 1e-8 * np.eye(n_kernels)

    # Solve the QP: min v^T M v - 2 v^T a, s.t. v >= 0
    try:
        import cvxpy as cp

        v = cp.Variable(n_kernels)
        objective = cp.Minimize(cp.quad_form(v, M) - 2 * a @ v)
        constraints = [v >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)

        weights = np.maximum(v.value, 0)
    except ImportError:
        # Fallback: closed-form solution (may have negative weights)
        weights = np.linalg.solve(M, a)
        weights = np.maximum(weights, 0)

    # Normalize
    norm = np.linalg.norm(weights)
    if norm > 0:
        weights = weights / norm

    return weights


# --- Strategy 3: Iterative projection ---

def projection_alignment(K_list, K_target, threshold=1e-6):
    """Optimize kernel weights using iterative projection-based alignment.

    Custom method from the IBM QMKL paper:
    1. Start with K_target
    2. Find the kernel K closest to the residual
    3. Subtract its projection, update weights
    4. Repeat until residual stops decreasing

    Args:
        K_list: List of kernel matrices.
        K_target: Target kernel matrix.
        threshold: Convergence threshold on residual norm.
    """
    n_kernels = len(K_list)
    weights = np.zeros(n_kernels)

    # Normalize target
    K_target_norm = np.linalg.norm(K_target, "fro")
    if K_target_norm == 0:
        return weights

    K_y = K_target / K_target_norm
    residual = K_y.copy()
    prev_norm = np.linalg.norm(residual, "fro")

    used = set()

    for _ in range(n_kernels):
        # Find kernel closest to residual
        best_idx = -1
        best_dist = np.inf

        for i in range(n_kernels):
            if i in used:
                continue
            # Normalize kernel
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

        # Project and subtract
        K_i = K_list[best_idx]
        K_i_norm = np.linalg.norm(K_i, "fro")
        if K_i_norm == 0:
            continue

        K_i_hat = K_i / K_i_norm
        projection = _frobenius_inner(K_i_hat, residual)
        residual = residual - projection * K_i_hat

        current_norm = np.linalg.norm(residual, "fro")

        # The weight is proportional to the projection magnitude
        weights[best_idx] = abs(projection)

        # Termination: residual stopped decreasing or below threshold
        if current_norm >= prev_norm or current_norm < threshold:
            break

        prev_norm = current_norm

    return weights
