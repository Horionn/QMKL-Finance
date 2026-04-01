"""QUBO formulation for optimal feature-to-kernel assignment.

This is the core novelty of QMKL-v2.

Problem: given d features and M quantum kernels (each using Q qubits),
find the binary assignment x_{k,m} ∈ {0,1} that maximizes the total
centered kernel-target alignment while satisfying:
  - Exactly Q features assigned to each kernel
  - (optionally) Each feature used in at most P kernels

Formulation C (recommended):

  min  -Σ_{m,k} a_{k,m} · x_{k,m}                          [quality]
      + λ · Σ_k Σ_{m<m'} x_{k,m} · x_{k,m'}               [diversity]
      + μ₁ · Σ_m (Σ_k x_{k,m} - Q)²                        [size constraint]

where a_{k,m} = centered_alignment(K_m restricted to feature k + Q-1 padding).

Variable indexing: flat index i = k * M + m
  - k ∈ [0, d)  : feature index
  - m ∈ [0, M)  : kernel index
"""

import numpy as np


# ── Centered alignment (scalar version for a single kernel) ───────────────────

def _center_kernel(K):
    m = K.shape[0]
    H = np.eye(m) - np.ones((m, m)) / m
    return H @ K @ H


def _centered_alignment_scalar(K, Yc):
    """Centered alignment between kernel K and pre-centered target Yc."""
    Kc = _center_kernel(K)
    num = np.sum(Kc * Yc)
    denom = np.sqrt(np.sum(Kc ** 2) * np.sum(Yc ** 2))
    if denom < 1e-12:
        return 0.0
    return float(num / denom)


# ── Marginal alignment computation ────────────────────────────────────────────

def compute_marginal_alignments(X, y, M, Q, kernel_configs, padding="zero"):
    """Compute a_{k,m}: marginal alignment of feature k in kernel m.

    For each (k, m) pair:
      1. Build a Q-feature slice: feature k + Q-1 padding features
      2. Compute K_m on that slice
      3. Return centered alignment with target

    Args:
        X              : (N, d) scaled feature matrix
        y              : (N,) binary labels
        M              : number of kernels
        Q              : qubits per kernel (features per kernel)
        kernel_configs : list of M tuples (family, alpha)
        padding        : 'zero' | 'random' | 'top' — how to fill Q-1 other features

    Returns:
        a : (d, M) float array of marginal alignments
    """
    from ..kernels.analytical import compute_kernel

    N, d = X.shape
    y_pm = 2 * y.astype(float) - 1
    Y = np.outer(y_pm, y_pm)
    Yc = _center_kernel(Y)

    a = np.zeros((d, M))

    for m, (family, alpha) in enumerate(kernel_configs):
        for k in range(d):
            # Build a Q-feature slice with feature k first
            if Q == 1:
                feat_ids = [k]
            elif padding == "zero":
                feat_ids = [k] + [k] * (Q - 1)   # replicate feature k
            elif padding == "random":
                others = [j for j in range(d) if j != k]
                rng = np.random.RandomState(k * M + m)
                fill = rng.choice(others, size=Q - 1, replace=False).tolist()
                feat_ids = [k] + fill
            elif padding == "top":
                # Use k + next Q-1 features cyclically
                feat_ids = [(k + j) % d for j in range(Q)]
            else:
                feat_ids = [k] + [k] * (Q - 1)

            X_sub = X[:, feat_ids]
            K = compute_kernel(X_sub, X_sub, family, alpha)
            a[k, m] = _centered_alignment_scalar(K, Yc)

    return a


# ── QUBO matrix construction ──────────────────────────────────────────────────

def build_qubo_matrix(a, d, M, Q, lambda_div=0.5, mu1=2.0, mu2=0.0):
    """Build the QUBO matrix Q of size (d*M, d*M).

    Variable ordering: flat index i = k*M + m
    (row = feature k, col = kernel m)

    Objective (Formulation C):
      min  -Σ_{k,m} a[k,m] · x_{k,m}
          + λ · Σ_k Σ_{m<m'} x_{k,m} · x_{k,m'}      [overlap penalty]
          + μ₁ · Σ_m (Σ_k x_{k,m} - Q)²               [size constraint]
          + μ₂ · Σ_k Σ_{m<m'} x_{k,m} · x_{k,m'}      [optional extra overlap]

    The size constraint (Σ_k x_{k,m} - Q)² expands to:
      Σ_k x_{k,m}² + 2 Σ_{k<k'} x_{k,m}·x_{k',m} - 2Q Σ_k x_{k,m} + Q²
    Since x² = x for binary variables: Σ_k x_{k,m} → diagonal (1 - 2Q) term.

    Args:
        a         : (d, M) marginal alignment matrix
        d         : number of features
        M         : number of kernels
        Q         : features per kernel
        lambda_div: diversity penalty (λ), penalizes same feature in multiple kernels
        mu1       : size constraint penalty (μ₁)
        mu2       : additional overlap penalty (μ₂, optional)

    Returns:
        Q_mat : (d*M, d*M) upper-triangular QUBO matrix
    """
    n_vars = d * M
    Q_mat = np.zeros((n_vars, n_vars))

    for k in range(d):
        for m in range(M):
            i = k * M + m

            # --- Diagonal: quality + size constraint diagonal term ---
            # Quality: -a[k,m] (minimization → negative sign in maximization)
            # Size:    mu1 * (1 - 2Q) from expanding (sum_k x_{k,m} - Q)²
            Q_mat[i, i] += -a[k, m] + mu1 * (1 - 2 * Q)

            # --- Same kernel m, different features k' > k ---
            # Size constraint cross-term: mu1 * 2 * x_{k,m} * x_{k',m}
            for k2 in range(k + 1, d):
                j = k2 * M + m
                Q_mat[i, j] += 2 * mu1

            # --- Same feature k, different kernels m' > m ---
            # Diversity/overlap penalty
            overlap_coeff = lambda_div + mu2
            for m2 in range(m + 1, M):
                j = k * M + m2
                Q_mat[i, j] += overlap_coeff

    return Q_mat


def energy(x, Q_mat):
    """Evaluate QUBO energy for a binary solution vector x.

    E(x) = x^T Q x  (using upper-triangular Q)
    """
    return float(x @ Q_mat @ x)


# ── Solution decoding ──────────────────────────────────────────────────────────

def decode_assignment(x_binary, d, M, Q, repair=True):
    """Convert flat binary vector to feature-to-kernel assignment dict.

    Args:
        x_binary : (d*M,) binary vector
        d        : number of features
        M        : number of kernels
        Q        : target features per kernel
        repair   : if True, enforce exactly Q features per kernel by
                   greedy top-k selection on the continuous relaxation

    Returns:
        assignment : dict[kernel_id -> list[feature_ids]]
    """
    x = np.array(x_binary).reshape(d, M)  # (d, M)
    assignment = {}

    for m in range(M):
        col = x[:, m]
        selected = np.where(col > 0.5)[0].tolist()

        if repair and len(selected) != Q:
            # Repair: take top-Q by magnitude
            top_k = np.argsort(col)[::-1][:Q]
            selected = sorted(top_k.tolist())

        assignment[m] = sorted(selected)

    return assignment


def assignment_to_vector(assignment, d, M):
    """Convert assignment dict to flat binary vector."""
    x = np.zeros(d * M)
    for m, feat_ids in assignment.items():
        for k in feat_ids:
            x[k * M + m] = 1.0
    return x


def check_constraints(assignment, d, M, Q):
    """Verify that assignment satisfies constraints.

    Returns:
        valid       : bool
        violations  : dict with details
    """
    violations = {}

    for m, feat_ids in assignment.items():
        if len(feat_ids) != Q:
            violations[f"kernel_{m}_size"] = f"has {len(feat_ids)} features, expected {Q}"
        if len(set(feat_ids)) != len(feat_ids):
            violations[f"kernel_{m}_duplicates"] = "duplicate features"
        for k in feat_ids:
            if k < 0 or k >= d:
                violations[f"kernel_{m}_range"] = f"feature {k} out of range [0, {d})"

    return len(violations) == 0, violations
