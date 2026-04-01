"""Analytical quantum kernel formulas (numpy, no Qiskit required).

These are the closed-form expressions of fidelity quantum kernels
induced by ZFeatureMap and ZZFeatureMap circuits, exact in the
noiseless simulation case.

K_Z(x, x') = prod_{k} cos²(alpha * (x_k - x'_k))
K_ZZ(x, x') = K_Z(x, x') * prod_{k<l} cos²(alpha * (x_k*x_l - x'_k*x'_l))

Reference: Havlíček et al., Nature 2019 + IBM QMKL paper 2023.
"""

import numpy as np


# ── Kernel families ───────────────────────────────────────────────────────────

def K_Z(X1, X2, alpha=1.0):
    """Z-feature-map kernel (local, 1-body interactions only).

    K_Z(x, x') = prod_{k=1}^{Q} cos²(alpha * (x_k - x'_k))

    Args:
        X1 : (n1, Q) array — Q features only
        X2 : (n2, Q) array — Q features only
        alpha : bandwidth parameter

    Returns:
        K : (n1, n2) kernel matrix
    """
    n1, Q = X1.shape
    n2 = X2.shape[0]
    K = np.ones((n1, n2))
    for k in range(Q):
        diff = X1[:, k:k+1] - X2[:, k].reshape(1, -1)   # (n1, n2)
        K *= np.cos(alpha * diff) ** 2
    return K


def K_ZZ(X1, X2, alpha=1.0):
    """ZZ-feature-map kernel (pairwise 2-body interactions).

    K_ZZ(x, x') = K_Z(x, x') * prod_{k<l} cos²(alpha * (x_k*x_l - x'_k*x'_l))

    Args:
        X1 : (n1, Q) array
        X2 : (n2, Q) array
        alpha : bandwidth parameter

    Returns:
        K : (n1, n2) kernel matrix
    """
    K = K_Z(X1, X2, alpha=alpha)
    _, Q = X1.shape
    for k in range(Q):
        for l in range(k + 1, Q):
            prod1 = X1[:, k:k+1] * X1[:, l:l+1]         # (n1, 1)
            prod2 = (X2[:, k] * X2[:, l]).reshape(1, -1)  # (1, n2)
            diff = prod1 - prod2
            K *= np.cos(alpha * diff) ** 2
    return K


def K_XZ(X1, X2, alpha=1.0):
    """XZ-feature-map kernel (mixed X and ZZ interactions, 2-body).

    Approximation using alternating X and Z rotations:
    K_XZ = prod_k cos²(alpha * x_k) * prod_{k<l} cos²(alpha * (x_k - x'_k) * (x_l + x'_l))
    """
    n1, Q = X1.shape
    n2 = X2.shape[0]
    K = np.ones((n1, n2))
    # Local X-rotation term
    for k in range(Q):
        term1 = np.cos(alpha * X1[:, k:k+1]) ** 2       # (n1, 1)
        term2 = np.cos(alpha * X2[:, k]).reshape(1, -1)  # (1, n2)
        K *= term1 * term2
    # Cross-term
    for k in range(Q):
        for l in range(k + 1, Q):
            diff = (X1[:, k:k+1] - X2[:, k].reshape(1, -1))
            summ = (X1[:, l:l+1] + X2[:, l].reshape(1, -1))
            K *= np.cos(alpha * diff * summ) ** 2
    return K


# ── Registry of kernel functions ──────────────────────────────────────────────

KERNEL_REGISTRY = {
    "Z":  K_Z,
    "ZZ": K_ZZ,
    "XZ": K_XZ,
}

KERNEL_CONFIGS = [
    ("Z",  0.5),
    ("Z",  2.0),
    ("ZZ", 0.5),
    ("ZZ", 2.0),
    ("XZ", 0.5),
    ("XZ", 2.0),
]


def compute_kernel(X1, X2, family, alpha):
    """Compute a kernel matrix for a given family and alpha.

    Args:
        X1 : (n1, Q) array — Q features
        X2 : (n2, Q) array — Q features
        family : str, one of 'Z', 'ZZ', 'XZ'
        alpha  : float, bandwidth

    Returns:
        K : (n1, n2) matrix
    """
    return KERNEL_REGISTRY[family](X1, X2, alpha=alpha)
