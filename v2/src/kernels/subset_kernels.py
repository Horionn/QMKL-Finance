"""Build quantum kernel matrices on feature subsets.

This module implements the core idea of QMKL-v2:
each quantum kernel K_m only sees Q features (Q = num qubits),
but different kernels see different subsets — together covering
the full d-dimensional feature space.

Assignment dict format:
    assignment = {
        0: [3, 7, 12, 18],   # kernel 0 uses features 3, 7, 12, 18
        1: [0, 4, 9, 15],    # kernel 1 uses features 0, 4, 9, 15
        ...
    }
"""

import numpy as np
from .analytical import compute_kernel, KERNEL_CONFIGS


# ── Subset generation strategies (baselines) ──────────────────────────────────

def non_overlapping_subsets(d, Q, M=None):
    """Generate non-overlapping sequential feature subsets.

    If M is None, creates floor(d/Q) subsets covering d features.
    Remaining features (d mod Q) are discarded.

    Returns:
        assignment : dict[kernel_id -> list[feature_ids]]
    """
    n_subsets = d // Q if M is None else M
    assignment = {}
    for m in range(n_subsets):
        start = (m * Q) % d
        assignment[m] = list(range(start, min(start + Q, d)))
        # Pad if last subset is smaller
        if len(assignment[m]) < Q:
            assignment[m] = list(range(d - Q, d))
    return assignment


def random_subsets(d, Q, M, seed=42):
    """Generate M random (possibly overlapping) feature subsets of size Q.

    Returns:
        assignment : dict[kernel_id -> list[feature_ids]]
    """
    rng = np.random.RandomState(seed)
    return {m: sorted(rng.choice(d, size=Q, replace=False).tolist()) for m in range(M)}


def pca_informed_subsets(X, Q, M):
    """Assign features by their PCA loading magnitude.

    Groups features by their dominant principal component, so each
    kernel covers a statistically coherent subspace.

    Returns:
        assignment : dict[kernel_id -> list[feature_ids]]
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(M, X.shape[1]))
    pca.fit(X)
    loadings = np.abs(pca.components_)   # (M, d)
    assignment = {}
    for m in range(min(M, loadings.shape[0])):
        top_features = np.argsort(loadings[m])[::-1][:Q].tolist()
        assignment[m] = sorted(top_features)
    return assignment


# ── Kernel matrix computation ──────────────────────────────────────────────────

def build_subset_kernels(X, assignment, kernel_configs=None, method="analytical"):
    """Compute kernel matrices for each kernel on its feature subset.

    Args:
        X              : (N, d) full feature matrix (scaled to [0, 2])
        assignment     : dict[kernel_id -> list[feature_ids]]
        kernel_configs : list of (family, alpha) tuples, one per kernel.
                         If None, uses KERNEL_CONFIGS[0] for all.
        method         : 'analytical' (fast numpy) or 'qiskit' (circuit-based)

    Returns:
        kernel_matrices : list of (N, N) matrices, one per kernel in assignment
    """
    if kernel_configs is None:
        kernel_configs = [KERNEL_CONFIGS[0]] * len(assignment)

    kernel_matrices = []
    for m, feat_ids in assignment.items():
        X_sub = X[:, feat_ids]
        family, alpha = kernel_configs[m % len(kernel_configs)]

        if method == "analytical":
            K = compute_kernel(X_sub, X_sub, family, alpha)
        elif method == "qiskit":
            K = _build_qiskit_kernel(X_sub, family, alpha)
        else:
            raise ValueError(f"Unknown method: {method}")

        kernel_matrices.append(K)
    return kernel_matrices


def build_subset_kernels_train_test(X_train, X_test, assignment,
                                    kernel_configs=None, method="analytical"):
    """Compute train and test kernel matrices.

    Returns:
        K_trains : list of (n_train, n_train) matrices
        K_tests  : list of (n_test, n_train) matrices
    """
    if kernel_configs is None:
        kernel_configs = [KERNEL_CONFIGS[0]] * len(assignment)

    K_trains, K_tests = [], []
    for m, feat_ids in assignment.items():
        X_tr = X_train[:, feat_ids]
        X_te = X_test[:, feat_ids]
        family, alpha = kernel_configs[m % len(kernel_configs)]

        if method == "analytical":
            K_tr = compute_kernel(X_tr, X_tr, family, alpha)
            K_te = compute_kernel(X_te, X_tr, family, alpha)
        elif method == "qiskit":
            K_tr = _build_qiskit_kernel(X_tr, X_tr, family, alpha)
            K_te = _build_qiskit_kernel(X_te, X_tr, family, alpha)
        else:
            raise ValueError(f"Unknown method: {method}")

        K_trains.append(K_tr)
        K_tests.append(K_te)
    return K_trains, K_tests


# ── Qiskit backend (used for NB05 hardware experiments) ───────────────────────

def _build_qiskit_kernel(X1, X2=None, family="ZZ", alpha=1.0, backend=None, shots=1024):
    """Compute kernel matrix using Qiskit FidelityQuantumKernel.

    Uses ZZFeatureMap with the given alpha as data_map_func scaling.
    Falls back to analytical if Qiskit is not available.

    Args:
        X1      : (n1, Q) feature matrix
        X2      : (n2, Q) feature matrix (None = symmetric)
        family  : 'Z' or 'ZZ' (XZ not supported via Qiskit feature maps directly)
        alpha   : bandwidth
        backend : Qiskit backend (None = StatevectorSampler)
        shots   : number of shots for hardware

    Returns:
        K : (n1, n2) kernel matrix
    """
    try:
        from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
        from qiskit.primitives import StatevectorSampler
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        from qiskit_algorithms.state_fidelities import ComputeUncompute
    except ImportError:
        # Fallback to analytical
        from .analytical import compute_kernel
        return compute_kernel(X1, X2 if X2 is not None else X1, family, alpha)

    Q = X1.shape[1]

    if family == "Z":
        feature_map = ZFeatureMap(feature_dimension=Q, reps=1)
    else:  # ZZ (default)
        feature_map = ZZFeatureMap(feature_dimension=Q, reps=1)

    if backend is None:
        sampler = StatevectorSampler()
    else:
        from qiskit_ibm_runtime import SamplerV2
        sampler = SamplerV2(backend)

    fidelity = ComputeUncompute(sampler=sampler)
    qkernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

    if X2 is None:
        return qkernel.evaluate(X1)
    return qkernel.evaluate(X1, X2)
