"""Kernel matrix computation and utilities."""

import numpy as np


def compute_kernel_matrix(kernel, X_train, X_test=None):
    """Compute the kernel matrix using a quantum kernel.

    Args:
        kernel: A quantum kernel object (FidelityQuantumKernel or ProjectedQuantumKernel).
        X_train: Training data array of shape (n_train, n_features).
        X_test: Test data array. If None, computes K(X_train, X_train).

    Returns:
        Kernel matrix as numpy array.
    """
    if X_test is None:
        K = kernel.evaluate(X_train)
    else:
        K = kernel.evaluate(X_test, X_train)

    return np.array(K)


def ensure_psd(K, epsilon=1e-10):
    """Ensure a kernel matrix is positive semi-definite.

    Adds a small diagonal perturbation if needed.
    """
    eigenvalues = np.linalg.eigvalsh(K)
    if np.min(eigenvalues) < 0:
        K = K + (abs(np.min(eigenvalues)) + epsilon) * np.eye(K.shape[0])
    return K


def normalize_kernel(K):
    """Normalize a kernel matrix so that diagonal elements are 1.

    K_norm(i,j) = K(i,j) / sqrt(K(i,i) * K(j,j))
    """
    diag = np.sqrt(np.diag(K))
    diag[diag == 0] = 1.0  # Avoid division by zero
    return K / np.outer(diag, diag)


def kernel_statistics(K):
    """Compute statistics of kernel matrix elements.

    Useful for diagnosing exponential concentration.
    """
    n = K.shape[0]
    # Get off-diagonal elements
    mask = ~np.eye(n, dtype=bool)
    off_diag = K[mask]

    return {
        "mean": np.mean(off_diag),
        "variance": np.var(off_diag),
        "std": np.std(off_diag),
        "min": np.min(off_diag),
        "max": np.max(off_diag),
        "diag_mean": np.mean(np.diag(K)),
    }
