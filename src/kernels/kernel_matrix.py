"""Kernel matrix computation and utilities.

Performance features:
- Disk caching: computed matrices saved to results/kernel_cache/ as .npy files
- Parallel computation: joblib for multi-core kernel evaluation
- Batch evaluation: leverages Qiskit's native batching
"""

import hashlib
import os
from pathlib import Path

import numpy as np

# Default cache directory (relative to project root)
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "results" / "kernel_cache"


# ──────────────────────────────────────────────
# CACHING UTILITIES
# ──────────────────────────────────────────────

def _array_hash(arr: np.ndarray) -> str:
    """Stable hash of a numpy array for cache keys."""
    return hashlib.md5(arr.tobytes()).hexdigest()[:16]


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.npy"


def _save_kernel(K: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), K)


def _load_kernel(path: Path) -> np.ndarray:
    return np.load(str(path))


# ──────────────────────────────────────────────
# MAIN FUNCTION
# ──────────────────────────────────────────────

def compute_kernel_matrix(
    kernel,
    X_train: np.ndarray,
    X_test: np.ndarray = None,
    cache_dir: Path = None,
    cache_key: str = None,
    use_cache: bool = True,
) -> np.ndarray:
    """Compute the kernel matrix using a quantum kernel, with optional disk cache.

    On first run: evaluates all circuits (slow), saves result to disk.
    On subsequent runs: loads from cache instantly (milliseconds).

    Args:
        kernel: A quantum kernel object (FidelityQuantumKernel or ProjectedQuantumKernel).
        X_train: Training data array of shape (n_train, n_features).
        X_test: Test data array. If None, computes K(X_train, X_train).
        cache_dir: Directory to store cached matrices. Defaults to results/kernel_cache/.
        cache_key: Custom cache key string. If None, derived from data hash.
        use_cache: If False, always recompute (useful for debugging).

    Returns:
        Kernel matrix as numpy array.
    """
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR

    # Build cache key from data content
    if cache_key is None:
        train_hash = _array_hash(X_train)
        if X_test is not None:
            test_hash = _array_hash(X_test)
            cache_key = f"kernel_test_{train_hash}_{test_hash}"
        else:
            cache_key = f"kernel_train_{train_hash}"

    path = _cache_path(Path(cache_dir), cache_key)

    # Try loading from cache
    if use_cache and path.exists():
        return _load_kernel(path)

    # Compute
    if X_test is None:
        K = kernel.evaluate(X_train)
    else:
        K = kernel.evaluate(X_test, X_train)

    K = np.array(K)

    # Save to cache
    if use_cache:
        _save_kernel(K, path)

    return K


def compute_kernel_matrix_parallel(
    kernels: list,
    X_train: np.ndarray,
    X_test: np.ndarray = None,
    kernel_names: list = None,
    cache_dir: Path = None,
    use_cache: bool = True,
    n_jobs: int = -1,
) -> list:
    """Compute multiple kernel matrices in parallel.

    Uses joblib for multi-core evaluation. Each kernel is independent,
    so perfect parallelism.

    Args:
        kernels: List of quantum kernel objects.
        X_train: Training data.
        X_test: Test data (optional).
        kernel_names: List of names for cache keys. If None, uses index.
        cache_dir: Cache directory.
        use_cache: Whether to use disk cache.
        n_jobs: Number of parallel jobs (-1 = all CPU cores).

    Returns:
        List of kernel matrices (same order as input).
    """
    from joblib import Parallel, delayed

    if kernel_names is None:
        kernel_names = [str(i) for i in range(len(kernels))]

    train_hash = _array_hash(X_train)
    if X_test is not None:
        test_hash = _array_hash(X_test)
        mode = "test"
    else:
        test_hash = None
        mode = "train"

    def _compute_one(kernel, name):
        if test_hash is not None:
            key = f"{name}_{mode}_{train_hash}_{test_hash}"
        else:
            key = f"{name}_{mode}_{train_hash}"
        return compute_kernel_matrix(
            kernel, X_train, X_test,
            cache_dir=cache_dir,
            cache_key=key,
            use_cache=use_cache,
        )

    K_list = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_compute_one)(k, name)
        for k, name in zip(kernels, kernel_names)
    )
    return K_list


def clear_cache(cache_dir: Path = None):
    """Delete all cached kernel matrices."""
    import shutil
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR
    if Path(cache_dir).exists():
        shutil.rmtree(cache_dir)
        print(f"Cache cleared: {cache_dir}")
    else:
        print("No cache found.")


def cache_info(cache_dir: Path = None) -> dict:
    """Show cache contents and sizes."""
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return {"files": 0, "total_size_mb": 0.0, "entries": []}

    entries = []
    total = 0
    for f in sorted(cache_dir.glob("*.npy")):
        size = f.stat().st_size
        total += size
        entries.append({"name": f.stem, "size_mb": size / 1e6})

    return {
        "files": len(entries),
        "total_size_mb": total / 1e6,
        "entries": entries,
    }


# ──────────────────────────────────────────────
# KERNEL QUALITY UTILITIES
# ──────────────────────────────────────────────

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
