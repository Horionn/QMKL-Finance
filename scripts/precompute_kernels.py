"""
Precompute all kernel matrices and save to disk cache.

Run ONCE before using the notebooks:
    python scripts/precompute_kernels.py

Then all notebooks will load from cache (milliseconds instead of hours).

Options:
    --dataset    german_credit | bank_marketing | synthetic (default: synthetic)
    --n-samples  Number of samples (default: 200)
    --n-qubits   Number of qubits / PCA components (default: 6)
    --no-cache   Force recompute even if cache exists
    --n-jobs     Parallel jobs for kernel computation (default: -1 = all cores)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.loaders import load_dataset
from src.preprocessing.feature_reduction import FeatureReducer
from src.preprocessing.scaler import QuantumScaler
from src.kernels.feature_maps import get_feature_map_library
from src.kernels.quantum_kernel import build_quantum_kernel
from src.kernels.kernel_matrix import (
    compute_kernel_matrix_parallel,
    cache_info,
)


def main():
    parser = argparse.ArgumentParser(description="Precompute QMKL kernel matrices")
    parser.add_argument("--dataset",   default="synthetic",
                        choices=["synthetic", "german_credit", "bank_marketing"])
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-qubits",  type=int, default=6)
    parser.add_argument("--no-cache",  action="store_true",
                        help="Force recompute even if cache exists")
    parser.add_argument("--n-jobs",    type=int, default=-1,
                        help="Parallel jobs (-1 = all cores)")
    args = parser.parse_args()

    print("=" * 60)
    print("QMKL Kernel Precomputation")
    print("=" * 60)
    print(f"Dataset : {args.dataset}")
    print(f"Samples : {args.n_samples}")
    print(f"Qubits  : {args.n_qubits}")
    print(f"Jobs    : {args.n_jobs}")
    print()

    # ── 1. Load data ──────────────────────────────────────────
    print("Loading dataset...")
    X_raw, y = load_dataset(args.dataset, n_samples=args.n_samples)
    print(f"  Shape: {X_raw.shape}, classes: {np.unique(y, return_counts=True)[1]}")

    # ── 2. Preprocess ─────────────────────────────────────────
    print("Preprocessing (StandardScaler → PCA → MinMaxScaler)...")
    reducer = FeatureReducer(n_components=args.n_qubits)
    scaler  = QuantumScaler()
    X_red   = reducer.fit_transform(X_raw)
    X_proc  = scaler.fit_transform(X_red)
    print(f"  Explained variance: "
          f"{reducer.get_explained_variance()['cumulative'][-1]:.1%}")

    # ── 3. Build all feature maps ──────────────────────────────
    print(f"\nBuilding {args.n_qubits}-qubit feature map library...")
    fm_library = get_feature_map_library(args.n_qubits)
    fm_names   = list(fm_library.keys())
    fm_list    = list(fm_library.values())
    print(f"  {len(fm_names)} feature maps: {fm_names}")

    # ── 4. Build quantum kernels ───────────────────────────────
    print("\nBuilding quantum kernel objects...")
    kernels_fidelity  = [build_quantum_kernel(fm, kernel_type="fidelity")
                         for fm in fm_list]
    kernels_projected = [build_quantum_kernel(fm, kernel_type="projected", gamma=1.0)
                         for fm in fm_list]

    kernel_names_fidelity  = [f"fidelity_{n}" for n in fm_names]
    kernel_names_projected = [f"projected_{n}" for n in fm_names]

    use_cache = not args.no_cache

    # ── 5. Compute fidelity kernels ────────────────────────────
    print(f"\nComputing {len(kernels_fidelity)} FIDELITY kernel matrices "
          f"(n_jobs={args.n_jobs})...")
    t0 = time.time()
    K_fidelity = compute_kernel_matrix_parallel(
        kernels_fidelity, X_proc,
        kernel_names=kernel_names_fidelity,
        use_cache=use_cache,
        n_jobs=args.n_jobs,
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── 6. Compute projected kernels ───────────────────────────
    print(f"\nComputing {len(kernels_projected)} PROJECTED kernel matrices "
          f"(n_jobs={args.n_jobs})...")
    t0 = time.time()
    K_projected = compute_kernel_matrix_parallel(
        kernels_projected, X_proc,
        kernel_names=kernel_names_projected,
        use_cache=use_cache,
        n_jobs=args.n_jobs,
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── 7. Summary ────────────────────────────────────────────
    print("\n" + "=" * 60)
    info = cache_info()
    print(f"Cache: {info['files']} files, {info['total_size_mb']:.1f} MB")
    print(f"  -> {ROOT / 'results' / 'kernel_cache'}")
    print("\n✅ Precomputation complete!")
    print("   Notebooks will now load instantly from cache.")


if __name__ == "__main__":
    main()
