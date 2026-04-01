"""Diagnostic utilities for quantum kernel analysis.

Used across NB01 (barren plateau) and NB02 (complexity crossover).
"""

import numpy as np
import time


def gradient_variance(X, kernel_fn, alpha=1.0, epsilon=0.01):
    """Finite-difference variance of dK/d_alpha over off-diagonal entries.

    Measures barren plateau severity: if this → 0 exponentially with Q,
    gradient-based training is impossible.

    Args:
        X          : (N, Q) feature matrix
        kernel_fn  : callable(X1, X2, alpha) → (N, N) kernel matrix
        alpha      : operating point
        epsilon    : finite-difference step

    Returns:
        var : float — variance of the gradient matrix off-diagonal entries
    """
    K_plus  = kernel_fn(X, X, alpha + epsilon)
    K_minus = kernel_fn(X, X, alpha - epsilon)
    dK      = (K_plus - K_minus) / (2 * epsilon)

    # Off-diagonal only (diagonal is always 1 for fidelity kernels)
    n = dK.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(np.var(dK[mask]))


def kernel_concentration_std(K):
    """Std of off-diagonal kernel entries.

    A concentrated kernel has std → 0 (all pairs look identical).
    Returns the std of K[i,j] for i ≠ j.
    """
    n = K.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(np.std(K[mask]))


def kernel_mean_offdiag(K):
    """Mean of off-diagonal kernel entries."""
    n = K.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(np.mean(K[mask]))


def time_kernel_fn(kernel_fn, X, alpha=1.0, n_repeats=5):
    """Time a kernel computation, return median over n_repeats (seconds)."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        kernel_fn(X, X, alpha)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def count_circuit_resources(Q, feature_map='ZZ', reps=1):
    """Count gate resources for a ZZFeatureMap circuit of Q qubits.

    Returns dict with: n_cnot, n_single, depth, total_gates.
    Uses Qiskit circuit library (no execution).
    """
    try:
        from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
        from qiskit import transpile

        if feature_map == 'ZZ':
            fm = ZZFeatureMap(feature_dimension=Q, reps=reps)
        else:
            fm = ZFeatureMap(feature_dimension=Q, reps=reps)

        # Decompose to basis gates
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit.providers.fake_provider import GenericBackendV2
        backend = GenericBackendV2(num_qubits=max(Q, 5))
        pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
        isa = pm.run(fm)

        ops = isa.count_ops()
        n_cnot   = ops.get('cx', 0) + ops.get('ecr', 0) + ops.get('cz', 0)
        n_single = sum(v for k, v in ops.items() if k not in ('cx', 'ecr', 'cz', 'measure', 'barrier'))
        return {
            'Q': Q,
            'n_cnot': n_cnot,
            'n_single': n_single,
            'total_gates': n_cnot + n_single,
            'depth': isa.depth(),
        }
    except Exception as e:
        # Fallback: analytical formula for ZZFeatureMap
        # reps layers, each with Q Ry + Q Rz + Q*(Q-1)/2 CX pairs
        n_cnot   = reps * Q * (Q - 1) // 2 * 2   # CX + CX (each pair uses 2 CX)
        n_single = reps * Q * 3                    # Ry, Rz, Rz per qubit
        return {
            'Q': Q,
            'n_cnot': n_cnot,
            'n_single': n_single,
            'total_gates': n_cnot + n_single,
            'depth': reps * (Q + 1),              # rough estimate
        }
