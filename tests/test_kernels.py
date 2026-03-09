"""Tests for quantum kernel modules."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kernels.feature_maps import build_feature_map, get_feature_map_library
from src.kernels.quantum_kernel import build_quantum_kernel
from src.kernels.kernel_matrix import compute_kernel_matrix, kernel_statistics, normalize_kernel, ensure_psd


class TestFeatureMaps:
    def test_z_feature_map(self):
        fm = build_feature_map("Z", n_qubits=4, alpha=1.0, reps=1)
        assert fm.num_qubits == 4

    def test_zz_feature_map(self):
        fm = build_feature_map("ZZ", n_qubits=4, alpha=2.0, reps=1)
        assert fm.num_qubits == 4

    def test_pauli_feature_map(self):
        fm = build_feature_map("pauli", n_qubits=4, alpha=0.6, reps=2)
        assert fm.num_qubits == 4

    def test_unknown_feature_map(self):
        with pytest.raises(ValueError):
            build_feature_map("unknown", n_qubits=4)

    def test_feature_map_library(self):
        library = get_feature_map_library(n_qubits=4)
        assert len(library) > 0
        for label, fm in library:
            assert isinstance(label, str)
            assert fm.num_qubits == 4


class TestQuantumKernel:
    def test_fidelity_kernel_matrix(self):
        n_qubits = 2
        fm = build_feature_map("ZZ", n_qubits=n_qubits, alpha=1.0, reps=1)
        qk = build_quantum_kernel(fm, kernel_type="fidelity")

        X = np.random.rand(5, n_qubits) * 2
        K = compute_kernel_matrix(qk, X)

        assert K.shape == (5, 5)
        # Kernel matrix should be symmetric
        np.testing.assert_array_almost_equal(K, K.T, decimal=5)
        # Diagonal should be 1 (or close)
        np.testing.assert_array_almost_equal(np.diag(K), np.ones(5), decimal=3)

    def test_kernel_psd(self):
        n_qubits = 2
        fm = build_feature_map("Z", n_qubits=n_qubits, alpha=1.0, reps=1)
        qk = build_quantum_kernel(fm, kernel_type="fidelity")

        X = np.random.rand(10, n_qubits)
        K = compute_kernel_matrix(qk, X)
        K = ensure_psd(K)

        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues >= -1e-8)


class TestKernelMatrix:
    def test_kernel_statistics(self):
        K = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
        stats = kernel_statistics(K)

        assert "mean" in stats
        assert "variance" in stats
        assert stats["diag_mean"] == pytest.approx(1.0)

    def test_normalize_kernel(self):
        K = np.array([[4.0, 2.0], [2.0, 1.0]])
        K_norm = normalize_kernel(K)

        np.testing.assert_array_almost_equal(np.diag(K_norm), [1.0, 1.0])

    def test_ensure_psd(self):
        # Create a matrix that is not PSD
        K = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues: -1, 3
        K_psd = ensure_psd(K)

        eigenvalues = np.linalg.eigvalsh(K_psd)
        assert np.all(eigenvalues >= 0)
