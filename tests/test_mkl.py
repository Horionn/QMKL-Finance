"""Tests for Multiple Kernel Learning modules."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mkl.alignment import (
    kernel_target_alignment,
    centered_alignment,
    projection_alignment,
)
from src.mkl.combiner import MultipleKernelCombiner


def _make_dummy_kernels(n=20, n_kernels=3):
    """Create dummy kernel matrices for testing."""
    kernels = []
    for _ in range(n_kernels):
        A = np.random.rand(n, n)
        K = A @ A.T  # Ensure PSD
        K = K / np.max(K)  # Normalize
        kernels.append(K)

    y = np.random.choice([0, 1], size=n)
    return kernels, y


class TestAlignment:
    def test_kernel_target_alignment_score(self):
        K = np.eye(5)
        K_target = np.eye(5)
        score = kernel_target_alignment(K, K_target)
        assert score == pytest.approx(1.0)

    def test_centered_alignment_weights(self):
        kernels, y = _make_dummy_kernels(n=20, n_kernels=3)
        K_target = np.outer(y, y).astype(float)

        weights = centered_alignment(kernels, K_target)
        assert len(weights) == 3
        assert np.all(weights >= 0)

    def test_projection_alignment_weights(self):
        kernels, y = _make_dummy_kernels(n=20, n_kernels=3)
        K_target = np.outer(y, y).astype(float)

        weights = projection_alignment(kernels, K_target)
        assert len(weights) == 3
        assert np.all(weights >= 0)


class TestCombiner:
    def test_average_combiner(self):
        kernels, y = _make_dummy_kernels(n=20, n_kernels=4)

        combiner = MultipleKernelCombiner(method="average")
        K_combined = combiner.fit_combine(kernels, y)

        assert K_combined.shape == (20, 20)
        weights = combiner.get_weights()
        np.testing.assert_array_almost_equal(weights, [0.25] * 4)

    def test_centered_combiner(self):
        kernels, y = _make_dummy_kernels(n=20, n_kernels=3)

        combiner = MultipleKernelCombiner(method="centered")
        K_combined = combiner.fit_combine(kernels, y)

        assert K_combined.shape == (20, 20)
        weights = combiner.get_weights()
        assert len(weights) == 3
        assert np.sum(weights) == pytest.approx(1.0, abs=0.01)

    def test_combine_without_fit_raises(self):
        combiner = MultipleKernelCombiner(method="average")
        with pytest.raises(RuntimeError):
            combiner.combine([np.eye(5)])

    def test_combined_kernel_is_psd(self):
        kernels, y = _make_dummy_kernels(n=15, n_kernels=3)

        combiner = MultipleKernelCombiner(method="average")
        K_combined = combiner.fit_combine(kernels, y)

        eigenvalues = np.linalg.eigvalsh(K_combined)
        assert np.all(eigenvalues >= -1e-8)
