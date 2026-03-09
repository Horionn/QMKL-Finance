"""Multiple Kernel Combiner.

Combines multiple quantum kernel matrices into a single kernel
using a weighted linear combination: K = sum(w_i * K_i).
"""

import numpy as np
from .alignment import centered_alignment, projection_alignment, sdp_alignment


class MultipleKernelCombiner:
    """Combine multiple kernel matrices with optimized weights.

    Supports several weight optimization strategies from the literature:
    - "average": Equal weights (baseline)
    - "centered": Centered kernel-target alignment (Cortes et al.)
    - "sdp": SDP-based kernel-target alignment
    - "projection": Iterative projection-based alignment (IBM paper)
    - "bayesian": Bayesian optimization of weights (BO-MKQSVM paper)
    """

    def __init__(self, method="centered", normalize=True):
        self.method = method
        self.normalize = normalize
        self.weights_ = None
        self.n_kernels_ = None

    def fit(self, kernel_matrices, y_train):
        """Compute optimal kernel weights.

        Args:
            kernel_matrices: List of kernel matrices, each (n_train, n_train).
            y_train: Training labels array.

        Returns:
            self
        """
        self.n_kernels_ = len(kernel_matrices)
        K_list = [np.array(K) for K in kernel_matrices]

        # Build target kernel matrix
        K_target = self._build_target_kernel(y_train)

        if self.method == "average":
            self.weights_ = np.ones(self.n_kernels_) / self.n_kernels_

        elif self.method == "centered":
            self.weights_ = centered_alignment(K_list, K_target)

        elif self.method == "sdp":
            self.weights_ = sdp_alignment(K_list, K_target)

        elif self.method == "projection":
            self.weights_ = projection_alignment(K_list, K_target)

        else:
            raise ValueError(
                f"Unknown method: {self.method}. "
                f"Use 'average', 'centered', 'sdp', or 'projection'."
            )

        if self.normalize:
            w_sum = np.sum(self.weights_)
            if w_sum > 0:
                self.weights_ = self.weights_ / w_sum

        return self

    def combine(self, kernel_matrices):
        """Combine kernel matrices using fitted weights.

        Args:
            kernel_matrices: List of kernel matrices to combine.

        Returns:
            Combined kernel matrix.
        """
        if self.weights_ is None:
            raise RuntimeError("Call fit() before combine().")

        K_combined = np.zeros_like(kernel_matrices[0])
        for w, K in zip(self.weights_, kernel_matrices):
            K_combined += w * K

        return K_combined

    def fit_combine(self, kernel_matrices_train, y_train):
        """Fit weights and combine training kernels in one step."""
        self.fit(kernel_matrices_train, y_train)
        return self.combine(kernel_matrices_train)

    def _build_target_kernel(self, y):
        """Build the ideal target kernel matrix from labels.

        K_target[i,j] = 1 if y[i] == y[j], else 0.
        """
        n = len(y)
        K_target = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K_target[i, j] = 1.0 if y[i] == y[j] else 0.0
        return K_target

    def get_weights(self):
        if self.weights_ is None:
            raise RuntimeError("Call fit() first.")
        return self.weights_.copy()
