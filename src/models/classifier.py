"""High-level QMKL Classifier orchestrating the full pipeline."""

import numpy as np
from ..preprocessing import QuantumScaler, FeatureReducer
from ..kernels import build_feature_map, build_quantum_kernel, compute_kernel_matrix
from ..kernels.feature_maps import get_feature_map_library
from ..mkl import MultipleKernelCombiner
from ..mkl.bayesian_optimizer import BayesianKernelOptimizer
from .qsvm import QSVM


class QMKLClassifier:
    """End-to-end QMKL classifier.

    Orchestrates:
    1. Preprocessing (scaling, PCA)
    2. Quantum kernel computation (multiple feature maps)
    3. Multiple kernel combination (alignment or BO)
    4. SVM classification
    """

    def __init__(
        self,
        n_components=8,
        feature_range=(0, 2),
        kernel_type="fidelity",
        mkl_method="centered",
        C=1.0,
        feature_maps=None,
        projected_gamma=1.0,
    ):
        self.n_components = n_components
        self.feature_range = feature_range
        self.kernel_type = kernel_type
        self.mkl_method = mkl_method
        self.C = C
        self.projected_gamma = projected_gamma

        # Pipeline components
        self.scaler = QuantumScaler(feature_range=feature_range)
        self.reducer = FeatureReducer(n_components=n_components)
        self.combiner = None
        self.svm = QSVM(C=C)
        self.bo_optimizer = None

        # Feature maps: use default library if not specified
        self._feature_map_configs = feature_maps

        # Stored for prediction
        self._quantum_kernels = []
        self._X_train_processed = None
        self._kernel_matrices_train = []

    def fit(self, X_train, y_train):
        """Fit the full QMKL pipeline.

        Args:
            X_train: Raw training features.
            y_train: Training labels.
        """
        # Step 1: Preprocess
        X_scaled = self.scaler.fit_transform(X_train)
        X_reduced = self.reducer.fit_transform(X_scaled)
        self._X_train_processed = X_reduced

        n_qubits = X_reduced.shape[1]

        # Step 2: Build feature maps and quantum kernels
        if self._feature_map_configs is None:
            fm_library = get_feature_map_library(n_qubits)
        else:
            fm_library = []
            for cfg in self._feature_map_configs:
                fm = build_feature_map(
                    cfg["name"], n_qubits,
                    alpha=cfg.get("alpha", 1.0),
                    reps=cfg.get("reps", 1),
                )
                fm_library.append((cfg["name"], fm))

        # Step 3: Compute kernel matrices
        self._quantum_kernels = []
        self._kernel_matrices_train = []

        for label, fm in fm_library:
            qk = build_quantum_kernel(fm, kernel_type=self.kernel_type, gamma=self.projected_gamma)
            K_train = compute_kernel_matrix(qk, X_reduced)
            self._quantum_kernels.append((label, qk))
            self._kernel_matrices_train.append(K_train)

        # Step 4: Combine kernels
        if self.mkl_method == "bayesian":
            self.bo_optimizer = BayesianKernelOptimizer()
            weights, best_C = self.bo_optimizer.optimize(
                self._kernel_matrices_train, y_train
            )
            self.C = best_C
            self.svm = QSVM(C=best_C)

            # Combine with BO weights
            K_combined = np.zeros_like(self._kernel_matrices_train[0])
            for w, K in zip(weights, self._kernel_matrices_train):
                K_combined += w * K
        else:
            self.combiner = MultipleKernelCombiner(method=self.mkl_method)
            K_combined = self.combiner.fit_combine(self._kernel_matrices_train, y_train)

        # Step 5: Train SVM
        self.svm.fit(K_combined, y_train)

        return self

    def predict(self, X_test):
        """Predict labels for test data."""
        K_test_combined = self._compute_test_kernel(X_test)
        return self.svm.predict(K_test_combined)

    def predict_proba(self, X_test):
        """Predict class probabilities for test data."""
        K_test_combined = self._compute_test_kernel(X_test)
        return self.svm.predict_proba(K_test_combined)

    def _compute_test_kernel(self, X_test):
        """Compute combined test kernel matrix."""
        X_scaled = self.scaler.transform(X_test)
        X_reduced = self.reducer.transform(X_scaled)

        # Compute test kernel matrices for each quantum kernel
        test_kernel_matrices = []
        for (label, qk), K_train in zip(self._quantum_kernels, self._kernel_matrices_train):
            K_test = compute_kernel_matrix(qk, self._X_train_processed, X_reduced)
            test_kernel_matrices.append(K_test)

        # Combine with weights
        if self.mkl_method == "bayesian" and self.bo_optimizer is not None:
            weights = self.bo_optimizer.best_weights_
        elif self.combiner is not None:
            weights = self.combiner.get_weights()
        else:
            weights = np.ones(len(test_kernel_matrices)) / len(test_kernel_matrices)

        K_combined = np.zeros_like(test_kernel_matrices[0])
        for w, K in zip(weights, test_kernel_matrices):
            K_combined += w * K

        return K_combined

    def get_kernel_weights(self):
        """Return the learned kernel weights."""
        if self.mkl_method == "bayesian" and self.bo_optimizer is not None:
            return self.bo_optimizer.best_weights_
        elif self.combiner is not None:
            return self.combiner.get_weights()
        return None
