"""Quantum Support Vector Machine with precomputed kernel."""

import numpy as np
from sklearn.svm import SVC


class QSVM:
    """SVM classifier using a precomputed quantum kernel matrix.

    Wraps sklearn's SVC with kernel='precomputed'.
    """

    def __init__(self, C=1.0):
        self.C = C
        self.svm = SVC(kernel="precomputed", C=C, probability=True)
        self.is_fitted = False

    def fit(self, K_train, y_train):
        """Fit the SVM on a precomputed kernel matrix.

        Args:
            K_train: Training kernel matrix (n_train, n_train).
            y_train: Training labels.
        """
        self.svm.fit(K_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, K_test):
        """Predict labels for test data.

        Args:
            K_test: Test kernel matrix (n_test, n_train).
        """
        return self.svm.predict(K_test)

    def predict_proba(self, K_test):
        """Predict class probabilities.

        Args:
            K_test: Test kernel matrix (n_test, n_train).
        """
        return self.svm.predict_proba(K_test)

    def decision_function(self, K_test):
        """Compute decision function values."""
        return self.svm.decision_function(K_test)
