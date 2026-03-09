"""Feature dimensionality reduction for quantum encoding."""

import numpy as np
from sklearn.decomposition import PCA


class FeatureReducer:
    """Reduce feature dimensionality using PCA.

    The number of components determines the number of qubits needed.
    In quantum kernel methods, each feature maps to one qubit.
    """

    def __init__(self, n_components=8, method="pca"):
        self.n_components = n_components
        self.method = method

        if method == "pca":
            self._reducer = PCA(n_components=n_components)
        else:
            raise ValueError(f"Unknown reduction method: {method}")

        self.explained_variance_ratio_ = None

    def fit(self, X):
        self._reducer.fit(X)
        if self.method == "pca":
            self.explained_variance_ratio_ = self._reducer.explained_variance_ratio_
        return self

    def transform(self, X):
        return self._reducer.transform(X)

    def fit_transform(self, X):
        result = self._reducer.fit_transform(X)
        if self.method == "pca":
            self.explained_variance_ratio_ = self._reducer.explained_variance_ratio_
        return result

    def get_explained_variance(self):
        if self.explained_variance_ratio_ is None:
            raise RuntimeError("Call fit() first.")
        return {
            "per_component": self.explained_variance_ratio_,
            "cumulative": np.cumsum(self.explained_variance_ratio_),
            "total": np.sum(self.explained_variance_ratio_),
        }
