"""Preprocessing pipeline for quantum feature encoding."""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline


class QuantumScaler:
    """Standardize data then scale to a range suitable for quantum rotation angles.

    Following the IBM QMKL paper: standardize -> scale to [0, 2].
    The scaling range can be adjusted (e.g., [0, 2*pi] for angle encoding).
    """

    def __init__(self, feature_range=(0, 2)):
        self.feature_range = feature_range
        self._pipeline = Pipeline([
            ("standardize", StandardScaler()),
            ("scale", MinMaxScaler(feature_range=feature_range)),
        ])

    def fit(self, X):
        self._pipeline.fit(X)
        return self

    def transform(self, X):
        return self._pipeline.transform(X)

    def fit_transform(self, X):
        return self._pipeline.fit_transform(X)

    def inverse_transform(self, X):
        return self._pipeline.inverse_transform(X)
