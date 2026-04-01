"""Preprocessing pipeline for quantum feature encoding."""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline


class QuantumScaler:
    """Standardize then scale to [0, 2] for quantum rotation angles.

    Pipeline: StandardScaler → MinMaxScaler([0, 2])
    Following IBM QMKL paper convention.
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
