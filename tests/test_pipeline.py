"""Tests for the full preprocessing pipeline."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import QuantumScaler, FeatureReducer


class TestQuantumScaler:
    def test_fit_transform(self):
        X = np.random.rand(50, 10) * 100
        scaler = QuantumScaler(feature_range=(0, 2))
        X_scaled = scaler.fit_transform(X)

        assert X_scaled.shape == X.shape
        assert X_scaled.min() >= 0 - 1e-10
        assert X_scaled.max() <= 2 + 1e-10

    def test_transform_consistency(self):
        X = np.random.rand(50, 5)
        scaler = QuantumScaler(feature_range=(0, 2))
        X1 = scaler.fit_transform(X)
        X2 = scaler.transform(X)
        np.testing.assert_array_almost_equal(X1, X2)

    def test_custom_range(self):
        X = np.random.rand(20, 3)
        scaler = QuantumScaler(feature_range=(0, np.pi))
        X_scaled = scaler.fit_transform(X)

        assert X_scaled.min() >= 0 - 1e-10
        assert X_scaled.max() <= np.pi + 1e-10


class TestFeatureReducer:
    def test_pca_reduction(self):
        X = np.random.rand(100, 20)
        reducer = FeatureReducer(n_components=5)
        X_reduced = reducer.fit_transform(X)

        assert X_reduced.shape == (100, 5)

    def test_explained_variance(self):
        X = np.random.rand(100, 20)
        reducer = FeatureReducer(n_components=10)
        reducer.fit_transform(X)

        ev = reducer.get_explained_variance()
        assert "per_component" in ev
        assert "cumulative" in ev
        assert len(ev["per_component"]) == 10
        assert ev["cumulative"][-1] == pytest.approx(ev["total"])

    def test_variance_before_fit_raises(self):
        reducer = FeatureReducer(n_components=5)
        with pytest.raises(RuntimeError):
            reducer.get_explained_variance()
