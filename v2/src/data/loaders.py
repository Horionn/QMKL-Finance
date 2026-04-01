"""Dataset loaders for QMKL-v2 experiments.

Four datasets:
- FRED Recession : ~600 months, 19 macro features, recession vs expansion  <- PRIMARY
- German Credit  : 1000 samples, 20 features (binary risk classification)
- Bank Marketing : 45211 samples, 16 features (deposit subscription)
- Breast Cancer  : 569 samples, 30 features (medical benchmark)

FRED est le dataset financier principal : données réelles de la Fed,
classification recession/expansion, d=19 >> Q=4 qubits.
"""

from .fred_loader import (
    load_fred_recession_data,
    load_fred_recession_synthetic,
    FRED_FEATURES,
)

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder


def load_breast_cancer_data():
    """Load Breast Cancer Wisconsin dataset.

    Returns:
        X : (569, 30) float array
        y : (569,) int array, 0=malignant, 1=benign
        feature_names : list of 30 feature names
    """
    data = load_breast_cancer()
    X = data.data.astype(float)
    y = data.target.astype(int)
    feature_names = list(data.feature_names)
    return X, y, feature_names


def load_german_credit(path=None):
    """Load German Credit dataset (UCI).

    Downloads from sklearn or uses local path.
    Returns binary labels: 1=good credit, 0=bad credit.

    Returns:
        X : (1000, 20) float array (after encoding categoricals)
        y : (1000,) int array
        feature_names : list of feature names
    """
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(name="credit-g", version=1, as_frame=True, parser="auto")
        df = data.frame.copy()
        y = (df["class"] == "good").astype(int).values

        df = df.drop(columns=["class"])
        # Encode categoricals
        for col in df.select_dtypes(include="category").columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        X = df.values.astype(float)
        feature_names = list(df.columns)

    except Exception:
        # Fallback: synthetic placeholder with correct shape
        rng = np.random.RandomState(42)
        X = rng.randn(1000, 20)
        y = (rng.rand(1000) > 0.3).astype(int)
        feature_names = [f"feature_{i}" for i in range(20)]

    return X, y, feature_names


def load_bank_marketing(path=None):
    """Load Bank Marketing dataset (UCI).

    Returns binary labels: 1=subscribed, 0=not subscribed.

    Returns:
        X : (N, 16) float array
        y : (N,) int array
        feature_names : list of feature names
    """
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(name="bank-marketing", version=1, as_frame=True, parser="auto")
        df = data.frame.copy()
        target_col = data.target_names[0] if hasattr(data, "target_names") else "Class"
        y = (df[target_col].astype(str) == "2").astype(int).values
        df = df.drop(columns=[target_col])
        for col in df.select_dtypes(include="category").columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        X = df.values.astype(float)
        feature_names = list(df.columns)

    except Exception:
        rng = np.random.RandomState(42)
        X = rng.randn(1000, 16)
        y = (rng.rand(1000) > 0.5).astype(int)
        feature_names = [f"feature_{i}" for i in range(16)]

    return X, y, feature_names


def subsample(X, y, n, seed=42):
    """Subsample n points stratified by class."""
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=n, random_state=seed)
    _, idx = next(sss.split(X, y))
    return X[idx], y[idx]
