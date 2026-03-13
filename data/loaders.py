"""Data loaders for financial classification datasets."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.utils import resample


def load_german_credit(n_samples=None, random_state=42):
    """Load the German Credit dataset.

    Binary classification: good (1) vs bad (0) credit risk.
    20 features, 1000 instances.

    Uses synthetic data if OpenML is unavailable.
    """
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(data_id=31, as_frame=True, parser="auto")
        X = pd.get_dummies(data.data, drop_first=True).values.astype(np.float64)
        y = (data.target == "good").astype(int).values
    except Exception as e:
        print(f"⚠️  OpenML failed ({type(e).__name__}). Using synthetic data instead.")
        # Generate synthetic financial data as fallback
        X, y = make_classification(
            n_samples=1000, n_features=20, n_informative=12,
            n_redundant=5, random_state=random_state,
            weights=[0.7, 0.3],  # Imbalanced like real credit data
        )
        X = X.astype(np.float64)

    if n_samples and n_samples < len(X):
        X, y = resample(X, y, n_samples=n_samples, random_state=random_state, stratify=y)

    return X, y


def load_bank_marketing(n_samples=None, random_state=42):
    """Load the Bank Marketing dataset.

    Binary classification: client subscribes term deposit or not.
    16 features, ~45k instances.

    Uses synthetic data if OpenML is unavailable.
    """
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(data_id=1461, as_frame=True, parser="auto")
        X = pd.get_dummies(data.data, drop_first=True).values.astype(np.float64)
        y = (data.target == "2").astype(int).values
    except Exception as e:
        print(f"⚠️  OpenML failed ({type(e).__name__}). Using synthetic data instead.")
        # Generate synthetic financial data
        X, y = make_classification(
            n_samples=5000, n_features=16, n_informative=10,
            n_redundant=4, random_state=random_state,
            weights=[0.88, 0.12],  # Highly imbalanced like real marketing
        )
        X = X.astype(np.float64)

    if n_samples and n_samples < len(X):
        X, y = resample(X, y, n_samples=n_samples, random_state=random_state, stratify=y)

    return X, y


def load_iris_binary(n_samples=None, random_state=42):
    """Load Iris dataset reduced to binary classification (setosa vs non-setosa)."""
    from sklearn.datasets import load_iris

    data = load_iris()
    X = data.data.astype(np.float64)
    y = (data.target == 0).astype(int)

    if n_samples and n_samples < len(X):
        X, y = resample(X, y, n_samples=n_samples, random_state=random_state, stratify=y)

    return X, y


def load_custom_csv(path, target_column, n_samples=None, random_state=42):
    """Load a custom CSV dataset.

    Args:
        path: Path to CSV file.
        target_column: Name of the target column.
        n_samples: Optional subsample size.
        random_state: Random seed.
    """
    df = pd.read_csv(path)
    y = df[target_column].values.astype(int)
    X = df.drop(columns=[target_column]).values.astype(np.float64)

    if n_samples and n_samples < len(X):
        X, y = resample(X, y, n_samples=n_samples, random_state=random_state, stratify=y)

    return X, y


def load_synthetic(n_samples=None, random_state=42):
    """Generate a synthetic binary classification dataset.

    Mimics a financial fraud detection scenario with moderate difficulty.
    """
    n = n_samples or 500
    X, y = make_classification(
        n_samples=n, n_features=15, n_informative=8,
        n_redundant=4, n_clusters_per_class=2,
        flip_y=0.05, class_sep=0.8,
        weights=[0.6, 0.4], random_state=random_state,
    )
    return X.astype(np.float64), y


DATASET_REGISTRY = {
    "german_credit": load_german_credit,
    "bank_marketing": load_bank_marketing,
    "iris_binary": load_iris_binary,
    "synthetic": load_synthetic,
}


def load_dataset(name, n_samples=None, random_state=42, **kwargs):
    """Load a dataset by name.

    Args:
        name: Dataset name from registry or "custom".
        n_samples: Optional subsample size.
        random_state: Random seed.
        **kwargs: Additional arguments (e.g., path, target_column for custom).
    """
    if name == "custom":
        return load_custom_csv(
            path=kwargs["custom_path"],
            target_column=kwargs["target_column"],
            n_samples=n_samples,
            random_state=random_state,
        )

    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")

    return DATASET_REGISTRY[name](n_samples=n_samples, random_state=random_state)
