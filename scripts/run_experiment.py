"""Run a full QMKL experiment from a config file."""

import argparse
import yaml
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split

from data.loaders import load_dataset
from src.preprocessing import QuantumScaler, FeatureReducer
from src.kernels import build_feature_map, build_quantum_kernel, compute_kernel_matrix
from src.kernels.feature_maps import get_feature_map_library
from src.mkl import MultipleKernelCombiner
from src.mkl.bayesian_optimizer import BayesianKernelOptimizer
from src.models import QSVM
from src.evaluation.metrics import compute_all_metrics, print_results


def load_config(config_path, base_config_path=None):
    """Load config, optionally merging with a base config."""
    if base_config_path:
        with open(base_config_path) as f:
            config = yaml.safe_load(f)
        with open(config_path) as f:
            override = yaml.safe_load(f)
        # Shallow merge
        for key, val in override.items():
            if isinstance(val, dict) and key in config:
                config[key].update(val)
            else:
                config[key] = val
    else:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    return config


def run_experiment(config):
    """Run a single QMKL experiment."""
    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]
    mkl_cfg = config["mkl"]
    eval_cfg = config["evaluation"]

    n_components = prep_cfg["n_components"]
    seed = data_cfg.get("random_seed", 42)

    # Load data
    print(f"Loading dataset: {data_cfg['dataset']}...")
    X, y = load_dataset(
        data_cfg["dataset"],
        n_samples=data_cfg.get("n_samples"),
        random_state=seed,
        **{k: v for k, v in data_cfg.items() if k not in ("dataset", "n_samples", "random_seed", "test_size", "n_splits", "n_random_draws")},
    )
    print(f"  Shape: {X.shape}, Classes: {np.bincount(y)}")

    # Preprocess
    reducer = FeatureReducer(n_components=n_components)
    scaler = QuantumScaler(feature_range=tuple(prep_cfg.get("feature_range", [0, 2])))
    X_processed = scaler.fit_transform(reducer.fit_transform(X))

    test_size = data_cfg.get("test_size", 0.33)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Compute kernel matrices
    n_qubits = X_train.shape[1]
    print(f"Computing quantum kernels ({n_qubits} qubits)...")
    fm_library = get_feature_map_library(n_qubits)

    kernel_matrices_train = []
    kernel_matrices_test = []
    kernel_names = []

    for label, fm in fm_library:
        qk = build_quantum_kernel(fm, kernel_type="fidelity")
        K_tr = compute_kernel_matrix(qk, X_train)
        K_te = compute_kernel_matrix(qk, X_train, X_test)
        kernel_matrices_train.append(K_tr)
        kernel_matrices_test.append(K_te)
        kernel_names.append(label)
        print(f"  {label} done")

    # MKL combination
    method = mkl_cfg["alignment_method"]
    print(f"\nCombining kernels with method: {method}")

    if method == "bayesian":
        bo_cfg = config.get("bayesian_optimization", {})
        bo = BayesianKernelOptimizer(
            n_calls=bo_cfg.get("n_calls", 50),
            n_initial_points=bo_cfg.get("n_initial_points", 10),
        )
        weights, best_C = bo.optimize(kernel_matrices_train, y_train)
        C = best_C
    else:
        combiner = MultipleKernelCombiner(method=method)
        combiner.fit(kernel_matrices_train, y_train)
        weights = combiner.get_weights()
        C = config.get("svm", {}).get("C", 1.0)

    # Combine kernels
    K_train = sum(w * K for w, K in zip(weights, kernel_matrices_train))
    K_test = sum(w * K for w, K in zip(weights, kernel_matrices_test))

    # Train and evaluate
    svm = QSVM(C=C)
    svm.fit(K_train, y_train)
    y_pred = svm.predict(K_test)
    y_proba = svm.predict_proba(K_test)

    metrics = compute_all_metrics(y_test, y_pred, y_proba)
    print_results(metrics, title=f"QMKL-{method.upper()} Results")

    # Print top kernels
    top_k = np.argsort(weights)[-5:][::-1]
    print("Top kernel weights:")
    for idx in top_k:
        if weights[idx] > 0.01:
            print(f"  {kernel_names[idx]}: {weights[idx]:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run QMKL experiment")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--base-config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config, args.base_config)
    run_experiment(config)


if __name__ == "__main__":
    main()
