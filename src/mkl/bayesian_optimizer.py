"""Bayesian Optimization for kernel weights and hyperparameters.

Based on the BO-MKQSVM paper (MLPRAE 2025).
Uses scikit-optimize to find optimal kernel combination weights
and SVM regularization parameter.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


class BayesianKernelOptimizer:
    """Bayesian Optimization of QMKL kernel weights and SVM hyperparameters.

    Optimizes:
    - alpha_m: weights for each quantum kernel (sum to 1)
    - C: SVM regularization parameter

    Uses cross-validation accuracy as the objective function.
    """

    def __init__(
        self,
        n_calls=50,
        n_initial_points=10,
        acq_func="EI",
        cv_folds=5,
        random_state=42,
        optimize_C=True,
        C_range=(0.01, 100),
    ):
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.acq_func = acq_func
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.optimize_C = optimize_C
        self.C_range = C_range
        self.result_ = None
        self.best_weights_ = None
        self.best_C_ = None

    def optimize(self, kernel_matrices, y_train, scoring="accuracy"):
        """Run Bayesian Optimization to find optimal weights.

        Args:
            kernel_matrices: List of precomputed training kernel matrices.
            y_train: Training labels.
            scoring: Sklearn scoring metric.

        Returns:
            Optimal weights array and C value.
        """
        from skopt import gp_minimize
        from skopt.space import Real

        n_kernels = len(kernel_matrices)
        K_list = [np.array(K) for K in kernel_matrices]

        # Define search space: weights in [0, 1] for each kernel
        dimensions = [Real(0.0, 1.0, name=f"w_{i}") for i in range(n_kernels)]
        if self.optimize_C:
            dimensions.append(Real(self.C_range[0], self.C_range[1], name="C", prior="log-uniform"))

        def objective(params):
            # Extract weights and normalize to sum to 1
            raw_weights = np.array(params[:n_kernels])
            w_sum = np.sum(raw_weights)
            if w_sum == 0:
                weights = np.ones(n_kernels) / n_kernels
            else:
                weights = raw_weights / w_sum

            C = params[n_kernels] if self.optimize_C else 1.0

            # Combine kernels
            K_combined = np.zeros_like(K_list[0])
            for w, K in zip(weights, K_list):
                K_combined += w * K

            # Ensure positive semi-definite
            min_eig = np.min(np.linalg.eigvalsh(K_combined))
            if min_eig < 0:
                K_combined += (abs(min_eig) + 1e-10) * np.eye(K_combined.shape[0])

            # Train SVM with precomputed kernel and evaluate via CV
            svm = SVC(kernel="precomputed", C=C)
            try:
                scores = cross_val_score(
                    svm, K_combined, y_train,
                    cv=self.cv_folds, scoring=scoring,
                )
                return -np.mean(scores)  # Minimize negative score
            except Exception:
                return 0.0  # Worst case

        self.result_ = gp_minimize(
            objective,
            dimensions,
            n_calls=self.n_calls,
            n_initial_points=self.n_initial_points,
            acq_func=self.acq_func,
            random_state=self.random_state,
        )

        # Extract best parameters
        best_params = self.result_.x
        raw_weights = np.array(best_params[:n_kernels])
        w_sum = np.sum(raw_weights)
        self.best_weights_ = raw_weights / w_sum if w_sum > 0 else np.ones(n_kernels) / n_kernels
        self.best_C_ = best_params[n_kernels] if self.optimize_C else 1.0

        return self.best_weights_, self.best_C_

    def get_convergence_info(self):
        """Return optimization convergence information."""
        if self.result_ is None:
            raise RuntimeError("Call optimize() first.")

        return {
            "best_score": -self.result_.fun,
            "best_weights": self.best_weights_,
            "best_C": self.best_C_,
            "n_iterations": len(self.result_.func_vals),
            "convergence": [-v for v in self.result_.func_vals],
        }
