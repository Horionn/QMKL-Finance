from .combiner import MultipleKernelCombiner
from .alignment import kernel_target_alignment, centered_alignment, projection_alignment
from .bayesian_optimizer import BayesianKernelOptimizer
from .shapley_mkl import ShapleyMKL
from .riemannian_mkl import (RiemannianQMKL, frechet_mean, log_euclidean_mean,
                              riemannian_dist, geodesic_path,
                              riemannian_multi_run_evaluation)
