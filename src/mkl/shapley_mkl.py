"""Shapley-QMKL: Game-theoretic weight assignment for Quantum Multiple Kernel Learning.

Computes exact Shapley values to determine the marginal contribution of each
quantum kernel in an MKL combination. This provides axiomatic weight assignment
with theoretical guarantees (efficiency, symmetry, null-player, additivity).

Key idea: instead of optimizing an alignment objective (CKA, SDP...),
we compute each kernel's contribution via cooperative game theory:

    phi_m = sum_{S ⊆ M\\{m}} [|S|!(|M|-|S|-1)! / |M|!] * [v(S∪{m}) - v(S)]

where v(S) = AUC of SVM trained on the combined kernel K_S = mean(K_i, i ∈ S).

Also computes Shapley interaction indices to detect synergistic/redundant pairs:

    delta_{ij} = sum_{S ⊆ M\\{i,j}} weight * [v(S∪{i,j}) - v(S∪{i}) - v(S∪{j}) + v(S)]

References:
- Shapley (1953): A value for n-person games
- Lundberg & Lee (2017): SHAP — applied to predictions, not kernel weights
- This work: first application to quantum MKL weight assignment
"""

import numpy as np
from itertools import combinations
from math import factorial
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score


class ShapleyMKL:
    """Shapley-value based weight assignment for Multiple Kernel Learning.

    Computes exact or Monte Carlo-approximated Shapley values for each kernel
    in an MKL ensemble, providing axiomatically grounded weights.

    Parameters
    ----------
    scoring : str
        Metric for coalition value function: 'roc_auc' or 'accuracy'.
    n_cv : int
        Number of cross-validation folds.
    C : float
        SVM regularization parameter.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Print progress during computation.
    """

    def __init__(self, scoring='roc_auc', n_cv=5, C=1.0, seed=42, verbose=True):
        self.scoring = scoring
        self.n_cv = n_cv
        self.C = C
        self.seed = seed
        self.verbose = verbose

        # Internal state
        self._coalition_cache = {}
        self._shapley_values = None
        self._interaction_indices = None
        self._n_coalitions_evaluated = 0
        self._kernel_names = None

    # ──────────────────────────────────────────────────────────────────────
    # COALITION VALUE FUNCTION
    # ──────────────────────────────────────────────────────────────────────

    def _coalition_key(self, indices):
        """Canonical key for a coalition (frozenset for hashing)."""
        return tuple(sorted(indices))

    def _ensure_psd(self, K, eps=1e-8):
        """Fix non-PSD matrices by shifting eigenvalues."""
        ev = np.linalg.eigvalsh(K)
        if ev.min() < -eps:
            K = K + (abs(ev.min()) + eps) * np.eye(K.shape[0])
        return K

    def evaluate_coalition(self, K_list, indices, y):
        """Compute v(S) = cross-validated AUC for a kernel coalition.

        Parameters
        ----------
        K_list : list of np.ndarray
            List of kernel matrices (N x N).
        indices : list of int
            Indices of kernels in the coalition.
        y : np.ndarray
            Labels.

        Returns
        -------
        float
            Coalition value (AUC or accuracy).
        """
        key = self._coalition_key(indices)
        if key in self._coalition_cache:
            return self._coalition_cache[key]

        # Empty coalition: random performance baseline
        if len(indices) == 0:
            val = 0.5 if self.scoring == 'roc_auc' else 0.0
            self._coalition_cache[key] = val
            return val

        # Combine kernels (uniform within coalition)
        K_combined = np.mean([K_list[i] for i in indices], axis=0)

        # Cross-validated evaluation
        skf = StratifiedKFold(
            n_splits=self.n_cv, shuffle=True, random_state=self.seed
        )
        scores = []
        for tr, te in skf.split(np.zeros(len(y)), y):
            K_tr = self._ensure_psd(K_combined[np.ix_(tr, tr)])
            K_te = K_combined[np.ix_(te, tr)]
            svm = SVC(
                kernel='precomputed', C=self.C, probability=True,
                random_state=self.seed
            )
            svm.fit(K_tr, y[tr])
            if self.scoring == 'roc_auc':
                proba = svm.predict_proba(K_te)[:, 1]
                scores.append(roc_auc_score(y[te], proba))
            else:
                scores.append(accuracy_score(y[te], svm.predict(K_te)))

        val = np.mean(scores)
        self._coalition_cache[key] = val
        self._n_coalitions_evaluated += 1
        return val

    # ──────────────────────────────────────────────────────────────────────
    # EXACT SHAPLEY VALUES
    # ──────────────────────────────────────────────────────────────────────

    def compute_shapley_values(self, K_list, y, kernel_names=None):
        """Compute exact Shapley values for each kernel.

        Complexity: O(2^M) coalition evaluations (cached), where M = len(K_list).
        Feasible for M ≤ 15.

        Parameters
        ----------
        K_list : list of np.ndarray
            M kernel matrices, each (N x N).
        y : np.ndarray
            Binary labels (N,).
        kernel_names : list of str, optional
            Names for display.

        Returns
        -------
        np.ndarray
            Shapley values (M,), one per kernel.
        """
        M = len(K_list)
        self._coalition_cache = {}
        self._n_coalitions_evaluated = 0
        self._kernel_names = kernel_names or [f'K{i}' for i in range(M)]

        if self.verbose:
            total = 2 ** M
            print(f'Shapley-QMKL: M={M} kernels, {total} coalitions à évaluer')

        shapley = np.zeros(M)

        for m in range(M):
            others = [i for i in range(M) if i != m]
            phi_m = 0.0

            for size in range(M):  # |S| from 0 to M-1
                weight = factorial(size) * factorial(M - size - 1) / factorial(M)
                for S in combinations(others, size):
                    S_list = list(S)
                    v_with = self.evaluate_coalition(K_list, S_list + [m], y)
                    v_without = self.evaluate_coalition(K_list, S_list, y)
                    phi_m += weight * (v_with - v_without)

            shapley[m] = phi_m
            if self.verbose:
                print(f'  φ({self._kernel_names[m]:20s}) = {phi_m:+.6f}')

        self._shapley_values = shapley

        if self.verbose:
            print(f'\nCoalitions évaluées: {self._n_coalitions_evaluated} '
                  f'(cache hits: {2**M - self._n_coalitions_evaluated})')
            grand = self.evaluate_coalition(K_list, list(range(M)), y)
            print(f'v(grand coalition) = {grand:.4f}')
            print(f'Σ φ_m + v(∅) = {shapley.sum() + 0.5:.4f}  '
                  f'(efficacité: Σφ = v(M)-v(∅) = {grand - 0.5:.4f})')

        return shapley

    # ──────────────────────────────────────────────────────────────────────
    # MONTE CARLO APPROXIMATION
    # ──────────────────────────────────────────────────────────────────────

    def compute_shapley_montecarlo(self, K_list, y, n_permutations=1000,
                                    kernel_names=None):
        """Approximate Shapley values via permutation sampling.

        For M > 12 kernels where exact computation is infeasible.

        Parameters
        ----------
        K_list : list of np.ndarray
            M kernel matrices.
        y : np.ndarray
            Labels.
        n_permutations : int
            Number of random permutations to sample.
        kernel_names : list of str, optional

        Returns
        -------
        np.ndarray
            Approximate Shapley values (M,).
        """
        M = len(K_list)
        rng = np.random.RandomState(self.seed)
        self._coalition_cache = {}
        self._n_coalitions_evaluated = 0
        self._kernel_names = kernel_names or [f'K{i}' for i in range(M)]

        if self.verbose:
            print(f'Shapley MC: M={M}, {n_permutations} permutations')

        shapley = np.zeros(M)

        for p in range(n_permutations):
            perm = rng.permutation(M)
            for pos in range(M):
                m = perm[pos]
                prefix = list(perm[:pos])
                v_with = self.evaluate_coalition(K_list, prefix + [m], y)
                v_without = self.evaluate_coalition(K_list, prefix, y)
                shapley[m] += (v_with - v_without)

            if self.verbose and (p + 1) % 100 == 0:
                print(f'  Permutation {p+1}/{n_permutations}...')

        shapley /= n_permutations
        self._shapley_values = shapley

        if self.verbose:
            for m in range(M):
                print(f'  φ({self._kernel_names[m]:20s}) = {shapley[m]:+.6f}')

        return shapley

    # ──────────────────────────────────────────────────────────────────────
    # WEIGHTS
    # ──────────────────────────────────────────────────────────────────────

    def get_weights(self, K_list=None, y=None, kernel_names=None):
        """Get normalized MKL weights from Shapley values.

        Negative Shapley values are clipped to 0 (harmful kernels excluded).

        Returns
        -------
        np.ndarray
            Normalized weights (M,), summing to 1.
        """
        if self._shapley_values is None:
            if K_list is None or y is None:
                raise ValueError("Compute Shapley values first")
            self.compute_shapley_values(K_list, y, kernel_names)

        sv = self._shapley_values.copy()
        w = np.maximum(sv, 0.0)
        if w.sum() > 1e-12:
            w /= w.sum()
        else:
            w = np.ones(len(sv)) / len(sv)
        return w

    # ──────────────────────────────────────────────────────────────────────
    # INTERACTION INDICES
    # ──────────────────────────────────────────────────────────────────────

    def compute_interaction_indices(self, K_list, y, kernel_names=None):
        """Compute Shapley interaction indices for all kernel pairs.

        Positive interaction = synergy (pair performs better than sum of parts).
        Negative interaction = redundancy (kernels capture same information).

        Parameters
        ----------
        K_list : list of np.ndarray
        y : np.ndarray

        Returns
        -------
        np.ndarray
            Interaction matrix (M x M), symmetric, zero diagonal.
        """
        M = len(K_list)
        self._kernel_names = kernel_names or [f'K{i}' for i in range(M)]

        if not self._coalition_cache:
            self.compute_shapley_values(K_list, y, kernel_names)

        if self.verbose:
            print(f'Computing {M*(M-1)//2} interaction indices...')

        interactions = np.zeros((M, M))

        for i in range(M):
            for j in range(i + 1, M):
                others = [k for k in range(M) if k != i and k != j]
                delta = 0.0

                for size in range(M - 1):  # |S| from 0 to M-2
                    if size > len(others):
                        break
                    weight = (factorial(size) * factorial(M - size - 2)
                              / factorial(M - 1))
                    for S in combinations(others, size):
                        S_list = list(S)
                        v_ij = self.evaluate_coalition(K_list, S_list + [i, j], y)
                        v_i = self.evaluate_coalition(K_list, S_list + [i], y)
                        v_j = self.evaluate_coalition(K_list, S_list + [j], y)
                        v_0 = self.evaluate_coalition(K_list, S_list, y)
                        delta += weight * (v_ij - v_i - v_j + v_0)

                interactions[i, j] = delta
                interactions[j, i] = delta

        self._interaction_indices = interactions

        if self.verbose:
            print('Paires les plus synergiques :')
            pairs = [(i, j, interactions[i, j])
                     for i in range(M) for j in range(i+1, M)]
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            for i, j, val in pairs[:5]:
                kind = 'synergie' if val > 0 else 'redondance'
                print(f'  ({self._kernel_names[i]}, {self._kernel_names[j]}): '
                      f'δ={val:+.4f} ({kind})')

        return interactions

    # ──────────────────────────────────────────────────────────────────────
    # ANALYSIS UTILITIES
    # ──────────────────────────────────────────────────────────────────────

    def get_coalition_landscape(self, K_list, y):
        """Get v(S) for all possible coalition sizes.

        Returns dict: {size: [list of v(S) for all S of that size]}
        """
        M = len(K_list)
        # Ensure all coalitions are evaluated
        if not self._coalition_cache:
            self.compute_shapley_values(K_list, y)

        landscape = {}
        for size in range(M + 1):
            vals = []
            for S in combinations(range(M), size):
                val = self.evaluate_coalition(K_list, list(S), y)
                vals.append(val)
            landscape[size] = vals
        return landscape

    def get_marginal_matrix(self, K_list, y):
        """Compute marginal contribution of each kernel to each coalition size.

        Returns
        -------
        np.ndarray
            Shape (M, M) where entry [m, s] = avg marginal contribution
            of kernel m when added to a coalition of size s.
        """
        M = len(K_list)
        if not self._coalition_cache:
            self.compute_shapley_values(K_list, y)

        marginals = np.zeros((M, M))
        counts = np.zeros((M, M))

        for m in range(M):
            others = [i for i in range(M) if i != m]
            for size in range(M):
                for S in combinations(others, size):
                    S_list = list(S)
                    v_with = self.evaluate_coalition(K_list, S_list + [m], y)
                    v_without = self.evaluate_coalition(K_list, S_list, y)
                    marginals[m, size] += (v_with - v_without)
                    counts[m, size] += 1

        # Average
        mask = counts > 0
        marginals[mask] /= counts[mask]
        return marginals

    @property
    def shapley_values(self):
        return self._shapley_values

    @property
    def interaction_indices(self):
        return self._interaction_indices

    @property
    def n_coalitions_evaluated(self):
        return self._n_coalitions_evaluated
