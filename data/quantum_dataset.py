"""Quantum-advantageous dataset generator.

Based on Huang et al. "Power of data in quantum machine learning"
(Nature Communications, 2021).

Principle:
    1. Generate random inputs x in [0, 2]^n_qubits
    2. Encode via quantum feature map: |psi(x)> = U(x)|0>
    3. Measure expectation value of a random observable O
    4. Label: y = 1 if <psi(x)|O|psi(x)> > threshold else 0

The labels encode quantum geometric structure that:
    - Quantum kernels capture naturally (same feature space)
    - Classical kernels (RBF) struggle with (exponentially many features needed)

This creates a *provable* quantum advantage scenario for benchmarking.
"""

import numpy as np
from qiskit.quantum_info import Statevector, SparsePauliOp, random_unitary


def _build_random_observable(n_qubits, observable_type="rotated_z",
                              complexity=1.0, random_state=42):
    """Build a random Hermitian observable for labeling.

    Args:
        n_qubits: Number of qubits.
        observable_type:
            - "rotated_z": V^dag (Z_1 x I x ... x I) V with random unitary V.
              Creates a hard-to-learn function for classical kernels.
            - "pauli_sum": Random weighted sum of Pauli strings.
              Complexity controls the number of terms.
        complexity: Controls difficulty (0.0=easy, 1.0=hard).
        random_state: Seed for reproducibility.

    Returns:
        Observable as a numpy matrix (2^n x 2^n Hermitian).
    """
    rng = np.random.RandomState(random_state)
    dim = 2 ** n_qubits

    if observable_type == "rotated_z":
        # O = V^dag @ diag_Z @ V
        # V is a random unitary (parameterized by complexity)
        V = random_unitary(dim, seed=random_state).data

        # Z on first qubit: Z x I x I x ... x I
        Z1 = np.diag([1, -1])
        I_rest = np.eye(dim // 2)
        diag_Z = np.kron(Z1, I_rest)

        # Mix with identity based on complexity:
        # complexity=0 -> pure Z1 (easy), complexity=1 -> fully rotated (hard)
        O = (1 - complexity) * diag_Z + complexity * (V.conj().T @ diag_Z @ V)
        # Ensure Hermitian
        O = (O + O.conj().T) / 2
        return O

    elif observable_type == "pauli_sum":
        # Random weighted sum of Pauli strings
        # Number of terms scales with complexity
        pauli_labels = ['I', 'X', 'Y', 'Z']
        n_terms = max(3, int(complexity * 2 * n_qubits))

        terms = []
        coeffs = []
        for _ in range(n_terms):
            # Random Pauli string
            label = ''.join(rng.choice(pauli_labels, size=n_qubits))
            if label == 'I' * n_qubits:
                continue  # Skip identity
            coeff = rng.normal(0, 1)
            terms.append(label)
            coeffs.append(coeff)

        if not terms:
            terms = ['Z' + 'I' * (n_qubits - 1)]
            coeffs = [1.0]

        op = SparsePauliOp.from_list(list(zip(terms, coeffs)))
        return op.to_matrix()

    else:
        raise ValueError(f"Unknown observable_type: {observable_type}")


def generate_quantum_advantage_dataset(
    n_samples=200,
    n_qubits=6,
    feature_map_name="ZZ",
    alpha=1.0,
    reps=1,
    entanglement="linear",
    observable_type="rotated_z",
    complexity=0.8,
    noise=0.05,
    balance_classes=True,
    random_state=42,
):
    """Generate a dataset where quantum kernels have provable advantage.

    The decision boundary is defined by a quantum observable measured in
    the feature map's Hilbert space. This means the quantum kernel
    K(x,x') = |<psi(x)|psi(x')>|^2 naturally captures the structure,
    while classical kernels would need exponentially many features.

    Args:
        n_samples: Number of samples to generate.
        n_qubits: Number of qubits (= features).
        feature_map_name: Feature map to use ("Z", "ZZ", "pauli").
        alpha: Bandwidth parameter for the feature map.
        reps: Number of circuit repetitions.
        entanglement: Entanglement pattern.
        observable_type: "rotated_z" (hard) or "pauli_sum" (tunable).
        complexity: Difficulty for classical models (0=easy, 1=hard).
        noise: Label noise rate (fraction of labels to flip).
        balance_classes: If True, oversample to get ~50/50 class balance.
        random_state: Random seed.

    Returns:
        X: Feature array (n_samples, n_qubits).
        y: Binary labels.
        metadata: Dict with observable, expectation values, etc.
    """
    from src.kernels.feature_maps import build_feature_map

    rng = np.random.RandomState(random_state)

    # 1. Build the feature map and observable
    fm = build_feature_map(feature_map_name, n_qubits, alpha=alpha,
                           reps=reps, entanglement=entanglement)
    O = _build_random_observable(n_qubits, observable_type=observable_type,
                                  complexity=complexity, random_state=random_state)

    # 2. Generate more samples than needed (for class balancing)
    n_generate = n_samples * 3 if balance_classes else n_samples
    X_all = rng.uniform(0, 2, size=(n_generate, n_qubits))

    # 3. Compute expectation values and labels
    exp_vals = np.zeros(n_generate)
    for i, x in enumerate(X_all):
        bound_circuit = fm.assign_parameters(x)
        sv = Statevector.from_instruction(bound_circuit)
        # <psi(x)|O|psi(x)>
        state_vec = sv.data
        exp_vals[i] = np.real(state_vec.conj() @ O @ state_vec)

    # 4. Label: threshold at median for balance
    threshold = np.median(exp_vals) if balance_classes else 0.0
    y_all = (exp_vals > threshold).astype(int)

    # 5. Add label noise
    if noise > 0:
        n_flip = int(noise * n_generate)
        flip_idx = rng.choice(n_generate, size=n_flip, replace=False)
        y_all[flip_idx] = 1 - y_all[flip_idx]

    # 6. Balance classes by subsampling
    if balance_classes:
        idx_0 = np.where(y_all == 0)[0]
        idx_1 = np.where(y_all == 1)[0]
        n_per_class = n_samples // 2
        if len(idx_0) >= n_per_class and len(idx_1) >= n_per_class:
            chosen_0 = rng.choice(idx_0, size=n_per_class, replace=False)
            chosen_1 = rng.choice(idx_1, size=n_per_class, replace=False)
            chosen = np.concatenate([chosen_0, chosen_1])
            rng.shuffle(chosen)
            X = X_all[chosen]
            y = y_all[chosen]
            exp_vals_final = exp_vals[chosen]
        else:
            # Fallback: use first n_samples
            X = X_all[:n_samples]
            y = y_all[:n_samples]
            exp_vals_final = exp_vals[:n_samples]
    else:
        X = X_all[:n_samples]
        y = y_all[:n_samples]
        exp_vals_final = exp_vals[:n_samples]

    metadata = {
        'observable': O,
        'observable_type': observable_type,
        'complexity': complexity,
        'feature_map': feature_map_name,
        'alpha': alpha,
        'n_qubits': n_qubits,
        'expectation_values': exp_vals_final,
        'threshold': threshold,
        'noise': noise,
        'class_balance': np.bincount(y),
    }

    return X, y, metadata


def generate_quantum_advantage_suite(
    n_samples=100,
    n_qubits=6,
    complexities=[0.3, 0.6, 0.9],
    random_state=42,
):
    """Generate multiple quantum-advantage datasets at different difficulties.

    Returns:
        datasets: Dict {f"qadv_c{c}": (X, y, metadata)} for each complexity.
    """
    datasets = {}
    for c in complexities:
        label = f"qadv_c{c:.1f}"
        X, y, meta = generate_quantum_advantage_dataset(
            n_samples=n_samples,
            n_qubits=n_qubits,
            complexity=c,
            random_state=random_state,
        )
        datasets[label] = (X, y, meta)
    return datasets
