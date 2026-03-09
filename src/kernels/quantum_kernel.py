"""Quantum kernel computation using Qiskit.

Supports both fidelity and projected quantum kernels.
"""

import numpy as np
from qiskit_machine_learning.kernels import FidelityQuantumKernel


def build_quantum_kernel(feature_map, kernel_type="fidelity", gamma=1.0):
    """Build a quantum kernel from a feature map.

    Args:
        feature_map: A Qiskit QuantumCircuit feature map.
        kernel_type: "fidelity" or "projected".
        gamma: Gamma parameter for projected kernel.

    Returns:
        A Qiskit quantum kernel object.
    """
    if kernel_type == "fidelity":
        kernel = FidelityQuantumKernel(feature_map=feature_map)
    elif kernel_type == "projected":
        # For projected kernel, we compute 1-RDMs and use RBF on them
        # This is a custom implementation following the IBM paper
        kernel = ProjectedQuantumKernel(feature_map=feature_map, gamma=gamma)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}. Use 'fidelity' or 'projected'.")

    return kernel


class ProjectedQuantumKernel:
    """Projected quantum kernel following Huang et al.

    K_PQ(x, x') = exp(-gamma * sum_k ||rho_k(x) - rho_k(x')||_F^2)

    Instead of computing the full fidelity between quantum states,
    this kernel projects each state onto single-qubit reduced density
    matrices and computes an RBF kernel on those classical representations.
    """

    def __init__(self, feature_map, gamma=1.0):
        self.feature_map = feature_map
        self.gamma = gamma
        self.n_qubits = feature_map.num_qubits

    def evaluate(self, x_vec, y_vec=None):
        """Compute the projected quantum kernel matrix.

        Args:
            x_vec: Array of shape (n_samples_x, n_features).
            y_vec: Array of shape (n_samples_y, n_features). If None, use x_vec.

        Returns:
            Kernel matrix of shape (n_samples_x, n_samples_y).
        """
        from qiskit.quantum_info import Statevector

        if y_vec is None:
            y_vec = x_vec

        x_projections = self._compute_projections(x_vec)
        y_projections = self._compute_projections(y_vec)

        n_x = len(x_vec)
        n_y = len(y_vec)
        K = np.zeros((n_x, n_y))

        for i in range(n_x):
            for j in range(n_y):
                dist_sq = 0.0
                for k in range(self.n_qubits):
                    diff = x_projections[i][k] - y_projections[j][k]
                    dist_sq += np.real(np.trace(diff @ diff.conj().T))
                K[i, j] = np.exp(-self.gamma * dist_sq)

        return K

    def _compute_projections(self, x_vec):
        """Compute 1-RDMs for each data point and each qubit."""
        from qiskit.quantum_info import Statevector, partial_trace

        projections = []
        for x in x_vec:
            # Bind parameters and get statevector
            bound_circuit = self.feature_map.assign_parameters(x)
            sv = Statevector.from_instruction(bound_circuit)
            rho = sv.to_operator().data

            # Compute 1-RDM for each qubit
            qubit_rdms = []
            all_qubits = list(range(self.n_qubits))
            for k in range(self.n_qubits):
                trace_out = [q for q in all_qubits if q != k]
                rdm = partial_trace(sv, trace_out).data
                qubit_rdms.append(rdm)

            projections.append(qubit_rdms)

        return projections
