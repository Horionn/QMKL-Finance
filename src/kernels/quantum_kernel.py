"""Quantum kernel computation using Qiskit.

Supports both fidelity and projected quantum kernels.

Performance tip: use build_quantum_kernel(..., backend="aer_parallel") to
enable multi-threaded Aer statevector simulation (much faster than the default
Qiskit statevector simulator for n_qubits >= 6).
"""

import numpy as np
from qiskit_machine_learning.kernels import FidelityQuantumKernel


def _make_aer_sampler():
    """Build a Qiskit Aer Sampler with multi-threading enabled.

    Falls back to the default Qiskit sampler if Aer is not installed.
    """
    try:
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import Sampler as AerSampler

        # Use statevector method — exact and fast for <= 20 qubits
        # max_parallel_threads=0 → use all available CPU cores
        sim = AerSimulator(
            method="statevector",
            max_parallel_threads=0,       # all cores
            max_parallel_experiments=0,   # all cores
            statevector_parallel_threshold=12,
        )
        sampler = AerSampler(backend_options={
            "method": "statevector",
            "max_parallel_threads": 0,
        })
        return sampler
    except ImportError:
        return None  # fallback to default


def build_quantum_kernel(feature_map, kernel_type="fidelity", gamma=1.0,
                         backend="aer"):
    """Build a quantum kernel from a feature map.

    Args:
        feature_map: A Qiskit QuantumCircuit feature map.
        kernel_type: "fidelity" or "projected".
        gamma: Gamma parameter for projected kernel.
        backend: "aer" (fast, uses Aer multi-thread) or "default" (Qiskit statevector).
                 Ignored for projected kernels (always uses statevector directly).

    Returns:
        A Qiskit quantum kernel object.
    """
    if kernel_type == "fidelity":
        if backend == "aer":
            sampler = _make_aer_sampler()
        else:
            sampler = None

        if sampler is not None:
            kernel = FidelityQuantumKernel(feature_map=feature_map,
                                           fidelity=None)
            # Inject the Aer sampler via Fidelity
            try:
                from qiskit_algorithms.state_fidelities import ComputeUncompute
                fidelity = ComputeUncompute(sampler=sampler)
                kernel = FidelityQuantumKernel(feature_map=feature_map,
                                               fidelity=fidelity)
            except Exception:
                # If qiskit_algorithms not available or API changed, use default
                kernel = FidelityQuantumKernel(feature_map=feature_map)
        else:
            kernel = FidelityQuantumKernel(feature_map=feature_map)

    elif kernel_type == "projected":
        # For projected kernel, we compute 1-RDMs and use RBF on them
        # This is a custom implementation following the IBM paper
        kernel = ProjectedQuantumKernel(feature_map=feature_map, gamma=gamma)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}. Use 'fidelity' or 'projected'.")

    return kernel


def build_hardware_kernel(feature_map, sampler, pass_manager=None,
                          max_circuits_per_job=300):
    """Build a FidelityQuantumKernel for IBM Quantum hardware execution.

    Uses ComputeUncompute fidelity with an IBM Runtime SamplerV2.

    Args:
        feature_map: Qiskit feature map circuit.
        sampler: qiskit_ibm_runtime.SamplerV2 instance (or any BaseSamplerV2).
        pass_manager: Optional transpiler PassManager for the target backend.
        max_circuits_per_job: Max circuits per hardware batch (avoids timeout).

    Returns:
        FidelityQuantumKernel configured for hardware execution.
    """
    from qiskit_machine_learning.state_fidelities import ComputeUncompute

    fidelity = ComputeUncompute(sampler=sampler, pass_manager=pass_manager)
    kernel = FidelityQuantumKernel(
        feature_map=feature_map,
        fidelity=fidelity,
        max_circuits_per_job=max_circuits_per_job,
    )
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
