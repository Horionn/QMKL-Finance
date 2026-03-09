"""Quantum feature map definitions for QMKL.

Feature maps encode classical data into quantum states.
Multiple feature maps with different structures and bandwidths
form the basis of the multiple kernel learning approach.
"""

from collections import OrderedDict

from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, PauliFeatureMap


def build_feature_map(name, n_qubits, alpha=1.0, reps=1, entanglement="linear"):
    """Build a quantum feature map circuit.

    Args:
        name: Feature map type. One of:
            - "Z": Single-qubit Z rotations only (no entanglement)
            - "ZZ": Z rotations + ZZ entangling (Havlicek et al.)
            - "pauli": General Pauli feature map with configurable paulis
        n_qubits: Number of qubits (= number of features).
        alpha: Bandwidth parameter controlling rotation scaling.
        reps: Number of repetitions of the feature map circuit.
        entanglement: Entanglement pattern ("linear", "full", "circular").

    Returns:
        A Qiskit QuantumCircuit acting as a feature map.
    """
    if name == "Z":
        fm = ZFeatureMap(
            feature_dimension=n_qubits,
            reps=reps,
            data_map_func=_make_data_map_func(alpha),
        )
    elif name == "ZZ":
        fm = ZZFeatureMap(
            feature_dimension=n_qubits,
            reps=reps,
            entanglement=entanglement,
            data_map_func=_make_data_map_func(alpha),
        )
    elif name == "pauli":
        fm = PauliFeatureMap(
            feature_dimension=n_qubits,
            reps=reps,
            entanglement=entanglement,
            paulis=["Z", "ZZ"],
            data_map_func=_make_data_map_func(alpha),
        )
    elif name == "pauli_XZ":
        fm = PauliFeatureMap(
            feature_dimension=n_qubits,
            reps=reps,
            entanglement=entanglement,
            paulis=["X", "ZZ"],
            data_map_func=_make_data_map_func(alpha),
        )
    elif name == "pauli_YXX":
        fm = PauliFeatureMap(
            feature_dimension=n_qubits,
            reps=reps,
            entanglement=entanglement,
            paulis=["Y", "XX"],
            data_map_func=_make_data_map_func(alpha),
        )
    elif name == "pauli_YZX":
        fm = PauliFeatureMap(
            feature_dimension=n_qubits,
            reps=reps,
            entanglement=entanglement,
            paulis=["Y", "ZX"],
            data_map_func=_make_data_map_func(alpha),
        )
    else:
        raise ValueError(
            f"Unknown feature map: {name}. "
            f"Available: Z, ZZ, pauli, pauli_XZ, pauli_YXX, pauli_YZX"
        )

    return fm


def _make_data_map_func(alpha):
    """Create a data mapping function with bandwidth scaling.

    For single features: x -> alpha * x
    For feature pairs: (x1, x2) -> alpha * (pi - x1) * (pi - x2)
    """
    import numpy as np

    def data_map_func(x):
        coeff = x[0] if len(x) == 1 else (np.pi - x[0]) * (np.pi - x[1])
        return alpha * coeff

    return data_map_func


def get_feature_map_library(n_qubits, entanglement="linear"):
    """Generate the full library of feature maps used in the IBM QMKL paper.

    Returns an OrderedDict {label: feature_map} covering diverse
    Pauli structures and bandwidths.

    Library design rationale:
    - Each circuit type (Z, ZZ, Pauli variants) appears with 2 alpha values
    - Small alpha (0.5-2.0): preserves geometric structure, avoids concentration
    - Large alpha (4.0-8.0): increases expressivity, risk of concentration at high qubits
    - Very large alpha (14, 20) removed: systematically concentrated → always 0 weight
    """
    configs = [
        # (label, type, alpha, reps)
        # ── Single-qubit Z rotations (no entanglement, fast) ────────
        ("Z_a1.0",  "Z",  1.0, 1),
        ("Z_a3.0",  "Z",  3.0, 1),
        # ── Two-body ZZ interactions (linear entanglement) ──────────
        ("ZZ_a1.0", "ZZ", 1.0, 1),
        ("ZZ_a4.0", "ZZ", 4.0, 1),
        # ── Pauli Z+ZZ (standard, 2 reps for depth) ────────────────
        ("pauli_a0.6",  "pauli",  0.6, 2),
        ("pauli_a3.0",  "pauli",  3.0, 2),
        # ── Pauli X+ZZ (cross-terms) ───────────────────────────────
        ("pauli_XZ_a0.5", "pauli_XZ", 0.5, 2),
        ("pauli_XZ_a2.5", "pauli_XZ", 2.5, 2),
        # ── Pauli Y+XX ─────────────────────────────────────────────
        ("pauli_YXX_a0.6", "pauli_YXX", 0.6, 2),
        ("pauli_YXX_a3.0", "pauli_YXX", 3.0, 2),
        # ── Pauli Y+ZX ─────────────────────────────────────────────
        ("pauli_YZX_a0.6", "pauli_YZX", 0.6, 2),
        ("pauli_YZX_a3.0", "pauli_YZX", 3.0, 2),
    ]

    library = OrderedDict()
    for label, name, alpha, reps in configs:
        fm = build_feature_map(
            name, n_qubits, alpha=alpha, reps=reps, entanglement=entanglement
        )
        library[label] = fm

    return library


def get_extended_feature_map_library(n_qubits, entanglement="linear"):
    """Extended library with 20 kernels for comprehensive studies.

    Adds:
    - 3 alpha values per type (small, medium, large) instead of 2
    - Full entanglement variants for ZZ
    - More Pauli combinations
    """
    configs = [
        # ── Z (no entanglement) ─────────────────────────────────────
        ("Z_a0.5",  "Z",  0.5, 1, "linear"),
        ("Z_a1.5",  "Z",  1.5, 1, "linear"),
        ("Z_a4.0",  "Z",  4.0, 1, "linear"),
        # ── ZZ linear entanglement ──────────────────────────────────
        ("ZZ_lin_a0.8",  "ZZ", 0.8, 1, "linear"),
        ("ZZ_lin_a2.0",  "ZZ", 2.0, 1, "linear"),
        ("ZZ_lin_a5.0",  "ZZ", 5.0, 1, "linear"),
        # ── ZZ full entanglement ────────────────────────────────────
        ("ZZ_full_a1.0", "ZZ", 1.0, 1, "full"),
        ("ZZ_full_a3.0", "ZZ", 3.0, 1, "full"),
        # ── Pauli Z+ZZ (2 reps) ────────────────────────────────────
        ("pauli_a0.4",  "pauli",  0.4, 2, "linear"),
        ("pauli_a1.5",  "pauli",  1.5, 2, "linear"),
        ("pauli_a4.0",  "pauli",  4.0, 2, "linear"),
        # ── Pauli X+ZZ ─────────────────────────────────────────────
        ("pauli_XZ_a0.5", "pauli_XZ", 0.5, 2, "linear"),
        ("pauli_XZ_a2.0", "pauli_XZ", 2.0, 2, "linear"),
        ("pauli_XZ_a5.0", "pauli_XZ", 5.0, 2, "linear"),
        # ── Pauli Y+XX ─────────────────────────────────────────────
        ("pauli_YXX_a0.5", "pauli_YXX", 0.5, 2, "linear"),
        ("pauli_YXX_a2.0", "pauli_YXX", 2.0, 2, "linear"),
        ("pauli_YXX_a5.0", "pauli_YXX", 5.0, 2, "linear"),
        # ── Pauli Y+ZX ─────────────────────────────────────────────
        ("pauli_YZX_a0.5", "pauli_YZX", 0.5, 2, "linear"),
        ("pauli_YZX_a2.0", "pauli_YZX", 2.0, 2, "linear"),
        ("pauli_YZX_a5.0", "pauli_YZX", 5.0, 2, "linear"),
    ]

    library = OrderedDict()
    for label, name, alpha, reps, ent in configs:
        fm = build_feature_map(
            name, n_qubits, alpha=alpha, reps=reps, entanglement=ent
        )
        library[label] = fm

    return library
