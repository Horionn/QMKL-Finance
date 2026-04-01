"""QUBO solvers: brute-force, simulated annealing, greedy, and QAOA.

All solvers share the same interface:
    result = solver(Q_mat, d, M, Q, **kwargs)
    result = {
        'x'       : (d*M,) binary vector (best solution found),
        'energy'  : float (QUBO energy of best solution),
        'history' : list of energies during optimization (optional),
        'solver'  : str name,
    }
"""

import numpy as np
from .assignment_qubo import energy, decode_assignment, check_constraints


# ── Brute-force (only feasible for n_vars ≤ ~24) ─────────────────────────────

def solve_brute_force(Q_mat, d, M, Q, **kwargs):
    """Exhaustive search over all 2^(d*M) binary solutions.

    CAUTION: exponential — only use for d*M ≤ 24 (demo/validation).

    Returns best valid solution (satisfying Q features per kernel),
    or globally best if no valid solution found.
    """
    n_vars = d * M
    if n_vars > 24:
        raise ValueError(
            f"Brute-force infeasible for {n_vars} variables (2^{n_vars} = {2**n_vars:,}). "
            f"Use solve_simulated_annealing instead."
        )

    best_energy = np.inf
    best_x = None

    for i in range(2 ** n_vars):
        x = np.array([(i >> b) & 1 for b in range(n_vars)], dtype=float)
        e = energy(x, Q_mat)
        if e < best_energy:
            assignment = decode_assignment(x, d, M, Q, repair=False)
            valid, _ = check_constraints(assignment, d, M, Q)
            if valid:
                best_energy = e
                best_x = x.copy()

    # If no valid solution, return unconstrained best
    if best_x is None:
        best_energy = np.inf
        for i in range(2 ** n_vars):
            x = np.array([(i >> b) & 1 for b in range(n_vars)], dtype=float)
            e = energy(x, Q_mat)
            if e < best_energy:
                best_energy = e
                best_x = x.copy()

    return {"x": best_x, "energy": best_energy, "solver": "brute_force"}


# ── Simulated Annealing ───────────────────────────────────────────────────────

def solve_simulated_annealing(Q_mat, d, M, Q,
                               n_iter=50_000, T_init=5.0, T_min=1e-4,
                               seed=42, **kwargs):
    """Simulated annealing for the QUBO assignment problem.

    Uses single-bit flips as moves. Starts from a feasible solution
    (non-overlapping sequential assignment).

    Args:
        Q_mat  : (d*M, d*M) QUBO matrix
        d, M   : dataset dims and number of kernels
        Q      : features per kernel
        n_iter : number of SA iterations
        T_init : initial temperature
        T_min  : minimum temperature (geometric cooling)

    Returns:
        result dict with 'x', 'energy', 'history', 'solver'
    """
    rng = np.random.RandomState(seed)
    n_vars = d * M

    # Initialize from feasible solution (non-overlapping)
    x = np.zeros(n_vars)
    for m in range(M):
        for q_idx in range(Q):
            k = (m * Q + q_idx) % d
            x[k * M + m] = 1.0

    current_energy = energy(x, Q_mat)
    best_x = x.copy()
    best_energy = current_energy

    history = [current_energy]
    T = T_init
    cooling = (T_min / T_init) ** (1.0 / n_iter)

    for it in range(n_iter):
        # Random single-bit flip
        idx = rng.randint(n_vars)
        x_new = x.copy()
        x_new[idx] = 1.0 - x_new[idx]

        new_energy = energy(x_new, Q_mat)
        delta = new_energy - current_energy

        if delta < 0 or rng.rand() < np.exp(-delta / T):
            x = x_new
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_x = x.copy()

        T = max(T * cooling, T_min)

        if it % 5000 == 0:
            history.append(best_energy)

    return {
        "x": best_x,
        "energy": best_energy,
        "history": history,
        "solver": "simulated_annealing",
    }


# ── Greedy constructive ───────────────────────────────────────────────────────

def solve_greedy(Q_mat, d, M, Q, **kwargs):
    """Greedy constructive: add features one by one by marginal gain.

    For each kernel m, selects the Q features with highest marginal
    reduction in QUBO energy.

    Returns:
        result dict
    """
    n_vars = d * M
    x = np.zeros(n_vars)

    for m in range(M):
        assigned = []
        available = list(range(d))

        for _ in range(Q):
            best_gain = np.inf
            best_k = None
            for k in available:
                idx = k * M + m
                x_trial = x.copy()
                x_trial[idx] = 1.0
                e = energy(x_trial, Q_mat)
                if e < best_gain:
                    best_gain = e
                    best_k = k
            if best_k is not None:
                x[best_k * M + m] = 1.0
                assigned.append(best_k)
                available.remove(best_k)

    return {"x": x, "energy": energy(x, Q_mat), "solver": "greedy"}


# ── QAOA solver (simulation via Qiskit) ───────────────────────────────────────

def solve_qaoa(Q_mat, d, M, Q, p=1, optimizer="COBYLA",
               max_iter=300, seed=42, backend=None, shots=4096, **kwargs):
    """QAOA solver for the QUBO assignment problem.

    Converts the QUBO matrix to a SparsePauliOp Hamiltonian (Ising encoding),
    then runs QAOA using Qiskit primitives.

    For simulation: uses StatevectorEstimator (exact).
    For hardware: uses EstimatorV2 with IBM Runtime Session.

    Args:
        Q_mat    : (d*M, d*M) upper-triangular QUBO matrix
        d, M, Q  : problem dimensions
        p        : number of QAOA layers (depth)
        optimizer: classical optimizer name ('COBYLA', 'SLSQP', 'L_BFGS_B')
        max_iter : maximum classical optimizer iterations
        seed     : random seed for initialization
        backend  : Qiskit backend (None = StatevectorEstimator)
        shots    : shots per evaluation (hardware only)

    Returns:
        result dict with 'x', 'energy', 'history', 'counts', 'solver'
    """
    try:
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.circuit import QuantumCircuit, Parameter
        from qiskit.primitives import StatevectorEstimator, StatevectorSampler
        from scipy.optimize import minimize
    except ImportError as e:
        raise ImportError(
            "Qiskit not available. Install with: pip install qiskit qiskit-aer"
        ) from e

    n_vars = d * M

    # ── Step 1: QUBO → Ising Hamiltonian ──────────────────────────────────────
    # Substitution: x_i = (1 - z_i) / 2,  z_i ∈ {-1, +1}
    # x_i x_j = (1 - z_i)(1 - z_j) / 4
    # H_ising = const + Σ h_i Z_i + Σ_{i<j} J_ij Z_i Z_j
    hamiltonian, offset = _qubo_to_ising(Q_mat, n_vars)

    # ── Step 2: Build QAOA circuit ─────────────────────────────────────────────
    qc, betas, gammas = _build_qaoa_circuit(n_vars, p, hamiltonian)

    # ── Step 3: Optimize parameters ───────────────────────────────────────────
    rng = np.random.RandomState(seed)
    theta0 = rng.uniform(0, np.pi, size=2 * p)

    history = []

    if backend is None:
        estimator = StatevectorEstimator()
        sampler = StatevectorSampler()

        def cost_fn(theta):
            params = dict(zip(betas + gammas, theta[:p].tolist() + theta[p:].tolist()))
            bound_qc = qc.assign_parameters(params)
            result = estimator.run([(bound_qc, hamiltonian)]).result()
            ev = float(result[0].data.evs) + offset
            history.append(ev)
            return ev

    else:
        # Job mode — compatible Open Plan (Sessions require paid plan)
        from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimatorV2
        from qiskit_ibm_runtime import SamplerV2 as RuntimeSamplerV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        estimator_hw = RuntimeEstimatorV2(backend)

        # Pre-transpile once; apply layout to Hamiltonian so qubit counts match
        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
        isa_qc_template = pm.run(qc)
        # Explicitly pass num_qubits so the observable is padded to the full backend size
        isa_hamiltonian = hamiltonian.apply_layout(
            isa_qc_template.layout,
            num_qubits=isa_qc_template.num_qubits,
        )

        # Map original parameters to transpiled circuit parameters by name
        _param_map = {p_obj.name: p_obj for p_obj in isa_qc_template.parameters}

        import os, json as _json
        _job_log_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "results", "qubo_solutions", "hw_jobs.json"
        )
        os.makedirs(os.path.dirname(_job_log_path), exist_ok=True)
        _jobs_log = []

        def cost_fn(theta):
            name_vals = list(zip(
                [b.name for b in betas] + [g.name for g in gammas],
                theta[:p].tolist() + theta[p:].tolist()
            ))
            param_bindings = {_param_map[name]: val for name, val in name_vals}
            bound_isa = isa_qc_template.assign_parameters(param_bindings)
            pub = (bound_isa, isa_hamiltonian)
            job = estimator_hw.run([pub])
            # Save job ID immediately so it can be retrieved later
            _jobs_log.append({"job_id": job.job_id(), "theta": list(theta), "iter": len(_jobs_log)})
            with open(_job_log_path, "w") as _f:
                _json.dump(_jobs_log, _f, indent=2)
            result = job.result()
            ev = float(result[0].data.evs) + offset
            history.append(ev)
            return ev

    # Multiple restarts to mitigate optimizer local minima
    # On hardware: 1 restart only (each restart = max_iter IBM jobs)
    n_restarts = 1 if backend is not None else (3 if p <= 2 else 1)
    best_opt_energy = np.inf
    best_opt_result = None
    best_theta = None

    for restart in range(n_restarts):
        seed_r = seed + restart * 13
        rng_r = np.random.RandomState(seed_r)
        theta_init = rng_r.uniform(0, np.pi / 2, size=2 * p)
        res = minimize(cost_fn, theta_init, method=optimizer,
                       options={"maxiter": max_iter})
        if res.fun < best_opt_energy:
            best_opt_energy = res.fun
            best_opt_result = res
            best_theta = res.x

    opt_result = best_opt_result
    optimal_theta = best_theta

    # ── Step 4: Sample optimal circuit → binary solution ──────────────────────
    params_opt = dict(zip(betas + gammas,
                          optimal_theta[:p].tolist() + optimal_theta[p:].tolist()))
    bound_qc = qc.assign_parameters(params_opt)
    bound_qc.measure_all()

    if backend is None:
        counts = StatevectorSampler().run([bound_qc], shots=shots).result()[0].data.meas.get_counts()
    else:
        isa_qc_meas = pm.run(bound_qc)
        hw_sampler = RuntimeSamplerV2(backend)
        counts = hw_sampler.run([isa_qc_meas], shots=shots).result()[0].data.meas.get_counts()

    # Best-energy bitstring from all sampled (not just most frequent)
    best_x = None
    best_e = np.inf
    for bitstring in counts:
        x_candidate = np.array([int(b) for b in reversed(bitstring)], dtype=float)
        e_candidate = energy(x_candidate, Q_mat)
        if e_candidate < best_e:
            best_e = e_candidate
            best_x = x_candidate
    x_best = best_x

    return {
        "x": x_best,
        "energy": energy(x_best, Q_mat),
        "history": history,
        "counts": counts,
        "opt_result": opt_result,
        "solver": f"qaoa_p{p}",
    }


# ── Internal helpers ───────────────────────────────────────────────────────────

def _qubo_to_ising(Q_mat, n):
    """Convert upper-triangular QUBO Q to Ising (h, J) + offset.

    x_i = (1 - z_i) / 2,  x_i x_j = (1 - z_i - z_j + z_i z_j) / 4

    Returns:
        hamiltonian : SparsePauliOp
        offset      : float constant
    """
    from qiskit.quantum_info import SparsePauliOp

    h = np.zeros(n)     # linear Ising coefficients
    J = {}              # quadratic: (i,j) -> coeff
    offset = 0.0

    for i in range(n):
        Qii = Q_mat[i, i]
        # x_i = (1-z_i)/2 → Qii * x_i = Qii/2 - Qii/2 * z_i
        offset += Qii / 2
        h[i] -= Qii / 2

        for j in range(i + 1, n):
            Qij = Q_mat[i, j]
            if abs(Qij) < 1e-12:
                continue
            # Qij * x_i * x_j = Qij/4 * (1 - z_i - z_j + z_i z_j)
            offset += Qij / 4
            h[i] -= Qij / 4
            h[j] -= Qij / 4
            J[(i, j)] = Qij / 4

    # Build SparsePauliOp
    paulis, coeffs = [], []

    for i in range(n):
        if abs(h[i]) > 1e-12:
            pauli_str = "I" * (n - 1 - i) + "Z" + "I" * i
            paulis.append(pauli_str)
            coeffs.append(h[i])

    for (i, j), coeff in J.items():
        if abs(coeff) > 1e-12:
            pauli_arr = ["I"] * n
            pauli_arr[n - 1 - i] = "Z"
            pauli_arr[n - 1 - j] = "Z"
            paulis.append("".join(pauli_arr))
            coeffs.append(coeff)

    if len(paulis) == 0:
        hamiltonian = SparsePauliOp(["I" * n], coeffs=[0.0])
    else:
        hamiltonian = SparsePauliOp(paulis, coeffs=coeffs).simplify()

    return hamiltonian, offset


def _build_qaoa_circuit(n_qubits, p, hamiltonian):
    """Build QAOA ansatz circuit for p layers.

    Structure:
      H^⊗n  →  [U_C(γ) · U_B(β)]^p

    where U_C(γ) applies the problem Hamiltonian (ZZ and Z terms),
    and U_B(β) applies the mixing Hamiltonian (Rx gates).

    Returns:
        qc     : ParameterizedQuantumCircuit (no measurement)
        betas  : list of p beta parameters
        gammas : list of p gamma parameters
    """
    from qiskit.circuit import QuantumCircuit, Parameter
    from qiskit.quantum_info import SparsePauliOp

    betas  = [Parameter(f"β_{i}") for i in range(p)]
    gammas = [Parameter(f"γ_{i}") for i in range(p)]

    qc = QuantumCircuit(n_qubits)

    # Initial superposition
    qc.h(range(n_qubits))

    for layer in range(p):
        gamma, beta = gammas[layer], betas[layer]

        # Problem unitary U_C(γ): exp(-i γ H_C)
        # Apply ZZ terms as CX + Rz + CX, Z terms as Rz
        for pauli_term, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
            pauli_str = pauli_term.to_label()
            z_qubits = [n_qubits - 1 - k for k, p_char in enumerate(pauli_str)
                        if p_char == "Z"]
            if len(z_qubits) == 1:
                qc.rz(2 * gamma * float(coeff.real), z_qubits[0])
            elif len(z_qubits) == 2:
                i, j = z_qubits
                qc.cx(i, j)
                qc.rz(2 * gamma * float(coeff.real), j)
                qc.cx(i, j)

        # Mixer unitary U_B(β): exp(-i β Σ X_i)
        for qubit in range(n_qubits):
            qc.rx(2 * beta, qubit)

    return qc, betas, gammas
