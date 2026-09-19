"""
OWNER: Member 3 (Quantum & Metrics Engineer).
Layer 5 — CVaR-QAOA Quantum Core optimization over triaged bottleneck nodes.
"""
import functools
import itertools
from typing import Dict, List, Tuple
import numpy as np
import pennylane as qml
from scipy.optimize import minimize

from config import NODE_IDS, PHASE_ADAPTIVE_SHORT, PHASE_MAX_GREEN
from contracts import Phases, TriageResult
from classical_solver import compute_true_cost


def _build_ising_hamiltonian(critical_nodes: List[str],
                             qubo_weights: Dict[str, float],
                             couplings: Dict[Tuple[str, str], float]):
    """Expands QUBO formulation to Ising Hamiltonian H_C = sum h_i Z_i + sum J'_ij Z_i Z_j + const."""
    n = len(critical_nodes)
    if n == 0:
        return np.array([]), {}, 0.0, np.array([]), {}

    w = np.array([qubo_weights[node] for node in critical_nodes])
    K = max(1, int(np.ceil(0.5 * n)))
    lam = 0.5 * float(np.mean(w)) if n > 0 else 0.5

    # QUBO linear and quadratic terms: C(x) = sum a_i x_i + sum Q_ij x_i x_j + const_qubo
    a = -w + lam * (1.0 - 2.0 * K)
    Q: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            pair_key = (critical_nodes[i], critical_nodes[j])
            alt_pair_key = (critical_nodes[j], critical_nodes[i])
            edge_j = couplings.get(pair_key, couplings.get(alt_pair_key, 0.0))
            Q[(i, j)] = 2.0 * lam + edge_j

    # Ising mapping: x_i = (1 - Z_i) / 2
    h = np.array([-a[i] / 2.0 - sum(Q[tuple(sorted((i, j)))] for j in range(n) if j != i) / 4.0 for i in range(n)])
    J_prime = {pair: q_val / 4.0 for pair, q_val in Q.items()}
    const = lam * (K ** 2) + sum(a) / 2.0 + sum(Q.values()) / 4.0

    return h, J_prime, const, a, Q


@functools.lru_cache(maxsize=128)
def _solve_cvar_qaoa_cached(critical_nodes_tuple: Tuple[str, ...],
                             qubo_weights_tuple: Tuple[Tuple[str, float], ...],
                             couplings_tuple: Tuple[Tuple[Tuple[str, str], float], ...]) -> Tuple[Tuple[int, ...], dict]:
    """Cached solver for CVaR-QAOA optimization."""
    critical_nodes = list(critical_nodes_tuple)
    qubo_weights = dict(qubo_weights_tuple)
    couplings = dict(couplings_tuple)
    n = len(critical_nodes)

    if n == 0:
        return (), {"r": 1.0, "prob_opt": 1.0, "elite_cost": 0.0, "opt_cost": 0.0, "history": []}

    h, J_prime, const, a, Q = _build_ising_hamiltonian(critical_nodes, qubo_weights, couplings)
    states = list(itertools.product([0, 1], repeat=n))

    def ising_energy(x):
        z = [1 - 2 * xi for xi in x]
        return sum(h[i] * z[i] for i in range(n)) + sum(J_prime[k] * z[k[0]] * z[k[1]] for k in J_prime) + const

    # Assert QUBO == Ising == True Cost on all 2^n states
    E = np.array([ising_energy(x) for x in states])
    for idx, x in enumerate(states):
        true_c = compute_true_cost(x, critical_nodes, qubo_weights, couplings)
        assert abs(E[idx] - true_c) < 1e-7, f"QUBO/Ising cost assertion failed for state {x}"

    dev = qml.device("default.qubit", wires=n)
    p = 2  # 2 QAOA layers

    @qml.qnode(dev)
    def circ(theta):
        g, b = theta[:p], theta[p:]
        for i in range(n):
            qml.Hadamard(wires=i)
        for l in range(p):
            for i in range(n):
                qml.RZ(2.0 * g[l] * h[i], wires=i)
            for (i, j), jv in J_prime.items():
                qml.IsingZZ(2.0 * g[l] * jv, wires=[i, j])
            for i in range(n):
                qml.RX(2.0 * b[l], wires=i)
        return qml.probs(wires=range(n))

    history = []

    def cvar_objective(theta, alpha=0.2, shots=1024, s_seed=0):
        probs_val = circ(theta)
        r_gen = np.random.default_rng(s_seed)
        idx_samples = r_gen.choice(len(probs_val), size=shots, p=probs_val / probs_val.sum())
        sampled_energies = np.sort(E[idx_samples])
        k_count = max(1, int(np.ceil(alpha * shots)))
        val = float(sampled_energies[:k_count].mean())
        history.append(val)
        return val

    best_res = None
    for s in range(3):
        r_init = np.random.default_rng(42 + s)
        x0 = np.concatenate([r_init.uniform(0, 1, p), r_init.uniform(0, 1, p)])
        res = minimize(lambda t: cvar_objective(t, s_seed=s), x0, method="COBYLA", options={"maxiter": 80})
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    opt_probs = circ(best_res.x)
    r_samp = np.random.default_rng(123)
    samples_2048 = r_samp.choice(len(opt_probs), size=2048, p=opt_probs / opt_probs.sum())
    elite_idx = samples_2048[np.argmin(E[samples_2048])]
    elite_bitstring = states[elite_idx]

    opt_idx = int(np.argmin(E))
    e_max, e_min = float(E.max()), float(E.min())
    approx_ratio = (e_max - E[elite_idx]) / (e_max - e_min) if e_max > e_min else 1.0

    diagnostics = {
        "r": float(approx_ratio),
        "prob_opt": float(opt_probs[opt_idx]),
        "elite_cost": float(E[elite_idx]),
        "opt_cost": float(E[opt_idx]),
        "history": history,
    }

    return elite_bitstring, diagnostics


def optimize(triage: TriageResult) -> Phases:
    """Return {critical_node: PHASE_CODE}. Only critical nodes appear in returned dictionary."""
    phases, _ = optimize_with_diagnostics(triage)
    return phases


def optimize_with_diagnostics(triage: TriageResult) -> Tuple[Phases, dict]:
    """Approved PR-5: Returns (phases, diagnostics_dict) exposing QAOA details."""
    critical_nodes = triage.critical_nodes
    if not critical_nodes:
        return {}, {"r": 1.0, "prob_opt": 1.0, "elite_cost": 0.0, "opt_cost": 0.0, "history": []}

    c_tuple = tuple(critical_nodes)
    w_tuple = tuple(sorted(triage.qubo_weights.items()))
    j_tuple = tuple(sorted(triage.couplings.items()))

    elite_bitstring, diagnostics = _solve_cvar_qaoa_cached(c_tuple, w_tuple, j_tuple)

    phases: Phases = {}
    for i, node in enumerate(critical_nodes):
        phases[node] = PHASE_MAX_GREEN if elite_bitstring[i] == 1 else PHASE_ADAPTIVE_SHORT

    return phases, diagnostics
