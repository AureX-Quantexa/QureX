"""
OWNER: Member 3 (Quantum & Metrics Engineer).
Layer 5 Classical Reference Oracle — Exact brute-force solver over all 2^k states.
"""
import itertools
from typing import Dict, List, Tuple
import numpy as np

from config import NODE_IDS, PHASE_ADAPTIVE_SHORT, PHASE_MAX_GREEN
from contracts import Phases, TriageResult


def compute_true_cost(x: Tuple[int, ...], critical_nodes: List[str],
                       qubo_weights: Dict[str, float],
                       couplings: Dict[Tuple[str, str], float]) -> float:
    """Compute true objective cost C(x) for bitstring x.
    x_i in {0, 1}: 1 = MAX_GREEN, 0 = ADAPTIVE_SHORT.
    """
    n = len(critical_nodes)
    if n == 0:
        return 0.0

    w = np.array([qubo_weights[node] for node in critical_nodes])
    K = max(1, int(np.ceil(0.5 * n)))
    lam = 0.5 * float(np.mean(w)) if n > 0 else 0.5

    # Reward serving urgent nodes
    reward = sum(w[i] * x[i] for i in range(n))

    # Spillback penalty between adjacent critical nodes
    spillback = 0.0
    for (a, b), j_val in couplings.items():
        if a in critical_nodes and b in critical_nodes:
            idx_a = critical_nodes.index(a)
            idx_b = critical_nodes.index(b)
            spillback += j_val * x[idx_a] * x[idx_b]

    # Budget penalty: at most ~K nodes on max green
    budget_penalty = lam * (sum(x) - K) ** 2

    return -reward + spillback + budget_penalty


def solve_bruteforce(triage: TriageResult) -> Phases:
    """Exact brute-force solver evaluating all 2^n states (n <= 6) as correctness oracle.
    Returns {critical_node: PHASE_CODE}. Only critical nodes appear.
    """
    critical_nodes = triage.critical_nodes
    n = len(critical_nodes)
    if n == 0:
        return {}

    states = list(itertools.product([0, 1], repeat=n))
    best_cost = float("inf")
    best_state = states[0]

    for x in states:
        cost = compute_true_cost(x, critical_nodes, triage.qubo_weights, triage.couplings)
        if cost < best_cost:
            best_cost = cost
            best_state = x

    phases: Phases = {}
    for i, node in enumerate(critical_nodes):
        phases[node] = PHASE_MAX_GREEN if best_state[i] == 1 else PHASE_ADAPTIVE_SHORT

    return phases
