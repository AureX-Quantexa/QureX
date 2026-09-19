"""OWNER: Member 2.  Layer 4 - city graph + triage.  (Graph Filtration Triage)"""
import networkx as nx
import numpy as np

from config import (EDGE_TRAVEL_COST, EVENT_NORMAL, GRID_COLS, GRID_ROWS, MAX_QUBITS,
                    NODE_IDS, TRIAGE_UTILIZATION_THRESHOLD)
from contracts import GridState, TriageResult


def build_city_graph() -> nx.Graph:
    """Constructs 2x3 grid NetworkX graph, row-major labels: Intersection_1..6.
    Edge attribute: 'travel_cost' (seconds)."""
    G = nx.Graph()
    G.add_nodes_from(NODE_IDS)  # Explicit order matching NODE_IDS
    name = lambda r, c: NODE_IDS[r * GRID_COLS + c]
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if c + 1 < GRID_COLS:
                G.add_edge(name(r, c), name(r, c + 1), travel_cost=EDGE_TRAVEL_COST)
            if r + 1 < GRID_ROWS:
                G.add_edge(name(r, c), name(r + 1, c), travel_cost=EDGE_TRAVEL_COST)
    return G


def triage(grid_state: GridState, forecast: np.ndarray, G: nx.Graph) -> TriageResult:
    """Evaluates network demand, identifies critical nodes, and computes QUBO weights & couplings."""
    # 1. Identify critical nodes based on predicted utilization or non-normal events
    critical_candidates = []
    for i, node in enumerate(NODE_IDS):
        capacity = float(grid_state[node]["capacity"])
        inflow = float(forecast[i]) if i < len(forecast) else float(grid_state[node]["queue"])
        utilization = inflow / capacity if capacity > 0 else 0.0

        if utilization > TRIAGE_UTILIZATION_THRESHOLD or grid_state[node]["event"] != EVENT_NORMAL:
            critical_candidates.append(node)

    # 2. Cap critical nodes to MAX_QUBITS (top nodes by raw urgency if over capacity)
    if len(critical_candidates) > MAX_QUBITS:
        ranked = sorted(
            critical_candidates,
            key=lambda n: float(grid_state[n]["queue"]) / max(1.0, float(grid_state[n]["capacity"])),
            reverse=True
        )
        selected = ranked[:MAX_QUBITS]
        critical_nodes = sorted(selected, key=NODE_IDS.index)
    else:
        critical_nodes = sorted(critical_candidates, key=NODE_IDS.index)

    # 3. Compute normalized urgency weights: w_i = q_i / C_i, normalized so max w_i = 1.0
    raw_weights = {}
    for node in critical_nodes:
        cap = float(grid_state[node]["capacity"])
        raw_weights[node] = float(grid_state[node]["queue"]) / cap if cap > 0 else 0.0

    max_raw = max(raw_weights.values()) if raw_weights else 0.0
    if max_raw > 0.0:
        qubo_weights = {n: raw_weights[n] / max_raw for n in critical_nodes}
    else:
        qubo_weights = {n: 0.0 for n in critical_nodes}

    # 4. Compute coupling weights between adjacent critical nodes:
    #    J_ij = kappa * ((w_i + w_j) / 2) * (tau_ref / tau_ij), with kappa = 0.5
    couplings = {}
    kappa = 0.5
    tau_ref = EDGE_TRAVEL_COST

    for u, v in G.edges():
        if u in critical_nodes and v in critical_nodes:
            idx_u = NODE_IDS.index(u)
            idx_v = NODE_IDS.index(v)
            a, b = (u, v) if idx_u < idx_v else (v, u)

            tau_ij = float(G[u][v].get("travel_cost", EDGE_TRAVEL_COST))
            tau_ij = max(1e-6, tau_ij)

            j_val = kappa * ((qubo_weights[a] + qubo_weights[b]) / 2.0) * (tau_ref / tau_ij)
            couplings[(a, b)] = float(j_val)

    return TriageResult(critical_nodes, qubo_weights, couplings)

