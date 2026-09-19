"""OWNER: Member 2.  Layer 4 - city graph + triage.  (STUB v0 - build_city_graph is final-shape)"""
import networkx as nx
import numpy as np

from config import (EDGE_TRAVEL_COST, EVENT_NORMAL, GRID_COLS, GRID_ROWS, MAX_QUBITS,
                    NODE_IDS, TRIAGE_UTILIZATION_THRESHOLD)
from contracts import GridState, TriageResult


def build_city_graph() -> nx.Graph:
    """2x3 grid, row-major labels: Intersection_1..6. Edge attr: 'travel_cost' (seconds)."""
    G = nx.Graph()
    G.add_nodes_from(NODE_IDS)                       # explicit order, not G.nodes() luck
    name = lambda r, c: NODE_IDS[r * GRID_COLS + c]
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if c + 1 < GRID_COLS:
                G.add_edge(name(r, c), name(r, c + 1), travel_cost=EDGE_TRAVEL_COST)
            if r + 1 < GRID_ROWS:
                G.add_edge(name(r, c), name(r + 1, c), travel_cost=EDGE_TRAVEL_COST)
    return G


def triage(grid_state: GridState, forecast: np.ndarray, G: nx.Graph) -> TriageResult:
    critical = [n for i, n in enumerate(NODE_IDS)
                if forecast[i] / grid_state[n]["capacity"] > TRIAGE_UTILIZATION_THRESHOLD
                or grid_state[n]["event"] != EVENT_NORMAL]
    critical = critical[:MAX_QUBITS]
    weights = {n: grid_state[n]["queue"] / grid_state[n]["capacity"] for n in critical}
    couplings = {}
    for a, b in G.edges():
        if a in critical and b in critical:
            key = tuple(sorted((a, b), key=NODE_IDS.index))
            couplings[key] = 0.5 * (weights[a] + weights[b]) / 2
    return TriageResult(critical, weights, couplings)
