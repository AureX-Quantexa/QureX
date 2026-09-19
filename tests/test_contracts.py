"""Contract tests. RUN BEFORE EVERY PUSH:  pytest -q
Each member may ADD tests for their own layer below (in your own section) but never weaken the ones here."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from config import (EVENT_CODES, EVENT_EMERGENCY, EVENT_FESTIVAL, EVENT_NORMAL, NODE_IDS,
                    PHASE_ADAPTIVE_SHORT, PHASE_MAX_GREEN)
from contracts import (flatten_grid_state, validate_forecast, validate_grid_state,
                       validate_kpis, validate_phases, validate_stats, validate_triage)
from graph_layer import build_city_graph
from pipeline import run_pipeline
from scenario import apply_event, default_grid_state

G = build_city_graph()
SCENARIOS = [(n, e) for n in (NODE_IDS[0], NODE_IDS[2], NODE_IDS[5]) for e in EVENT_CODES]


def test_default_state_valid():
    validate_grid_state(default_grid_state())


def test_flatten_shape_and_order():
    gs = default_grid_state()
    v = flatten_grid_state(gs)
    assert v.shape == (18,)
    assert v[0] == gs[NODE_IDS[0]]["queue"] and v[3] == gs[NODE_IDS[1]]["queue"]


def test_graph_shape():
    assert list(G.nodes()) == NODE_IDS
    assert G.number_of_edges() == 7
    assert all("travel_cost" in d for _, _, d in G.edges(data=True))


def test_apply_event_does_not_mutate():
    gs = default_grid_state()
    apply_event(gs, NODE_IDS[1], EVENT_EMERGENCY)
    assert gs[NODE_IDS[1]]["event"] == EVENT_NORMAL


@pytest.mark.parametrize("node,event", SCENARIOS)
def test_full_pipeline_contracts(node, event):
    gs = apply_event(default_grid_state(), node, event)
    validate_grid_state(gs)
    out = run_pipeline(gs, G)
    validate_stats(out.stats)
    validate_forecast(out.forecast)
    validate_triage(out.triage)
    validate_phases(out.phases)
    validate_kpis(out.kpis)
    assert all(n in NODE_IDS for n in out.corridor)
    assert isinstance(out.narrative, str) and out.narrative


def test_festival_marks_all_nodes():
    gs = apply_event(default_grid_state(), NODE_IDS[0], EVENT_FESTIVAL)
    assert all(s["event"] == EVENT_FESTIVAL for s in gs.values())


# ------------------------------------------------------------ add your own tests below
# --- M1 (stats/forecast):
# --- M2 (graph/routing/scenario):
# --- M3 (quantum/metrics):
def test_m3_qubo_ising_assertion():
    from quantum_layer import _build_ising_hamiltonian
    from classical_solver import compute_true_cost, solve_bruteforce
    from contracts import TriageResult
    nodes = NODE_IDS[:3]
    w = {n: 0.8 for n in nodes}
    c = {(nodes[0], nodes[1]): 0.4}
    t = TriageResult(nodes, w, c)
    sol = solve_bruteforce(t)
    assert len(sol) == 3
    assert all(ph in (PHASE_MAX_GREEN, PHASE_ADAPTIVE_SHORT) for ph in sol.values())


def test_m3_qaoa_vs_bruteforce_accuracy():
    from quantum_layer import optimize_with_diagnostics
    from classical_solver import solve_bruteforce
    from contracts import TriageResult
    nodes = NODE_IDS[:3]
    w = {nodes[0]: 0.9, nodes[1]: 0.4, nodes[2]: 0.7}
    c = {(nodes[0], nodes[1]): 0.5, (nodes[1], nodes[2]): 0.3}
    t = TriageResult(nodes, w, c)
    phases, diag = optimize_with_diagnostics(t)
    bf = solve_bruteforce(t)
    assert diag["r"] >= 0.95
    assert set(phases.keys()) == set(nodes)


def test_m3_kpis_change_with_phases():
    from metrics import compute_kpis
    from contracts import TriageResult
    gs = default_grid_state()
    t = TriageResult([], {}, {})
    p1 = {n: PHASE_MAX_GREEN for n in NODE_IDS}
    p2 = {n: PHASE_ADAPTIVE_SHORT for n in NODE_IDS}
    k1 = compute_kpis(gs, p1, t, [])
    k2 = compute_kpis(gs, p2, t, [])
    assert k1.total_quantum != k2.total_quantum
    assert k1.fuel_saved_gal >= 0 and k1.co2_saved_kg >= 0


def test_m3_zero_qubits():
    from quantum_layer import optimize
    from classical_solver import solve_bruteforce
    from contracts import TriageResult
    t = TriageResult([], {}, {})
    assert optimize(t) == {}
    assert solve_bruteforce(t) == {}


# --- M4 (copilot/app):

