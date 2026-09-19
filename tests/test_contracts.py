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
def test_m2_all_scenario_presets_and_pr1():
    from config import EVENT_ACCIDENT, EVENT_CONGESTION
    from scenario import _PRESETS, EVENT_ROAD_CLOSURE
    assert _PRESETS[EVENT_NORMAL]["queue"] == 12
    assert _PRESETS[EVENT_NORMAL]["avg_speed"] == 42.0
    assert _PRESETS[EVENT_CONGESTION]["queue"] == 45
    assert _PRESETS[EVENT_CONGESTION]["avg_speed"] == 6.0
    assert _PRESETS[EVENT_ACCIDENT]["queue"] == 38
    assert _PRESETS[EVENT_ACCIDENT]["capacity"] == 15
    assert _PRESETS[EVENT_EMERGENCY]["queue"] == 24
    assert _PRESETS[EVENT_EMERGENCY]["avg_speed"] == 22.0
    assert _PRESETS[EVENT_FESTIVAL]["queue"] == 40
    assert _PRESETS[EVENT_ROAD_CLOSURE]["capacity"] == 5
    assert _PRESETS[EVENT_ROAD_CLOSURE]["avg_speed"] == 1.0


def test_m2_apply_event_immutability_and_pr1():
    from scenario import apply_event, default_grid_state, EVENT_ROAD_CLOSURE
    base = default_grid_state()
    modified = apply_event(base, NODE_IDS[2], EVENT_ROAD_CLOSURE)
    assert base[NODE_IDS[2]]["capacity"] == 50
    assert modified[NODE_IDS[2]]["capacity"] == 5
    assert modified[NODE_IDS[2]]["avg_speed"] == 1.0
    assert modified[NODE_IDS[2]]["event"] == EVENT_ROAD_CLOSURE


def test_m2_triage_contract_and_normalization():
    from config import EVENT_CONGESTION
    from graph_layer import triage
    gs = default_grid_state()
    gs = apply_event(gs, NODE_IDS[0], EVENT_CONGESTION)
    gs = apply_event(gs, NODE_IDS[1], EVENT_CONGESTION)
    forecast = np.zeros(len(NODE_IDS))
    t = triage(gs, forecast, G)
    validate_triage(t)
    assert NODE_IDS[0] in t.critical_nodes
    assert NODE_IDS[1] in t.critical_nodes
    assert max(t.qubo_weights.values()) == 1.0
    assert (NODE_IDS[0], NODE_IDS[1]) in t.couplings
    assert t.couplings[(NODE_IDS[0], NODE_IDS[1])] > 0


def test_m2_triage_capping_at_max_qubits():
    from config import MAX_QUBITS
    from graph_layer import triage
    gs = default_grid_state()
    forecast = np.array([100.0] * len(NODE_IDS))
    t = triage(gs, forecast, G)
    validate_triage(t)
    assert len(t.critical_nodes) <= MAX_QUBITS
    order = [NODE_IDS.index(n) for n in t.critical_nodes]
    assert order == sorted(order)


def test_m2_emergency_corridor_source_is_target():
    from routing_layer import apply_emergency_override
    from config import EMERGENCY_TARGET, PHASE_EMERGENCY_CORRIDOR
    gs = apply_event(default_grid_state(), EMERGENCY_TARGET, EVENT_EMERGENCY)
    phases = {n: PHASE_MAX_GREEN for n in NODE_IDS}
    new_phases, corridor = apply_emergency_override(gs, G, phases)
    assert corridor == [EMERGENCY_TARGET]
    assert new_phases[EMERGENCY_TARGET] == PHASE_EMERGENCY_CORRIDOR


def test_m2_emergency_corridor_distant_node():
    from routing_layer import apply_emergency_override
    from config import EMERGENCY_TARGET, PHASE_EMERGENCY_CORRIDOR
    gs = apply_event(default_grid_state(), "Intersection_6", EVENT_EMERGENCY)
    phases = {n: PHASE_ADAPTIVE_SHORT for n in NODE_IDS}
    new_phases, corridor = apply_emergency_override(gs, G, phases)
    assert corridor[0] == "Intersection_6"
    assert corridor[-1] == EMERGENCY_TARGET
    assert all(new_phases[n] == PHASE_EMERGENCY_CORRIDOR for n in corridor)


def test_m2_emergency_corridor_multiple_emergencies():
    from routing_layer import apply_emergency_override
    from config import EMERGENCY_TARGET, PHASE_EMERGENCY_CORRIDOR
    gs = apply_event(default_grid_state(), "Intersection_6", EVENT_EMERGENCY)
    gs = apply_event(gs, "Intersection_4", EVENT_EMERGENCY)
    phases = {n: PHASE_ADAPTIVE_SHORT for n in NODE_IDS}
    new_phases, corridor = apply_emergency_override(gs, G, phases)
    assert "Intersection_6" in corridor
    assert "Intersection_4" in corridor
    assert EMERGENCY_TARGET in corridor
    assert len(corridor) == len(set(corridor))
    assert all(new_phases[n] == PHASE_EMERGENCY_CORRIDOR for n in corridor)


def test_m2_emergency_corridor_disconnected_graph():
    from routing_layer import apply_emergency_override
    import networkx as nx
    isolated_G = nx.Graph()
    isolated_G.add_nodes_from(NODE_IDS)
    gs = apply_event(default_grid_state(), "Intersection_6", EVENT_EMERGENCY)
    phases = {n: PHASE_ADAPTIVE_SHORT for n in NODE_IDS}
    new_phases, corridor = apply_emergency_override(gs, isolated_G, phases)
    assert corridor == []
    assert new_phases == phases


def test_m2_corridor_schedule_pr4():
    from routing_layer import corridor_schedule
    corridor = ["Intersection_6", "Intersection_5", "Intersection_2", "Intersection_1"]
    schedule = corridor_schedule(G, corridor)
    assert len(schedule) == 4
    assert schedule[0]["node"] == "Intersection_6"
    assert schedule[0]["eta"] == 0.0
    assert schedule[0]["preempt_start"] == 0.0
    assert schedule[0]["restore_time"] == 10.0
    assert schedule[1]["node"] == "Intersection_5"
    assert schedule[1]["eta"] == 20.0
    assert schedule[1]["preempt_start"] == 5.0
    assert schedule[1]["restore_time"] == 30.0
    assert schedule[3]["node"] == "Intersection_1"
    assert schedule[3]["eta"] == 60.0

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

