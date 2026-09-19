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
def test_m4_copilot_deterministic_fallback():
    import copilot_layer
    from scenario import default_grid_state
    from contracts import StatsResult
    import streamlit as st

    orig_secrets = getattr(st, "secrets", None)
    try:
        st.secrets = {}
        gs = default_grid_state()
        stats = StatsResult(mahalanobis=2.3, threshold=6.5, is_anomaly=False,
                            eigen_features=np.zeros(6), variance_retained=0.96)
        phases = {NODE_IDS[0]: PHASE_MAX_GREEN}
        corridor = [NODE_IDS[1], NODE_IDS[0]]
        summary = copilot_layer.summarize(gs, phases, corridor, stats)
        assert isinstance(summary, str) and len(summary) > 0
        assert summary.startswith(copilot_layer.FALLBACK_TAG)
        assert "D=2.30" in summary
        assert "Intersection_1" in summary
    finally:
        if orig_secrets is not None:
            st.secrets = orig_secrets


def test_m4_copilot_format_and_facts():
    import copilot_layer
    from scenario import default_grid_state
    from contracts import StatsResult, KPIResult

    gs = default_grid_state()
    stats = StatsResult(mahalanobis=7.8, threshold=6.5, is_anomaly=True,
                        eigen_features=np.zeros(6), variance_retained=0.95)
    phases = {NODE_IDS[0]: PHASE_MAX_GREEN, NODE_IDS[1]: PHASE_ADAPTIVE_SHORT}
    kpis = KPIResult(
        delay_fixed={n: 100.0 for n in NODE_IDS},
        delay_actuated={n: 80.0 for n in NODE_IDS},
        delay_quantum={n: 60.0 for n in NODE_IDS},
        throughput_baseline=1200.0,
        throughput_optimized=1350.0,
        fuel_saved_gal=2.45,
        co2_saved_kg=21.77,
    )
    summary = copilot_layer._generate_deterministic_summary(gs, phases, [], stats, kpis=kpis)
    assert copilot_layer.FALLBACK_TAG in summary
    assert "D=7.80" in summary
    assert f"{kpis.delay_saved_pct:.1f}%" in summary


def test_m4_warm_pipeline_latency():
    import time
    from scenario import default_grid_state
    gs = default_grid_state()
    # Warm up
    run_pipeline(gs, G)
    # Timed run
    t0 = time.perf_counter()
    out = run_pipeline(gs, G)
    latency = time.perf_counter() - t0
    assert latency < 2.0, f"Warm pipeline latency {latency:.3f}s exceeded 2.0s threshold"
    assert out.stats is not None
    assert out.kpis is not None


def test_m4_apptest_smoke_all_events():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from config import EVENT_NORMAL, EVENT_CONGESTION, EVENT_ACCIDENT, EVENT_EMERGENCY, EVENT_FESTIVAL

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert not at.exception, f"App initial run exception: {at.exception}"

    events = [EVENT_NORMAL, EVENT_CONGESTION, EVENT_ACCIDENT, EVENT_EMERGENCY, EVENT_FESTIVAL]
    for ev in events:
        at.sidebar.selectbox[0].select(NODE_IDS[1])
        at.sidebar.selectbox[1].select(ev)
        at.sidebar.button[0].click().run()
        assert not at.exception, f"AppTest failed for event {ev}: {at.exception}"

    at.sidebar.button[1].click().run()
    assert not at.exception, f"AppTest Reset failed: {at.exception}"

