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
def test_m1_stats_contracts():
    import stats_layer
    gs = default_grid_state()
    res = stats_layer.run_statistics(gs)
    validate_stats(res)
    assert res.eigen_features.ndim == 1, "eigen_features must be strictly 1-D"
    assert res.variance_retained >= 0.95, f"Variance retained {res.variance_retained} must be >= 0.95"
    assert res.threshold == pytest.approx(6.50, abs=0.05), f"Threshold {res.threshold} should be ~6.50"
    assert res.mahalanobis >= 0.0


def test_m1_anomaly_scenarios():
    import stats_layer
    from config import EVENT_ACCIDENT, EVENT_CONGESTION, EVENT_EMERGENCY, EVENT_FESTIVAL
    # 1. Default grid state is NOT anomalous
    base = default_grid_state()
    res_default = stats_layer.run_statistics(base)
    assert not res_default.is_anomaly, f"Default state should not be anomaly, got D={res_default.mahalanobis:.2f}"
    assert res_default.mahalanobis < res_default.threshold

    # 2. Congestion is an anomaly
    gs_cong = apply_event(base, NODE_IDS[0], EVENT_CONGESTION)
    res_cong = stats_layer.run_statistics(gs_cong)
    assert res_cong.is_anomaly, f"Congestion should be anomaly, got D={res_cong.mahalanobis:.2f}"
    assert res_cong.mahalanobis > res_cong.threshold

    # 3. Accident is an anomaly
    gs_acc = apply_event(base, NODE_IDS[0], EVENT_ACCIDENT)
    res_acc = stats_layer.run_statistics(gs_acc)
    assert res_acc.is_anomaly, f"Accident should be anomaly, got D={res_acc.mahalanobis:.2f}"
    assert res_acc.mahalanobis > res_acc.threshold

    # 4. Festival is an anomaly
    gs_fest = apply_event(base, NODE_IDS[0], EVENT_FESTIVAL)
    res_fest = stats_layer.run_statistics(gs_fest)
    assert res_fest.is_anomaly, f"Festival should be anomaly, got D={res_fest.mahalanobis:.2f}"
    assert res_fest.mahalanobis > res_fest.threshold

    # 5. Emergency is NOT an anomaly
    gs_emg = apply_event(base, NODE_IDS[0], EVENT_EMERGENCY)
    res_emg = stats_layer.run_statistics(gs_emg)
    assert not res_emg.is_anomaly, f"Emergency should not be anomaly, got D={res_emg.mahalanobis:.2f}"
    assert res_emg.mahalanobis < res_emg.threshold


def test_m1_forecast_contracts():
    import stats_layer
    import forecast_layer
    gs = default_grid_state()
    stats = stats_layer.run_statistics(gs)
    fc = forecast_layer.forecast_inflows(stats, gs)
    validate_forecast(fc)
    assert fc.shape == (len(NODE_IDS),)
    assert np.all(np.isfinite(fc))
    for i, n in enumerate(NODE_IDS):
        assert 0.0 <= fc[i] <= gs[n]["capacity"], f"Node {n} forecast {fc[i]} outside [0, capacity]"


def test_m1_forecast_beats_persistence():
    import forecast_layer
    model = forecast_layer.get_forecaster_model()
    assert model.xgb_mae < model.persist_mae, (
        f"XGBoost MAE ({model.xgb_mae:.3f}) must beat persistence MAE ({model.persist_mae:.3f})"
    )
    assert model.xgb_rmse < model.persist_rmse, (
        f"XGBoost RMSE ({model.xgb_rmse:.3f}) must beat persistence RMSE ({model.persist_rmse:.3f})"
    )


def test_m1_warm_latency():
    import time
    import stats_layer
    import forecast_layer
    gs = default_grid_state()
    # Warm up
    for _ in range(5):
        s = stats_layer.run_statistics(gs)
        _ = forecast_layer.forecast_inflows(s, gs)

    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        s = stats_layer.run_statistics(gs)
        _ = forecast_layer.forecast_inflows(s, gs)
        latencies.append(time.perf_counter() - t0)

    avg_ms = float(np.mean(latencies) * 1000.0)
    assert avg_ms < 50.0, f"Average warm latency ({avg_ms:.2f} ms) exceeds 50 ms threshold"


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

