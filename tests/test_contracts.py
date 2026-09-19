"""Contract tests. RUN BEFORE EVERY PUSH:  pytest -q
Each member may ADD tests for their own layer below (in your own section) but never weaken the ones here."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from config import (EVENT_CODES, EVENT_EMERGENCY, EVENT_FESTIVAL, EVENT_NORMAL, NODE_IDS)
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
# --- M4 (copilot/app):
