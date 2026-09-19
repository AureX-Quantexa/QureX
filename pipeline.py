"""OWNER: Member 4.  The ONE place that defines the call order. app.py and tests both use it.

state -> statistics -> forecast -> triage -> optimize -> emergency override -> KPIs + co-pilot
"""
from dataclasses import dataclass
from typing import List

import networkx as nx
import numpy as np

import copilot_layer
import forecast_layer
import graph_layer
import metrics
import quantum_layer
import routing_layer
import stats_layer
from contracts import GridState, KPIResult, Phases, StatsResult, TriageResult


@dataclass
class PipelineOutput:
    stats: StatsResult
    forecast: np.ndarray
    triage: TriageResult
    phases: Phases
    corridor: List[str]
    kpis: KPIResult
    narrative: str


def run_pipeline(grid_state: GridState, G: nx.Graph) -> PipelineOutput:
    stats = stats_layer.run_statistics(grid_state)
    forecast = forecast_layer.forecast_inflows(stats, grid_state)
    triage = graph_layer.triage(grid_state, forecast, G)
    phases = quantum_layer.optimize(triage)
    phases, corridor = routing_layer.apply_emergency_override(grid_state, G, phases)
    kpis = metrics.compute_kpis(grid_state, phases, triage, corridor)
    narrative = copilot_layer.summarize(grid_state, phases, corridor, stats)
    return PipelineOutput(stats, forecast, triage, phases, corridor, kpis, narrative)
