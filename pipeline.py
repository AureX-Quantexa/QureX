"""
OWNER: Member 4.
The ONE place that defines the call order. app.py and tests both use it.

state -> statistics -> forecast -> triage -> optimize -> emergency override -> KPIs + co-pilot
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any

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
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def run_pipeline(grid_state: GridState, G: nx.Graph) -> PipelineOutput:
    """
    Execute the frozen 6-layer pipeline:
    L1: Ingestion (grid_state provided)
    L2: Statistical sanitization (Mahalanobis + PCA)
    L3: Predictive wave forecast (XGBoost)
    L4: Graph triage (bottleneck subgraph identification)
    L5: Quantum core (CVaR-QAOA optimization)
    L6a: Emergency corridor override (Dijkstra)
    Metrics: KPIs derived from queue model
    L6b: Operator co-pilot narrative
    """
    stats = stats_layer.run_statistics(grid_state)
    forecast = forecast_layer.forecast_inflows(stats, grid_state)
    triage = graph_layer.triage(grid_state, forecast, G)
    
    # QAOA Quantum Core optimization with PR-5 diagnostics
    phases, diagnostics = quantum_layer.optimize_with_diagnostics(triage)
    
    # Layer 6a Emergency override
    phases, corridor = routing_layer.apply_emergency_override(grid_state, G, phases)
    
    # Metrics queue model & KPIs
    kpis = metrics.compute_kpis(grid_state, phases, triage, corridor)
    
    # Layer 6b Operator co-pilot summary
    narrative = copilot_layer.summarize(grid_state, phases, corridor, stats, kpis=kpis)
    
    return PipelineOutput(
        stats=stats,
        forecast=forecast,
        triage=triage,
        phases=phases,
        corridor=corridor,
        kpis=kpis,
        narrative=narrative,
        diagnostics=diagnostics
    )
