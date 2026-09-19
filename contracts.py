"""
FROZEN DATA CONTRACTS between layers  (all 4 members must approve any change)

If integration reveals a mismatch, the PRODUCER of the data fixes it to match this file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from config import (EVENT_CODES, FEATURES, N_NODES, N_RAW_FEATURES, NODE_IDS,
                    PHASE_CODES)

# -------------------------------------------------------------------- aliases
# GridState: {node_id: {"queue": int, "capacity": int, "event": EVENT_CODE,
#                       "occupancy": float, "avg_speed": float}}
GridState = Dict[str, dict]
# Phases: {node_id: PHASE_CODE}  (only nodes that got a decision; others = STANDARD_FIXED)
Phases = Dict[str, str]


# -------------------------------------------------------------------- dataclasses
@dataclass(frozen=True)
class StatsResult:
    """Produced by stats_layer.run_statistics (M1)."""
    mahalanobis: float            # sqrt(D^2), for display
    threshold: float              # anomaly cut-off on the same scale (sqrt of chi2 quantile)
    is_anomaly: bool              # mahalanobis > threshold
    eigen_features: np.ndarray    # 1-D PCA vector for the CURRENT observation
    variance_retained: float      # real explained_variance_ratio_.sum(), 0..1


@dataclass(frozen=True)
class TriageResult:
    """Produced by graph_layer.triage (M2), consumed by quantum_layer.optimize (M3).

    critical_nodes : subset of NODE_IDS, in NODE_IDS order. Length <= MAX_QUBITS.
                     Qubit i corresponds to critical_nodes[i].
    qubo_weights   : {node_id: w >= 0}  urgency of giving that node MAX_GREEN
                     (keys == critical_nodes).
    couplings      : {(a, b): j >= 0}   interaction between ADJACENT critical nodes.
                     Key convention: NODE_IDS.index(a) < NODE_IDS.index(b).
                     Both a and b must be in critical_nodes.
    """
    critical_nodes: List[str]
    qubo_weights: Dict[str, float]
    couplings: Dict[Tuple[str, str], float]


@dataclass(frozen=True)
class KPIResult:
    """Produced by metrics.compute_kpis (M3). All values are SIMULATED estimates."""
    delay_fixed: Dict[str, float]       # seconds per node, fixed-time baseline
    delay_actuated: Dict[str, float]    # seconds per node, rule-based baseline
    delay_quantum: Dict[str, float]     # seconds per node, QE-AUTOP
    throughput_baseline: float          # veh/hr
    throughput_optimized: float         # veh/hr
    fuel_saved_gal: float
    co2_saved_kg: float
    label: str = "Simulated estimate (queue model)"

    @property
    def total_fixed(self) -> float:
        return sum(self.delay_fixed.values())

    @property
    def total_quantum(self) -> float:
        return sum(self.delay_quantum.values())

    @property
    def delay_saved_pct(self) -> float:
        return 100.0 * (self.total_fixed - self.total_quantum) / max(1.0, self.total_fixed)


# -------------------------------------------------------------------- shared helper
def flatten_grid_state(grid_state: GridState) -> np.ndarray:
    """The ONE definition of the raw feature vector. Shape (18,), float.
    Layout: node1[queue, occupancy, avg_speed], node2[...], ... in NODE_IDS order."""
    return np.array([float(grid_state[n][f]) for n in NODE_IDS for f in FEATURES])


# -------------------------------------------------------------------- validators
# Used by tests/test_contracts.py (and can be used in app.py while debugging).
def validate_grid_state(gs: GridState) -> None:
    assert list(gs.keys()) == NODE_IDS, "grid_state keys must equal NODE_IDS, in order"
    for n, s in gs.items():
        for key in ("queue", "capacity", "event", "occupancy", "avg_speed"):
            assert key in s, f"{n} missing '{key}'"
        assert s["event"] in EVENT_CODES, f"{n}: event '{s['event']}' is not an EVENT_CODE"
        assert s["capacity"] > 0, f"{n}: capacity must be > 0"


def validate_stats(r: StatsResult) -> None:
    assert isinstance(r, StatsResult)
    assert isinstance(r.eigen_features, np.ndarray) and r.eigen_features.ndim == 1, \
        "eigen_features must be a 1-D numpy array"
    assert 0.0 < r.variance_retained <= 1.0 + 1e-9
    assert r.mahalanobis >= 0.0 and r.threshold > 0.0
    assert isinstance(r.is_anomaly, (bool, np.bool_))


def validate_forecast(f: np.ndarray) -> None:
    assert isinstance(f, np.ndarray) and f.shape == (N_NODES,), \
        f"forecast must be ndarray of shape ({N_NODES},), got {getattr(f, 'shape', type(f))}"
    assert np.all(np.isfinite(f)) and np.all(f >= 0)


def validate_triage(t: TriageResult) -> None:
    from config import MAX_QUBITS
    assert isinstance(t, TriageResult)
    order = [NODE_IDS.index(n) for n in t.critical_nodes]
    assert order == sorted(order), "critical_nodes must be in NODE_IDS order"
    assert len(t.critical_nodes) <= MAX_QUBITS
    assert set(t.qubo_weights) == set(t.critical_nodes), "qubo_weights keys must equal critical_nodes"
    assert all(w >= 0 for w in t.qubo_weights.values())
    for (a, b), j in t.couplings.items():
        assert a in t.critical_nodes and b in t.critical_nodes, "coupling endpoints must be critical"
        assert NODE_IDS.index(a) < NODE_IDS.index(b), "coupling key order must follow NODE_IDS"
        assert j >= 0


def validate_phases(p: Phases) -> None:
    for n, ph in p.items():
        assert n in NODE_IDS, f"unknown node '{n}' in phases"
        assert ph in PHASE_CODES, f"'{ph}' is not a PHASE_CODE"


def validate_kpis(k: KPIResult) -> None:
    assert isinstance(k, KPIResult)
    for d in (k.delay_fixed, k.delay_actuated, k.delay_quantum):
        assert list(d.keys()) == NODE_IDS, "per-node delay dicts must follow NODE_IDS order"
        assert all(v >= 0 for v in d.values())
    assert k.fuel_saved_gal >= 0 and k.co2_saved_kg >= 0


assert N_RAW_FEATURES == 18
