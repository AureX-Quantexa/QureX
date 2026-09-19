"""OWNER: Member 3.  KPIs from a queue model of the CHOSEN phases.  (STUB v0)"""
from typing import List

from config import NODE_IDS, PHASE_MAX_GREEN
from contracts import GridState, KPIResult, Phases, TriageResult


def compute_kpis(grid_state: GridState, phases: Phases, triage: TriageResult,
                 corridor: List[str]) -> KPIResult:
    fixed = {n: grid_state[n]["queue"] * 5.5 for n in NODE_IDS}
    actuated = {n: grid_state[n]["queue"] * 4.1 for n in NODE_IDS}
    quantum = {n: fixed[n] * (0.7 if phases.get(n) == PHASE_MAX_GREEN else 0.9) for n in NODE_IDS}
    saved = sum(fixed.values()) - sum(quantum.values())
    return KPIResult(fixed, actuated, quantum, 2000.0, 2000.0 + saved,
                     fuel_saved_gal=max(0.0, saved * 0.0025), co2_saved_kg=max(0.0, saved * 0.0025) * 8.887)
