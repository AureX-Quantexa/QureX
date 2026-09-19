"""OWNER: Member 4.  Layer 6b - operator co-pilot (LLM + deterministic fallback).  (STUB v0)"""
from typing import List

from contracts import GridState, Phases, StatsResult


def summarize(grid_state: GridState, phases: Phases, corridor: List[str],
              stats: StatsResult) -> str:
    return f"[STUB] {len(phases)} nodes optimised, corridor length {len(corridor)}, D={stats.mahalanobis:.2f}."
