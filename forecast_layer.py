"""OWNER: Member 1.  Layer 3 - XGBoost forecast.  (STUB v0 - replace body, keep signature)"""
import numpy as np

from config import NODE_IDS
from contracts import GridState, StatsResult


def forecast_inflows(stats: StatsResult, grid_state: GridState) -> np.ndarray:
    """Return shape (6,) predicted 5-min queue per node, in NODE_IDS order."""
    return np.clip(np.array([grid_state[n]["queue"] * 1.1 for n in NODE_IDS]), 0, 50)
