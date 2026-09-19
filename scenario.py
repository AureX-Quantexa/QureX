"""OWNER: Member 2.  Grid state + event injection.  (Layer 1 - Ingestion & Scenario Simulator)"""
from copy import deepcopy

from config import (DEFAULT_CAPACITY, EVENT_ACCIDENT, EVENT_CONGESTION, EVENT_EMERGENCY,
                    EVENT_FESTIVAL, EVENT_NORMAL, NODE_IDS)
from contracts import GridState

# Approved PR-1 constant
EVENT_ROAD_CLOSURE = "ROAD_CLOSURE"

# queue, capacity, occupancy, avg_speed for each event code
_PRESETS = {
    EVENT_NORMAL:       dict(queue=12, capacity=DEFAULT_CAPACITY, occupancy=25.0, avg_speed=42.0),
    EVENT_CONGESTION:   dict(queue=45, capacity=DEFAULT_CAPACITY, occupancy=90.0, avg_speed=6.0),
    EVENT_ACCIDENT:     dict(queue=38, capacity=15,               occupancy=95.0, avg_speed=2.0),
    EVENT_EMERGENCY:    dict(queue=24, capacity=DEFAULT_CAPACITY, occupancy=60.0, avg_speed=22.0),
    EVENT_FESTIVAL:     dict(queue=40, capacity=DEFAULT_CAPACITY, occupancy=82.0, avg_speed=12.0),
    EVENT_ROAD_CLOSURE: dict(queue=25, capacity=5,                occupancy=95.0, avg_speed=1.0),
}


def default_grid_state() -> GridState:
    """Constructs baseline grid state where all nodes exhibit EVENT_NORMAL conditions."""
    return {
        n: dict(_PRESETS[EVENT_NORMAL], event=EVENT_NORMAL)
        for n in NODE_IDS
    }


def apply_event(state: GridState, node: str, event_code: str) -> GridState:
    """Returns a NEW grid_state (does not mutate the input). FESTIVAL applies to all nodes."""
    new_state = deepcopy(state)
    targets = NODE_IDS if event_code == EVENT_FESTIVAL else [node]
    preset = _PRESETS.get(event_code, dict(_PRESETS[EVENT_NORMAL]))
    for n in targets:
        if n in new_state:
            new_state[n] = dict(preset, event=event_code)
    return new_state

