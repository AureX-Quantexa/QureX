"""OWNER: Member 2.  Grid state + event injection.  (STUB v0 - already functional, refine freely)"""
from copy import deepcopy

from config import (DEFAULT_CAPACITY, EVENT_ACCIDENT, EVENT_CONGESTION, EVENT_EMERGENCY,
                    EVENT_FESTIVAL, EVENT_NORMAL, NODE_IDS)
from contracts import GridState

# queue, capacity, occupancy, avg_speed for each event code
_PRESETS = {
    EVENT_NORMAL:     dict(queue=12, capacity=DEFAULT_CAPACITY, occupancy=25.0, avg_speed=42.0),
    EVENT_CONGESTION: dict(queue=45, capacity=DEFAULT_CAPACITY, occupancy=90.0, avg_speed=6.0),
    EVENT_ACCIDENT:   dict(queue=38, capacity=15,               occupancy=95.0, avg_speed=2.0),
    EVENT_EMERGENCY:  dict(queue=24, capacity=DEFAULT_CAPACITY, occupancy=60.0, avg_speed=22.0),
    EVENT_FESTIVAL:   dict(queue=40, capacity=DEFAULT_CAPACITY, occupancy=82.0, avg_speed=12.0),
}


def default_grid_state() -> GridState:
    return {n: dict(queue=15, capacity=DEFAULT_CAPACITY, event=EVENT_NORMAL,
                    occupancy=30.0, avg_speed=40.0) for n in NODE_IDS}


def apply_event(state: GridState, node: str, event_code: str) -> GridState:
    """Returns a NEW grid_state (does not mutate the input). FESTIVAL applies to all nodes."""
    new_state = deepcopy(state)
    targets = NODE_IDS if event_code == EVENT_FESTIVAL else [node]
    for n in targets:
        new_state[n] = dict(_PRESETS[event_code], event=event_code)
    return new_state
