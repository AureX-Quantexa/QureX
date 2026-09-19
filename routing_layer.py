"""OWNER: Member 2.  Layer 6a - emergency Dijkstra corridor.  (STUB v0 - no override yet)"""
from typing import List, Tuple

import networkx as nx

from contracts import GridState, Phases


def apply_emergency_override(grid_state: GridState, G: nx.Graph,
                             phases: Phases) -> Tuple[Phases, List[str]]:
    """Return (phases_with_override, corridor_node_list). corridor == [] if no emergency."""
    return dict(phases), []
