"""OWNER: Member 2.  Layer 6a - emergency Dijkstra corridor.  (Emergency Green Corridor Routing)"""
from typing import Dict, List, Tuple

import networkx as nx

from config import (EDGE_TRAVEL_COST, EMERGENCY_TARGET, EVENT_EMERGENCY, NODE_IDS,
                    PHASE_EMERGENCY_CORRIDOR)
from contracts import GridState, Phases


def apply_emergency_override(grid_state: GridState, G: nx.Graph,
                             phases: Phases) -> Tuple[Phases, List[str]]:
    """Applies Dijkstra shortest-path emergency routing targeting EMERGENCY_TARGET ("Intersection_1").
    
    Overrides phases along the corridor with PHASE_EMERGENCY_CORRIDOR.
    Safely handles edge cases:
      - No emergencies: returns unchanged phases and empty list.
      - Source is target: returns [EMERGENCY_TARGET].
      - Disconnected components / no path: returns empty corridor without raising exceptions.
      - Multiple simultaneous emergencies: computes ordered union of path nodes.
    """
    emergency_sources = [
        n for n in NODE_IDS
        if grid_state.get(n, {}).get("event") == EVENT_EMERGENCY
    ]

    if not emergency_sources:
        return dict(phases), []

    corridor: List[str] = []

    for src in emergency_sources:
        if src == EMERGENCY_TARGET:
            if EMERGENCY_TARGET not in corridor:
                corridor.append(EMERGENCY_TARGET)
            continue

        try:
            path = nx.shortest_path(G, source=src, target=EMERGENCY_TARGET, weight="travel_cost")
            for node in path:
                if node not in corridor:
                    corridor.append(node)
        except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXError):
            # Safe degradation if disconnected or path unavailable
            continue

    new_phases = dict(phases)
    for node in corridor:
        new_phases[node] = PHASE_EMERGENCY_CORRIDOR

    return new_phases, corridor


def corridor_schedule(G: nx.Graph, corridor: List[str]) -> List[Dict[str, float]]:
    """Computes timeline schedule for emergency corridor preemption and restoration (Approved PR-4).
    
    Returns hop ETAs, preemption start windows (15s lead time), and restore times (10s lag time).
    """
    if not corridor:
        return []

    schedule = []
    current_time = 0.0
    lead_time = 15.0
    lag_time = 10.0

    for idx, node in enumerate(corridor):
        if idx > 0:
            prev_node = corridor[idx - 1]
            if G.has_edge(prev_node, node):
                hop_cost = float(G[prev_node][node].get("travel_cost", EDGE_TRAVEL_COST))
            else:
                hop_cost = EDGE_TRAVEL_COST
            current_time += hop_cost

        preempt_start = max(0.0, current_time - lead_time)
        restore_time = current_time + lag_time

        schedule.append({
            "node": node,
            "hop": idx,
            "eta": round(current_time, 2),
            "eta_s": round(current_time, 2),
            "preempt_start": round(preempt_start, 2),
            "preempt_start_s": round(preempt_start, 2),
            "restore_time": round(restore_time, 2),
            "restore_time_s": round(restore_time, 2),
            "lead_time_s": lead_time,
            "lag_time_s": lag_time,
        })

    return schedule

