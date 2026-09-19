"""OWNER: Member 3.  Layer 5 - CVaR-QAOA.  (STUB v0 - replace body, keep signature)"""
from config import PHASE_MAX_GREEN
from contracts import Phases, TriageResult


def optimize(triage: TriageResult) -> Phases:
    """Return {critical_node: PHASE_CODE}. Only critical nodes appear."""
    return {n: PHASE_MAX_GREEN for n in triage.critical_nodes}
