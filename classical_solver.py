"""OWNER: Member 3.  Brute-force reference solver over all 2^k states.  (STUB v0)"""
from contracts import Phases, TriageResult
from config import PHASE_MAX_GREEN


def solve_bruteforce(triage: TriageResult) -> Phases:
    return {n: PHASE_MAX_GREEN for n in triage.critical_nodes}
