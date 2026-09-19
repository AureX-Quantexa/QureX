"""
OWNER: Member 4.
Layer 6b — Operator Co-pilot (Featherless AI LLM + Deterministic Fallback).
"""
import functools
from typing import List, Optional
import openai
import streamlit as st

from config import EVENT_NORMAL, PHASE_EMERGENCY_CORRIDOR, PHASE_MAX_GREEN
from contracts import GridState, KPIResult, Phases, StatsResult

FALLBACK_TAG = "[LOCAL AI CO-PILOT - DETERMINISTIC MODE]"
FEATHERLESS_BASE_URL = "https://api.featherless.ai/v1"
FEATHERLESS_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


def _generate_deterministic_summary(grid_state: GridState, phases: Phases,
                                   corridor: List[str], stats: StatsResult,
                                   kpis: Optional[KPIResult] = None) -> str:
    """Deterministic template fallback using strictly provided structured facts in <= 2 sentences."""
    anomaly_status = "Statistical anomaly detected" if stats.is_anomaly else "Network flow remains nominal"

    # Active disruptions
    disruptions = [f"{n} ({grid_state[n]['event']})" for n in grid_state if grid_state[n]["event"] != EVENT_NORMAL]
    disruption_clause = f" with incident at {', '.join(disruptions)}" if disruptions else ""

    s1 = f"{anomaly_status}{disruption_clause} (Mahalanobis D={stats.mahalanobis:.2f} vs threshold {stats.threshold:.2f})."

    # Corridor and KPI action
    if corridor:
        corridor_clause = f"Emergency green corridor deployed across {len(corridor)} nodes ({' -> '.join(corridor)}) with preemption priority."
    else:
        corridor_clause = f"QureX hybrid core coordinated {len(phases)} critical signals."

    if kpis is not None:
        kpi_clause = f" Simulation projects a {kpis.delay_saved_pct:.1f}% delay reduction and {kpis.fuel_saved_gal:.2f} gal fuel savings."
    else:
        kpi_clause = ""

    s2 = f"{corridor_clause}{kpi_clause}".strip()
    return f"{FALLBACK_TAG} {s1} {s2}"


@functools.lru_cache(maxsize=128)
def _query_featherless_cached(api_key: str, system_prompt: str, user_prompt: str) -> Optional[str]:
    """Cached Featherless AI query to guarantee warm pipeline rerun latency < 2.0s."""
    client = openai.OpenAI(base_url=FEATHERLESS_BASE_URL, api_key=api_key, timeout=10.0)
    response = client.chat.completions.create(
        model=FEATHERLESS_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=120,
        temperature=0.2,
    )
    content = response.choices[0].message.content
    if content and content.strip():
        return content.strip()
    return None


def summarize(grid_state: GridState, phases: Phases, corridor: List[str],
              stats: StatsResult, kpis: Optional[KPIResult] = None) -> str:
    """
    Summarize traffic status and optimization decisions using Featherless AI or deterministic fallback.
    Returns a formal narrative <= 2 sentences using strictly provided facts.
    """
    # Safe secrets retrieval (never crashes if secrets.toml is missing)
    api_key = None
    try:
        api_key = st.secrets["FEATHERLESS_API_KEY"]
    except Exception:
        api_key = None

    if not api_key or not isinstance(api_key, str) or not api_key.strip():
        return _generate_deterministic_summary(grid_state, phases, corridor, stats, kpis)

    # Prepare structured facts
    anomalies = "Yes (Anomaly detected)" if stats.is_anomaly else "No (Nominal)"
    critical_nodes = list(phases.keys())
    phase_summary = ", ".join(f"{n}: {ph}" for n, ph in phases.items()) if phases else "None"
    corridor_summary = f"{len(corridor)} nodes ({' -> '.join(corridor)})" if corridor else "None"
    delay_saved = f"{kpis.delay_saved_pct:.1f}%" if kpis else "N/A"
    fuel_saved = f"{kpis.fuel_saved_gal:.2f} gal" if kpis else "N/A"

    system_prompt = (
        "You are the QureX Traffic Operations Co-pilot. "
        "Summarize grid status and signal adjustments in at most 2 concise sentences using a formal tone. "
        "Use ONLY the structured facts provided. Do not invent any numbers, percentages, or facts."
    )
    user_prompt = (
        f"FACTS:\n"
        f"- Statistical Anomaly: {anomalies} (Mahalanobis D={stats.mahalanobis:.2f}, threshold={stats.threshold:.2f})\n"
        f"- Critical Intersections: {critical_nodes}\n"
        f"- Selected Signal Phases: {phase_summary}\n"
        f"- Emergency Green Corridor: {corridor_summary}\n"
        f"- KPI Delay Savings: {delay_saved}, Fuel Saved: {fuel_saved}\n"
        f"\nGenerate a formal 1 to 2 sentence summary of the current operational state."
    )

    try:
        content = _query_featherless_cached(api_key, system_prompt, user_prompt)
        if content:
            return content
    except Exception:
        pass

    return _generate_deterministic_summary(grid_state, phases, corridor, stats, kpis)
