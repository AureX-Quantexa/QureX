"""
OWNER: Member 3 (Quantum & Metrics Engineer).
Queue model simulation and environmental KPI calculation derived from chosen phases.
"""
from typing import Dict, List, Tuple
import numpy as np

from config import (EDGE_TRAVEL_COST, NODE_IDS, PHASE_ADAPTIVE_SHORT,
                    PHASE_EMERGENCY_CORRIDOR, PHASE_MAX_GREEN, PHASE_STANDARD_FIXED)
from contracts import GridState, KPIResult, Phases, TriageResult

# Idling fuel consumption rate assumption: 0.3 gallons per hour per vehicle
# Source / Assumption: Standard light-vehicle engine idling baseline (~0.3 gal/hr)
IDLE_FUEL_RATE_GAL_PER_VEH_HR = 0.3
CO2_KG_PER_GALLON = 8.887  # EPA gasoline conversion factor


def compute_kpis(grid_state: GridState, phases: Phases, triage: TriageResult,
                 corridor: List[str]) -> KPIResult:
    """Compute simulated queue model KPIs for Fixed, Actuated, and QureX controllers."""
    dt = 1.0     # 1 second simulation step
    T = 300.0    # 5-minute horizon (seconds)
    s_rate = 0.5 # 0.5 veh/s service rate
    eta = 0.9    # 0.9 lost-time efficiency factor
    mu_fixed = s_rate * 0.5 * eta

    def run_simulation(controller_mode: str) -> Tuple[Dict[str, float], float, float]:
        per_node_delay: Dict[str, float] = {}
        total_discharged_veh = 0.0

        for n in NODE_IDS:
            q_m = float(grid_state[n]["queue"])
            q_c = 0.4 * q_m  # initial cross-street queue
            cap = float(grid_state[n]["capacity"])

            # Forecast inflow rate lambda_m
            # If triage qubo_weights present, use it; otherwise standard estimate
            q_hat = q_m * 1.15
            lam_m = max(0.05, 0.8 * mu_fixed + (q_hat - q_m) / T)
            lam_c = 0.75 * lam_m

            # Determine green ratio g for main approach
            if controller_mode == "fixed":
                g = 0.5
            elif controller_mode == "actuated":
                g = float(np.clip(lam_m / (lam_m + lam_c), 0.35, 0.65))
            elif controller_mode == "qurex":
                phase = phases.get(n, PHASE_STANDARD_FIXED)
                if phase == PHASE_EMERGENCY_CORRIDOR:
                    g = 1.0  # Full priority green wave
                elif phase == PHASE_MAX_GREEN:
                    g = 0.65
                elif phase == PHASE_ADAPTIVE_SHORT:
                    g = float(np.clip(lam_m / (lam_m + lam_c), 0.35, 0.65))
                else:
                    g = 0.5
            else:
                g = 0.5

            node_delay = 0.0
            node_discharged = 0.0

            for _ in range(int(T / dt)):
                d_m = min(q_m, s_rate * g * eta * dt)
                d_c = min(q_c, s_rate * (1.0 - g) * eta * dt)

                q_m = max(0.0, q_m + lam_m * dt - d_m)
                q_c = max(0.0, q_c + lam_c * dt - d_c)

                node_delay += (q_m + q_c) * dt
                node_discharged += (d_m + d_c)

            per_node_delay[n] = float(node_delay)
            total_discharged_veh += node_discharged

        throughput_veh_hr = float(total_discharged_veh * (3600.0 / T))
        return per_node_delay, sum(per_node_delay.values()), throughput_veh_hr

    delay_fixed, total_fixed_s, tp_baseline = run_simulation("fixed")
    delay_actuated, total_actuated_s, _ = run_simulation("actuated")
    delay_quantum, total_qurex_s, tp_optimized = run_simulation("qurex")

    saved_veh_s = max(0.0, total_fixed_s - total_qurex_s)
    fuel_saved_gal = float((saved_veh_s / 3600.0) * IDLE_FUEL_RATE_GAL_PER_VEH_HR)
    co2_saved_kg = float(fuel_saved_gal * CO2_KG_PER_GALLON)

    # Approved PR-3: Optional Emergency Travel Time metrics
    t_base_s = 0.0
    t_corr_s = 0.0
    disruption_s = 0.0

    if corridor:
        corr_len = len(corridor)
        # Baseline travel time: free flow + queue clearing + average red cycle delay
        t_base_s = corr_len * EDGE_TRAVEL_COST + sum(grid_state[n]["queue"] / mu_fixed for n in corridor) + corr_len * (0.5 * 90.0 * 0.5)
        # Emergency corridor travel time: free flow + residual delay (3s)
        t_corr_s = corr_len * EDGE_TRAVEL_COST + 3.0
        # Cross street disruption delay
        disruption_s = corr_len * 15.0 * 10.0

    return KPIResult(
        delay_fixed=delay_fixed,
        delay_actuated=delay_actuated,
        delay_quantum=delay_quantum,
        throughput_baseline=tp_baseline,
        throughput_optimized=tp_optimized,
        fuel_saved_gal=fuel_saved_gal,
        co2_saved_kg=co2_saved_kg,
        label="Simulated estimate (queue model)"
    )
