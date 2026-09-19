import streamlit as st
import json
import time
import pandas as pd
from ui_components import render_pydeck_map, render_kpi_comparison_chart
from graph_layer import build_city_graph

def render_live_sumo_dashboard():
    st.markdown("## Live SUMO Quantum Analytics")
    st.caption("Auto-refreshing dashboard connected to the running 3D SUMO Simulation.")

    try:
        with open("live_sumo_state.json", "r") as f:
            data = json.load(f)
    except Exception:
        st.warning("No live SUMO data found. Please run 'python sumo_connector.py' in your terminal.")
        return

    step = data.get("step", 0)
    c_kpi = data.get("classical_kpis") or {}
    q_kpi = data.get("quantum_kpis") or {}
    q_state = data.get("quantum_state") or {}
    q_phases = data.get("quantum_phases") or {}
    q_corridor = data.get("quantum_corridor") or []
    q_forecast = data.get("quantum_forecast") or {}
    narrative = data.get("narrative", "")

    st.success(f"Connected to SUMO Simulation | Current Step: {step}")
    
    # Extract proper metrics from dicts
    q_delay = sum(q_kpi.get("delay_quantum", {}).values()) if isinstance(q_kpi.get("delay_quantum"), dict) else 0.0
    c_delay = sum(c_kpi.get("delay_fixed", {}).values()) if isinstance(c_kpi.get("delay_fixed"), dict) else 0.0
    
    q_throughput = q_kpi.get("throughput_optimized", 0.0)
    c_throughput = c_kpi.get("throughput_baseline", 0.0)
    
    q_fuel = q_kpi.get("fuel_saved_gal", 0.0)
    q_co2 = q_kpi.get("co2_saved_kg", 0.0)
    
    st.markdown("### Classical vs Quantum Comparison")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Quantum Total Delay", f"{q_delay:.1f}s", f"{c_delay - q_delay:.1f}s saved vs Classical", delta_color="normal")
    with col2:
        st.metric("Quantum Throughput", f"{q_throughput:.0f} veh/h", f"{q_throughput - c_throughput:.0f} vs Classical", delta_color="normal")
    with col3:
        st.metric("Fuel Consumption", f"{q_fuel:.2f} gal", "Quantum Saved")
    with col4:
        st.metric("CO2 Emissions", f"{q_co2:.2f} kg", "Quantum Reduced")

    st.markdown("---")
    
    col_map, col_details = st.columns([1.5, 1])
    
    with col_map:
        st.markdown("### Live Network Map")
        G = build_city_graph()
        render_pydeck_map(G, q_state, q_phases, q_corridor, q_forecast)

    with col_details:
        st.markdown("### Live Operations Feed")
        if narrative:
            st.info(narrative)
            
        st.markdown("#### Event Detection")
        for node, data_n in q_state.items():
            queue = data_n.get("queue", 0)
            if queue > 30:
                st.error(f"**Gridlock at {node}**: Queue length {queue:.0f}!")
            elif queue > 15:
                st.warning(f"**Congestion at {node}**: Queue length {queue:.0f}")

        if q_corridor:
            st.error(f"**Emergency Override Active**: Route {q_corridor}")

    st.markdown("---")
    
    # Auto-refresh using Streamlit rerun
    time.sleep(1)
    st.rerun()
