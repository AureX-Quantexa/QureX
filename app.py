"""
OWNER: Member 4.
Streamlit Dashboard for QureX: "Where Quantum meets the Road."
Hybrid Quantum-Classical Urban Traffic Optimization Platform.

Run:  streamlit run app.py
"""
import streamlit as st
import pandas as pd

from config import (EVENT_ACCIDENT, EVENT_CODES, EVENT_CONGESTION, EVENT_EMERGENCY,
                    EVENT_FESTIVAL, EVENT_LABELS, EVENT_NORMAL, NODE_IDS)
from graph_layer import build_city_graph
from pipeline import run_pipeline
from scenario import apply_event, default_grid_state
from ui_components import (apply_custom_styles, render_copilot_box,
                           render_emergency_corridor_panel, render_header,
                           render_kpi_comparison_chart, render_network_graph,
                           render_qaoa_diagnostics, render_statistical_diagnostics,
                           render_status_table, render_pydeck_map)
from live_dashboard import render_live_sumo_dashboard
from chatbot import render_chat_interface

# Page configuration
st.set_page_config(
    page_title="QureX — Where Quantum meets the Road.",
    layout="wide",
    page_icon="🚦",
    initial_sidebar_state="expanded",
)

# Apply modern styling
apply_custom_styles()


@st.cache_resource
def get_graph():
    """Cache the city network graph object across reruns."""
    return build_city_graph()


# Initialize state
G = get_graph()
if "grid_state" not in st.session_state:
    st.session_state.grid_state = default_grid_state()

# ---------------------------------------------------------------- Sidebar Navigation
st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.radio("Select View:", ["Command Center", "Live SUMO Analytics", "COPS AI Assistant"])
st.sidebar.markdown("---")

if page == "Live SUMO Analytics":
    render_live_sumo_dashboard()
elif page == "COPS AI Assistant":
    render_chat_interface({})
else:
    # ---------------------------------------------------------------- Sidebar: Panel 1 (Incident Console)
    st.sidebar.markdown("## 🕹️ Incident Console & Playback")
    st.sidebar.caption("Inject disruptions or playback real historical data.")

@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("historical_traffic.csv")
    except Exception:
        return None

df_traffic = load_dataset()
if df_traffic is not None:
    st.sidebar.markdown("### ⏪ Historical Playback")
    max_idx = len(df_traffic) - 1
    t_idx = st.sidebar.slider("Scrub Dataset Time", 0, max_idx, 0, format="Row %d")
    
    if st.sidebar.button("Load Row to Grid", use_container_width=True):
        row = df_traffic.iloc[t_idx]
        new_state = default_grid_state()
        for node in NODE_IDS:
            new_state[node]["queue"] = float(row[f"{node}_queue"])
            new_state[node]["occupancy"] = float(row[f"{node}_occupancy"])
            new_state[node]["avg_speed"] = float(row[f"{node}_avg_speed"])
            new_state[node]["event"] = EVENT_NORMAL
        st.session_state.grid_state = new_state
        st.rerun()
    st.sidebar.markdown("---")

target = st.sidebar.selectbox("Target Intersection", NODE_IDS, index=0)
event = st.sidebar.selectbox(
    "Select Incident Event",
    EVENT_CODES,
    index=0,
    format_func=lambda code: EVENT_LABELS.get(code, code),
)

col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    if st.button("🚀 Deploy Event", type="primary", use_container_width=True):
        st.session_state.grid_state = apply_event(st.session_state.grid_state, target, event)
        st.rerun()

with col_btn2:
    if st.button("🔄 Reset Grid", use_container_width=True):
        st.session_state.grid_state = default_grid_state()
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Quick Scenario Presets")
c_p1, c_p2 = st.sidebar.columns(2)
with c_p1:
    if st.button("🚨 Ambulance", use_container_width=True):
        st.session_state.grid_state = apply_event(default_grid_state(), "Intersection_6", EVENT_EMERGENCY)
        st.rerun()
    if st.button("💥 Accident", use_container_width=True):
        st.session_state.grid_state = apply_event(default_grid_state(), "Intersection_2", EVENT_ACCIDENT)
        st.rerun()
with c_p2:
    if st.button("🚗 Congestion", use_container_width=True):
        st.session_state.grid_state = apply_event(default_grid_state(), "Intersection_1", EVENT_CONGESTION)
        st.rerun()
    if st.button("🎉 Festival", use_container_width=True):
        st.session_state.grid_state = apply_event(default_grid_state(), "Intersection_1", EVENT_FESTIVAL)
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(
    "**QureX Platform v1.0**\n"
    "• L1: Ingestion & Telemetry\n"
    "• L2: Mahalanobis & PCA\n"
    "• L3: XGBoost Wave Forecast\n"
    "• L4: Graph Bottleneck Triage\n"
    "• L5: CVaR-QAOA Quantum Core\n"
    "• L6a: Dijkstra Green Corridor\n"
    "• L6b: Operator Co-pilot (Featherless AI)"
)

# ---------------------------------------------------------------- Main Dashboard Body
# Official Header with Tagline & Simulated Data Badge
render_header()

# Run the 6-layer pipeline
out = run_pipeline(st.session_state.grid_state, G)

# Panel 8: Operator Co-pilot Summary Box (Displayed prominently at top of operations)
render_copilot_box(out.narrative)

# Two-column layout for Network Topology & Per-Node Status
col_left, col_right = st.columns([1.1, 1.0], gap="large")

with col_left:
    st.markdown("### 🗺️ Network Topology & Phase Allocation")
    
    tab1, tab2 = st.tabs(["🌎 Folium Map (GPS)", "🕸️ Logical Grid (Matplotlib)"])
    with tab1:
        st.caption("Interactive OpenStreetMap view with traffic & phase overlays.")
        render_pydeck_map(G, st.session_state.grid_state, out.phases, out.corridor, out.forecast)
    with tab2:
        st.caption("2×3 urban arterial grid. Nodes colored by predicted utilization $u_i = \\hat{q}_i / C_i$.")
        render_network_graph(G, st.session_state.grid_state, out.phases, out.corridor, out.forecast)

with col_right:
    st.markdown("### 📋 Intersection Telemetry & Strategy Table")
    st.caption("Real-time telemetry, 5-minute wave forecast, and QureX allocated signal strategy.")
    # Panel 3: Per-Node Status Table
    render_status_table(st.session_state.grid_state, out.forecast, out.phases)

st.markdown("---")

# Panel 4: Emergency Corridor Panel
render_emergency_corridor_panel(out.corridor, st.session_state.grid_state)

st.markdown("---")

# Two-column layout for KPI Delay Comparison & Statistical Diagnostics
col_kpi, col_stats = st.columns([1.1, 0.9], gap="large")

with col_kpi:
    # Panel 5: KPI Metrics & Bar Chart
    render_kpi_comparison_chart(out.kpis)

with col_stats:
    # Panel 6: Statistical Diagnostics (Mahalanobis D vs Threshold & PCA Scree)
    render_statistical_diagnostics(out.stats)

st.markdown("---")

# Panel 7: QAOA Diagnostics Expander (Approved PR-5)
render_qaoa_diagnostics(out.diagnostics, out.triage.critical_nodes)
