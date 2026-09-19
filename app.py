"""OWNER: Member 4.  Streamlit dashboard.  (STUB v0 - runs end-to-end on stubs; build the real UI here)

Run:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from config import (EVENT_CODES, EVENT_LABELS, NODE_IDS, PHASE_LABELS, PHASE_STANDARD_FIXED)
from graph_layer import build_city_graph
from pipeline import run_pipeline
from scenario import apply_event, default_grid_state

st.set_page_config(page_title="QE-AUTOP", layout="wide", page_icon="🚦")
st.title("🚦 QE-AUTOP: Quantum-Enhanced Urban Traffic Optimization")


@st.cache_resource
def get_graph():
    return build_city_graph()


G = get_graph()
if "grid_state" not in st.session_state:
    st.session_state.grid_state = default_grid_state()

# ---- sidebar: incident console
st.sidebar.header("🕹️ Live Traffic Disruption Console")
target = st.sidebar.selectbox("Target Intersection", NODE_IDS)
event = st.sidebar.selectbox("Inject Event", EVENT_CODES, format_func=EVENT_LABELS.get)
if st.sidebar.button("Deploy Event to Grid"):
    st.session_state.grid_state = apply_event(st.session_state.grid_state, target, event)

# ---- run the whole pipeline
out = run_pipeline(st.session_state.grid_state, G)

# ---- render
rows = []
for i, n in enumerate(NODE_IDS):
    s = st.session_state.grid_state[n]
    rows.append({
        "Intersection": n,
        "Event": EVENT_LABELS[s["event"]],
        "Queue": s["queue"],
        "5-min forecast": round(float(out.forecast[i]), 1),
        "Signal strategy": PHASE_LABELS[out.phases.get(n, PHASE_STANDARD_FIXED)],
    })
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
st.info(out.narrative)
c1, c2, c3 = st.columns(3)
c1.metric("Mahalanobis", f"{out.stats.mahalanobis:.2f}", "ANOMALY" if out.stats.is_anomaly else "normal")
c2.metric("Variance retained", f"{out.stats.variance_retained:.1%}")
c3.metric("Delay reduced", f"{out.kpis.delay_saved_pct:.1f}%", out.kpis.label)
