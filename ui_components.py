"""
OWNER: Member 4.
UI Component Library and Plotting Helpers for QureX Dashboard.
"Where Quantum meets the Road."
"""
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

from config import (EDGE_TRAVEL_COST, EMERGENCY_TARGET, EVENT_LABELS, EVENT_NORMAL,
                    GRID_COLS, GRID_ROWS, NODE_IDS, NODE_COORDS, PHASE_ADAPTIVE_SHORT,
                    PHASE_EMERGENCY_CORRIDOR, PHASE_LABELS, PHASE_MAX_GREEN,
                    PHASE_STANDARD_FIXED)
from contracts import GridState, KPIResult, Phases, StatsResult
import pydeck as pdk


def apply_custom_styles() -> None:
    """Inject custom modern SaaS light-mode CSS styling with Google Fonts."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
        
        /* Base typography and background */
        html, body, [class*="css"], .stApp, .stMarkdown, .stText, p, span, div, h1, h2, h3, h4, h5, h6, label {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            color: #3D4459 !important; /* Darkest Navy from palette for maximum contrast */
        }
        
        .stApp {
            background-color: #EAECEF !important;
        }
        
        /* Captions and secondary text */
        .stCaption, .stCaption p, [data-testid="stCaptionContainer"] {
            color: #9A9EA6 !important;
        }
        
        code, pre {
            font-family: 'JetBrains Mono', monospace !important;
            color: #3D4459 !important;
        }

        /* Clean White Header Box */
        .qurex-header-box {
            background: #FFFFFF;
            border: 1px solid #D0D4D9;
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        }

        .qurex-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #4A5060;
            margin: 0;
            letter-spacing: -0.02em;
        }

        .qurex-subtitle {
            font-size: 1.1rem;
            color: #9A9EA6;
            margin-top: 0.5rem;
            font-weight: 400;
        }

        .simulated-badge {
            display: inline-block;
            background: linear-gradient(90deg, #f59e0b, #d97706);
            color: #ffffff;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 3px 10px;
            border-radius: 9999px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-left: 12px;
            vertical-align: middle;
            box-shadow: 0 2px 8px rgba(245, 158, 11, 0.4);
        }

        .copilot-card {
            background: #FFFFFF;
            border-left: 4px solid #8b5cf6;
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
            color: #4A5060;
        }

        .metric-card {
            background: #FFFFFF;
            border: 1px solid #D0D4D9;
            border-radius: 12px;
            padding: 14px 18px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }

        .corridor-step {
            background: #FEF2F2;
            border: 1px solid #FCA5A5;
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 8px;
            color: #991B1B;
        }

        .standby-card {
            background: #F0FDF4;
            border: 1px solid #86EFAC;
            border-radius: 8px;
            padding: 14px;
            color: #166534;
        }

        /* Metric Cards Override */
        [data-testid="stMetric"] {
            background-color: #FFFFFF !important;
            border: 1px solid #D0D4D9 !important;
            border-radius: 10px !important;
            padding: 1.2rem !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            transition: all 0.2s ease-in-out;
        }
        
        [data-testid="stMetric"]:hover {
            box-shadow: 0 4px 6px rgba(0,0,0,0.08);
            transform: translateY(-2px);
        }

        [data-testid="stMetricLabel"] {
            color: #9A9EA6 !important;
            font-weight: 500 !important;
            font-size: 0.9rem !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        [data-testid="stMetricValue"] {
            color: #4A5060 !important;
            font-weight: 700 !important;
            font-size: 2rem !important;
            letter-spacing: -0.02em !important;
        }
        
        /* Metric Delta colors override to fit palette */
        [data-testid="stMetricDelta"] svg {
            color: #4A5060 !important;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #D0D4D9 !important;
        }
        
        /* Tabs */
        [data-testid="stTabs"] button {
            color: #9A9EA6 !important;
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
            color: #4A5060 !important;
            border-bottom-color: #4A5060 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render the official QureX header with brand tagline and prominent Simulated data badge."""
    st.markdown(
        """
        <div class="qurex-header-box">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
                <div>
                    <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em;">
                        🚦 QureX
                        <span class="simulated-badge">Simulated data</span>
                    </h1>
                    <p style="margin: 4px 0 0 0; font-size: 1.05rem; color: #94a3b8; font-weight: 400;">
                        <em>Where Quantum meets the Road.</em> &mdash; Hybrid Quantum-Classical Urban Traffic Optimization
                    </p>
                </div>
                <div style="margin-top: 8px; text-align: right;">
                    <span style="background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); color: #38bdf8; font-size: 0.8rem; font-weight: 600; padding: 4px 10px; border-radius: 6px;">
                        CVaR-QAOA L5 Core • Dijkstra Corridor L6a • Operator Co-pilot L6b
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_copilot_box(narrative: str) -> None:
    """Panel 8: Operator Co-pilot Summary Box."""
    is_fallback = "[LOCAL AI CO-PILOT - DETERMINISTIC MODE]" in narrative
    badge_label = "LOCAL DETERMINISTIC AI" if is_fallback else "FEATHERLESS AI ONLINE"
    badge_color = "#f59e0b" if is_fallback else "#10b981"
    clean_text = narrative.replace("[LOCAL AI CO-PILOT - DETERMINISTIC MODE]", "").strip()

    st.markdown(
        f"""
        <div class="copilot-card">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <span style="font-weight: 600; font-size: 0.95rem; color: #e2e8f0; display: flex; align-items: center; gap: 8px;">
                    🤖 Layer 6b Operator Co-pilot
                </span>
                <span style="font-size: 0.75rem; font-weight: 700; color: {badge_color}; background: rgba(0,0,0,0.3); padding: 2px 8px; border-radius: 4px; border: 1px solid {badge_color}44;">
                    {badge_label}
                </span>
            </div>
            <p style="margin: 0; font-size: 1.0rem; line-height: 1.5; color: #cbd5e1;">
                {clean_text}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_network_graph(G: nx.Graph, grid_state: GridState, phases: Phases,
                         corridor: List[str], forecast: np.ndarray) -> None:
    """
    Panel 2: Matplotlib 2x3 grid network topology graph.
    - Node color mapped to predicted utilization u_i = forecast_i / capacity_i.
    - Emergency corridor highlighted in bold red.
    - Phase labels displayed on nodes.
    - plt.close(fig) called after rendering.
    """
    # Explicit 2x3 grid layout (row 0 at top y=1, row 1 at bottom y=0)
    pos = {
        "Intersection_1": (0.0, 1.0),
        "Intersection_2": (1.0, 1.0),
        "Intersection_3": (2.0, 1.0),
        "Intersection_4": (0.0, 0.0),
        "Intersection_5": (1.0, 0.0),
        "Intersection_6": (2.0, 0.0),
    }

    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")

    # Compute utilization per node: u_i = forecast_i / capacity_i
    utilizations = []
    for i, n in enumerate(NODE_IDS):
        cap = max(1, grid_state[n]["capacity"])
        u = min(1.0, float(forecast[i]) / cap)
        utilizations.append(u)

    cmap = plt.cm.YlOrRd
    node_colors = [cmap(u) for u in utilizations]

    # Draw standard edges
    corridor_edges = set()
    if len(corridor) > 1:
        for k in range(len(corridor) - 1):
            corridor_edges.add((corridor[k], corridor[k + 1]))
            corridor_edges.add((corridor[k + 1], corridor[k]))

    for u, v in G.edges():
        is_corr = (u, v) in corridor_edges
        if not is_corr:
            x_vals = [pos[u][0], pos[v][0]]
            y_vals = [pos[u][1], pos[v][1]]
            ax.plot(x_vals, y_vals, color="#475569", lw=3.0, zorder=1, linestyle="-")

    # Draw emergency corridor edges highlighted in bold red
    for u, v in G.edges():
        if (u, v) in corridor_edges:
            x_vals = [pos[u][0], pos[v][0]]
            y_vals = [pos[u][1], pos[v][1]]
            ax.plot(x_vals, y_vals, color="#ef4444", lw=6.0, zorder=2, linestyle="-")
            # Draw glow line
            ax.plot(x_vals, y_vals, color="#f87171", lw=10.0, alpha=0.35, zorder=1)

    # Draw nodes
    for idx, n in enumerate(NODE_IDS):
        x, y = pos[n]
        is_corr = n in corridor
        edge_col = "#ef4444" if is_corr else "#38bdf8"
        edge_lw = 4.0 if is_corr else 2.0
        radius = 0.16 if is_corr else 0.14

        circle = plt.Circle((x, y), radius, facecolor=node_colors[idx], edgecolor=edge_col,
                            linewidth=edge_lw, zorder=3)
        ax.add_patch(circle)

        # Pulse glow for corridor nodes
        if is_corr:
            glow = plt.Circle((x, y), radius + 0.05, facecolor="none", edgecolor="#ef4444",
                              linewidth=2.0, linestyle="--", alpha=0.6, zorder=2)
            ax.add_patch(glow)

        # Labels on nodes: Node Name & Phase
        ph = phases.get(n, PHASE_STANDARD_FIXED)
        if ph == PHASE_MAX_GREEN:
            ph_text = "MAX GREEN"
            ph_col = "#15803d"
        elif ph == PHASE_ADAPTIVE_SHORT:
            ph_text = "ADAPT SHORT"
            ph_col = "#1e293b"
        elif ph == PHASE_EMERGENCY_CORRIDOR:
            ph_text = "PRIORITY CORR"
            ph_col = "#b91c1c"
        else:
            ph_text = "FIXED"
            ph_col = "#334155"

        # Short node label
        short_name = n.replace("Intersection_", "I-")
        ax.text(x, y + 0.02, short_name, ha="center", va="center", color="#0f172a",
                fontsize=8.5, fontweight="bold", zorder=4)
        ax.text(x, y - 0.045, f"u={utilizations[idx]:.2f}", ha="center", va="center",
                color="#0f172a", fontsize=7.0, fontweight="bold", zorder=4)

        # Phase tag badge above/below node
        tag_y = y + 0.22 if y == 0.0 else y + 0.22
        ax.text(x, tag_y, ph_text, ha="center", va="center", color="#ffffff",
                fontsize=6.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc=ph_col, ec=edge_col, lw=1.0),
                zorder=4)

    # Margins and bounds
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(-0.25, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")

    # Legend
    legend_patches = [
        mpatches.Patch(color="#fef08a", label="Low (u<0.4)"),
        mpatches.Patch(color="#f97316", label="Med (0.4≤u≤0.7)"),
        mpatches.Patch(color="#dc2626", label="High (u>0.7)"),
        mpatches.Patch(edgecolor="#ef4444", facecolor="none", lw=2, label="Corridor Path"),
    ]
    ax.legend(handles=legend_patches, loc="lower center", ncol=4, frameon=False,
              fontsize=7.5, labelcolor="#94a3b8", bbox_to_anchor=(0.5, -0.08))

    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def render_status_table(grid_state: GridState, forecast: np.ndarray, phases: Phases) -> None:
    """Panel 3: Per-Node Status Table with stretch width."""
    rows = []
    for i, n in enumerate(NODE_IDS):
        s = grid_state[n]
        ph = phases.get(n, PHASE_STANDARD_FIXED)
        rows.append({
            "Intersection": n,
            "Event Label": EVENT_LABELS.get(s["event"], s["event"]),
            "Queue (veh)": int(s["queue"]),
            "Capacity": int(s["capacity"]),
            "5-min Forecast": round(float(forecast[i]), 1),
            "Signal Strategy": PHASE_LABELS.get(ph, ph),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)


def render_emergency_corridor_panel(corridor: List[str], grid_state: GridState) -> None:
    """Panel 4: Emergency Green Corridor Route Breakdown & ETA per hop."""
    st.markdown("### 🚨 Emergency Green Corridor Status")
    if not corridor:
        st.markdown(
            f"""
            <div class="standby-card">
                <strong>🟢 Corridor Standby Mode:</strong> No emergency preemption active.<br>
                All 6 intersections operating under adaptive hybrid coordination. Target hospital node: <code>{EMERGENCY_TARGET}</code>.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    hop_count = len(corridor)
    total_eta = (hop_count - 1) * EDGE_TRAVEL_COST if hop_count > 1 else 0.0

    st.markdown(
        f"""
        <div style="background: rgba(239, 68, 68, 0.12); border: 1px solid #ef4444; border-radius: 10px; padding: 14px; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 700; color: #f87171; font-size: 1.05rem;">
                    🚨 PREEMPTION OVERRIDE ACTIVE &mdash; Priority Wave Deployed
                </span>
                <span style="background: #dc2626; color: white; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 0.8rem;">
                    Total Transit ETA: {total_eta:.0f} s
                </span>
            </div>
            <p style="color: #fca5a5; font-size: 0.9rem; margin: 6px 0 0 0;">
                Corridor Dijkstra route: <strong>{' &rarr; '.join(corridor)}</strong> targeting <strong>{EMERGENCY_TARGET}</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hop breakdown
    cols = st.columns(min(hop_count, 4))
    cum_time = 0.0
    for idx, node in enumerate(corridor):
        col = cols[idx % min(hop_count, 4)]
        if idx > 0:
            cum_time += EDGE_TRAVEL_COST
        is_dest = (node == EMERGENCY_TARGET)
        role = "Target (Hospital)" if is_dest else ("Origin" if idx == 0 else f"Hop {idx}")
        col.markdown(
            f"""
            <div class="corridor-step">
                <div style="font-size: 0.75rem; font-weight: 600; color: #f87171;">{role}</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: #ffffff;">{node}</div>
                <div style="font-size: 0.8rem; color: #94a3b8;">ETA: <strong>+{cum_time:.0f}s</strong> | Clear: 100%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_kpi_comparison_chart(kpis: KPIResult) -> None:
    """Panel 5: Grouped bar chart comparing Fixed-time vs Actuated vs QureX delays."""
    st.markdown("### 📊 Delay Comparison: Classical vs Actuated vs QureX")
    
    # Metric cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Delay Reduction", f"{kpis.delay_saved_pct:.1f}%", "vs Fixed Timing")
    m2.metric("Optimized Throughput", f"{kpis.throughput_optimized:.0f} veh/h",
              f"{kpis.throughput_optimized - kpis.throughput_baseline:+.0f} veh/h")
    m3.metric("Fuel Saved", f"{kpis.fuel_saved_gal:.2f} gal", "5-min simulated")
    m4.metric("CO₂ Reduced", f"{kpis.co2_saved_kg:.2f} kg", "EPA Standard Factor")

    fig, ax = plt.subplots(figsize=(8.0, 3.2), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")

    nodes = NODE_IDS
    x = np.arange(len(nodes))
    bar_width = 0.26

    fixed_vals = [kpis.delay_fixed[n] for n in nodes]
    actuated_vals = [kpis.delay_actuated[n] for n in nodes]
    qurex_vals = [kpis.delay_quantum[n] for n in nodes]

    rects1 = ax.bar(x - bar_width, fixed_vals, bar_width, label="Fixed-time (50/50)",
                    color="#64748b", edgecolor="#94a3b8", alpha=0.85)
    rects2 = ax.bar(x, actuated_vals, bar_width, label="Actuated (Rule-based)",
                    color="#38bdf8", edgecolor="#0284c7", alpha=0.85)
    rects3 = ax.bar(x + bar_width, qurex_vals, bar_width, label="QureX (Hybrid Quantum)",
                    color="#10b981", edgecolor="#059669", alpha=0.95)

    short_labels = [n.replace("Intersection_", "I-") for n in nodes]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Delay (veh·s)", color="#cbd5e1", fontsize=9)
    ax.tick_params(colors="#94a3b8")
    ax.grid(axis="y", linestyle="--", alpha=0.15, color="#ffffff")

    for spine in ax.spines.values():
        spine.set_color("#334155")

    ax.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f1f5f9", fontsize=8)
    fig.tight_layout()

    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def render_statistical_diagnostics(stats: StatsResult) -> None:
    """Panel 6: Mahalanobis D vs Threshold badge & PCA Scree plot."""
    st.markdown("### 🔬 Layer 2: Statistical Sanitization & PCA Scree")

    col_metric, col_plot = st.columns([1, 2])

    with col_metric:
        st.markdown("#### Anomaly Cutoff")
        status_text = "ANOMALOUS (Event Detected)" if stats.is_anomaly else "NOMINAL (Normal Bounds)"
        status_color = "#ef4444" if stats.is_anomaly else "#10b981"
        
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.7); border-radius: 10px; padding: 14px; border: 1px solid #334155;">
                <div style="font-size: 0.8rem; color: #94a3b8;">Mahalanobis Distance D(x)</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #ffffff;">{stats.mahalanobis:.2f}</div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">χ² Cutoff Threshold (α=0.001): <strong>{stats.threshold:.2f}</strong></div>
                <div style="margin-top: 10px; font-size: 0.85rem; font-weight: 700; color: {status_color};">
                    &bull; {status_text}
                </div>
                <div style="margin-top: 12px; font-size: 0.8rem; color: #94a3b8;">
                    Variance Retained: <strong>{stats.variance_retained:.1%}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_plot:
        # Scree plot for eigen_features
        k_components = len(stats.eigen_features) if hasattr(stats.eigen_features, "__len__") else 6
        k_components = max(1, k_components)

        fig, ax = plt.subplots(figsize=(5.5, 2.4), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")

        comp_indices = np.arange(1, k_components + 1)
        # Synthetic decayed scree distribution summing to variance_retained
        raw_weights = np.exp(-0.4 * np.arange(k_components))
        var_per_comp = (raw_weights / raw_weights.sum()) * stats.variance_retained
        cum_var = np.cumsum(var_per_comp)

        ax.bar(comp_indices, var_per_comp, color="#6366f1", alpha=0.7, label="Component Variance")
        ax.plot(comp_indices, cum_var, color="#f59e0b", marker="o", lw=2, label="Cumulative Variance")
        ax.axhline(0.95, color="#10b981", linestyle="--", lw=1.2, label="95% Threshold")

        ax.set_xlabel("PCA Component k", color="#94a3b8", fontsize=8)
        ax.set_ylabel("Variance", color="#94a3b8", fontsize=8)
        ax.set_xticks(comp_indices)
        ax.tick_params(colors="#94a3b8", labelsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.15, color="#ffffff")

        for spine in ax.spines.values():
            spine.set_color("#334155")

        ax.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f1f5f9", fontsize=7, loc="lower right")
        fig.tight_layout()

        st.pyplot(fig, clear_figure=True)
        plt.close(fig)


def render_qaoa_diagnostics(diagnostics: dict, critical_nodes: List[str]) -> None:
    """Panel 7: QAOA Diagnostics Expander (Approved PR-5)."""
    with st.expander("⚛️ Layer 5: CVaR-QAOA Quantum Core Diagnostics (Approved PR-5)", expanded=False):
        if not critical_nodes:
            st.info("Zero bottleneck nodes exceeded the triage utilization threshold (u > 0.60). Standard fixed baseline active; 0 qubits engaged.")
            return

        r_val = diagnostics.get("r", 1.0)
        p_opt = diagnostics.get("prob_opt", 1.0)
        history = diagnostics.get("history", [])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Qubits Used (n)", f"{len(critical_nodes)}")
        c2.metric("Approximation Ratio r", f"{r_val:.3f}", "vs Brute-Force Oracle")
        c3.metric("Optimum Probability", f"{p_opt:.1%}", f"vs Uniform 1/2^{len(critical_nodes)}")
        c4.metric("Elite QUBO Energy", f"{diagnostics.get('elite_cost', 0.0):.2f}")

        col_hist, col_bits = st.columns([3, 2])

        with col_hist:
            if history:
                fig, ax = plt.subplots(figsize=(5.0, 2.4), facecolor="#0f172a")
                ax.set_facecolor("#0f172a")
                ax.plot(history, color="#38bdf8", lw=1.8, label="CVaR_α(θ) Energy")
                ax.set_xlabel("COBYLA Evaluation Step", color="#94a3b8", fontsize=8)
                ax.set_ylabel("CVaR Objective", color="#94a3b8", fontsize=8)
                ax.tick_params(colors="#94a3b8", labelsize=8)
                ax.grid(True, linestyle="--", alpha=0.15, color="#ffffff")
                for spine in ax.spines.values():
                    spine.set_color("#334155")
                ax.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f1f5f9", fontsize=7)
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
            else:
                st.caption("No optimizer trajectory recorded.")

        with col_bits:
            st.markdown("**Critical Node Qubit Mapping:**")
            bit_data = []
            for idx, node in enumerate(critical_nodes):
                bit_data.append({
                    "Qubit Wire": f"Wire {idx}",
                    "Intersection": node,
                    "Decision": "MAX_GREEN (1)" if r_val >= 0.95 else "ADAPTIVE (0)",
                })
            st.dataframe(pd.DataFrame(bit_data), width="stretch", hide_index=True)


def render_pydeck_map(G, grid_state, phases, corridor, forecast) -> None:
    """Render an interactive map using PyDeck (Native WebGL)."""
    # Compute center
    lats = [coord[0] for coord in NODE_COORDS.values()]
    lons = [coord[1] for coord in NODE_COORDS.values()]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    # Compute utilization
    utilizations = {}
    for i, n in enumerate(NODE_IDS):
        cap = max(1, grid_state[n]["capacity"])
        u = min(1.0, float(forecast[i]) / cap)
        utilizations[n] = u

    corridor_edges = set()
    if len(corridor) > 1:
        for k in range(len(corridor) - 1):
            corridor_edges.add((corridor[k], corridor[k + 1]))
            corridor_edges.add((corridor[k + 1], corridor[k]))

    # Edges data
    edges_data = []
    for u, v in G.edges():
        is_corr = (u, v) in corridor_edges
        coord_u = NODE_COORDS[u]
        coord_v = NODE_COORDS[v]
        color = [239, 68, 68, 200] if is_corr else [71, 85, 105, 150]
        edges_data.append({
            "start": [coord_u[1], coord_u[0]],
            "end": [coord_v[1], coord_v[0]],
            "color": color,
            "width": 10 if is_corr else 4
        })

    # Nodes data
    nodes_data = []
    for n in NODE_IDS:
        coord = NODE_COORDS[n]
        u = utilizations[n]
        is_corr = n in corridor
        
        if u > 0.7:
            fill_col = [220, 38, 38, 200]
        elif u > 0.4:
            fill_col = [249, 115, 22, 200]
        else:
            fill_col = [254, 240, 138, 200]
            
        nodes_data.append({
            "name": n,
            "coord": [coord[1], coord[0]],
            "color": fill_col,
            "radius": 40 if is_corr else 25
        })

    # Layers
    line_layer = pdk.Layer(
        "LineLayer",
        edges_data,
        get_source_position="start",
        get_target_position="end",
        get_color="color",
        get_width="width",
        pickable=False
    )
    
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        nodes_data,
        get_position="coord",
        get_color="color",
        get_radius="radius",
        pickable=True
    )
    
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=15,
        pitch=0
    )
    
    r = pdk.Deck(layers=[line_layer, scatter_layer], initial_view_state=view_state, tooltip={"text": "{name}"})
    st.pydeck_chart(r, height=450, use_container_width=True)
