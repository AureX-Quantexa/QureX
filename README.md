# 🚦 QureX — Where Quantum meets the Road.

> **Hybrid Quantum-Classical Urban Traffic Optimization Platform**  
> Multi-Intersection Congestion Management & Emergency Green Corridor Preemption.

![Simulated Data](https://img.shields.io/badge/Traffic%20Data-Simulated%20Estimate-amber)
![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)
![PennyLane](https://img.shields.io/badge/Quantum%20Core-PennyLane%200.45-purple)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit%201.64-red)
![Tests](https://img.shields.io/badge/Tests-100%25%20Passing-brightgreen)

---

## 📖 Overview

**QureX** is an end-to-end urban traffic optimization platform that combines statistical anomaly detection, predictive wave forecasting, classical graph bottleneck triage, quantum combinatorial optimization (CVaR-QAOA), and emergency corridor priority routing.

### Core Objectives
1. **Adaptive Signal Timing:** Coordinate multi-intersection traffic signals dynamically based on real-time and forecasted queue lengths.
2. **Emergency Green Corridor:** Clear an uninterrupted green wave for emergency vehicles (e.g., ambulances) targeting critical hubs (`Intersection_1`), with rapid restoration of normal traffic flow.
3. **Environmental Savings:** Quantify waiting time reduction, fuel saved, and $\text{CO}_2$ emissions avoided through discrete-time queue simulation.
4. **Quantum Explainability:** Expose QAOA optimization trajectories, approximation ratios ($r$), and comparison against an exact classical brute-force oracle ($n \le 6$).

---

## 🏛️ 6-Layer Frozen Pipeline Architecture

```text
L1  URBAN GRID INGESTION            scenario.py       -> GridState
L2  STATISTICAL SANITIZATION (PCA)  stats_layer.py    -> StatsResult (Mahalanobis D + eigen-features)
L3  PREDICTIVE WAVE FORECAST (XGB)  forecast_layer.py -> np.ndarray (6,) 5-min queue forecast
L4  GRAPH FILTRATION TRIAGE         graph_layer.py    -> TriageResult (critical bottleneck subgraph)
L5  QUANTUM CORE (CVaR-QAOA)        quantum_layer.py  -> {critical node: phase code}
L6a EMERGENCY PRIORITY CORRIDOR     routing_layer.py  -> (phases, corridor_hops)
L6b OPERATOR AI CO-PILOT            copilot_layer.py  -> Natural language executive narrative
```

**Call Sequence (enforced in `pipeline.py`):**  
`state` $\to$ `statistics` $\to$ `forecast` $\to$ `triage` $\to$ `optimize` $\to$ `emergency override` $\to$ `KPIs + co-pilot` $\to$ `render`.

---

## 🖥️ Interactive Dashboard Panels

The QureX Streamlit dashboard (`app.py`) provides 8 coordinated operational panels:

1. **🕹️ Incident Console Sidebar:** Interactive target intersection and scenario event deployment (`NORMAL`, `CONGESTION`, `ACCIDENT`, `EMERGENCY`, `FESTIVAL`), 1-click scenario presets, and grid reset.
2. **🗺️ Network Topology Graph:** Matplotlib rendering of the $2 \times 3$ arterial grid with node color mapped to utilization ($u_i = \hat{q}_i / C_i$), signal strategy tags, and bold glowing emergency corridor highlights.
3. **📋 Per-Node Telemetry & Strategy Table:** Live tabular view showing intersection ID, active disruption, queue length, capacity, 5-minute wave forecast, and allocated signal strategy.
4. **🚨 Emergency Green Corridor Status:** Real-time route progression, hop-by-hop breakdown, and cumulative transit ETA for ambulances or VVIP escorts.
5. **📊 Environmental KPI Metrics & Bar Chart:** Grouped bar chart comparing Fixed-time (50/50) vs Actuated (Rule-based) vs QureX (Hybrid Quantum) delay (veh·s), plus aggregate throughput, fuel saved (gal), and $\text{CO}_2$ reduced (kg).
6. **🔬 Statistical Diagnostics (Layer 2):** Mahalanobis distance $D(x)$ vs $\chi^2_{18, 1-\alpha}$ anomaly threshold badge, and PCA scree plot showing real $k$ components and variance retained ($\ge 95\%$).
7. **⚛️ QAOA Quantum Core Diagnostics (Approved PR-5):** Expander displaying CVaR-QAOA optimization convergence trajectory, approximation ratio ($r$), probability of optimum $P(\text{opt})$, and qubit-to-intersection phase allocation.
8. **🤖 Operator Co-pilot Summary:** Real-time narrative generated via Featherless AI (`mistralai/Mistral-7B-Instruct-v0.2`) or graceful deterministic fallback tagged `[LOCAL AI CO-PILOT - DETERMINISTIC MODE]`.

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone and enter the repository
cd QureX

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Optional: Featherless AI Setup
To enable live Featherless AI LLM summarization, create `.streamlit/secrets.toml`:
```toml
FEATHERLESS_API_KEY = "your_featherless_api_key_here"
```
*Note: If no API key is provided, the co-pilot automatically engages deterministic fallback mode with zero degradation of functionality.*

### 3. Run the Dashboard
```bash
streamlit run app.py
```

### 4. Run Contract & Integration Tests
```bash
pytest -q
```

---

## 👥 Roles & File Ownership

| Role | Member | Files Owned |
|---|---|---|
| **M1** | Stats & Forecasting | `stats_layer.py`, `forecast_layer.py` |
| **M2** | Network & Routing | `scenario.py`, `graph_layer.py`, `routing_layer.py` |
| **M3** | Quantum & Metrics | `quantum_layer.py`, `classical_solver.py`, `metrics.py` |
| **M4** | UI Dashboard & Integration | `app.py`, `pipeline.py`, `copilot_layer.py`, `ui_components.py`, `requirements*.txt`, `README.md`, `docs/` |
| **FROZEN** | All Members | `config.py`, `contracts.py` |

---

## 🧪 Definition of Done Verification

- [x] **Graceful Fallback:** Verified clean run on fresh clone with NO secrets file.
- [x] **8 Dashboard Panels:** All 8 panels operational across all 5 scenario presets.
- [x] **Warm Pipeline Latency:** $< 2.0\text{ ms}$ (well below the $2.0\text{ s}$ threshold).
- [x] **AppTest Suite:** `streamlit.testing.v1.AppTest` passes with 0 exceptions across all state transitions.
- [x] **Full Contract Test Suite:** 28 / 28 tests passing 100% green (`pytest -q`).
