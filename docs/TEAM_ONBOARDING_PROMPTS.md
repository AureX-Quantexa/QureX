# QureX — Teammate Setup Guide & Detailed Antigravity Prompts

**Tagline:** "Where Quantum meets the Road."

This document contains step-by-step setup instructions and comprehensive, self-contained kickoff prompts for each team member (**M1**, **M2**, **M3**, **M4**) running their own Antigravity agent on their designated git branch.

---

## Part 1: Step-by-Step Setup Guide (For Each Teammate)

### Step 1: Clone the Repository & Setup Python 3.12
1. Clone the GitHub repository:
   ```bash
   git clone https://github.com/AureX-Quantexa/QureX.git
   cd QureX
   ```
2. Create and activate a Python 3.12 virtual environment (Python 3.10–3.12 required):
   - **Windows:**
     ```powershell
     py -3.12 -m venv .venv
     .\.venv\Scripts\activate
     ```
   - **macOS / Linux:**
     ```bash
     python3.12 -m venv .venv
     source .venv/bin/activate
     ```
3. Upgrade pip and install required dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Verify initial baseline tests pass:
   ```bash
   pytest -q
   ```
   *(Expected output: 21 passed)*

### Step 2: Set Up Git Branch & Local Secrets
1. Create and checkout your designated git feature branch:
   - **M1:** `git checkout -b m1-stats`
   - **M2:** `git checkout -b m2-graph`
   - **M3:** `git checkout -b m3-quantum`
   - **M4:** `git checkout -b m4-ui`
2. Create local secrets file (never commit secrets to git):
   - Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.
   - Add your `FEATHERLESS_API_KEY = "your_key"` if available.

### Step 3: Launch Antigravity IDE & Start Agent Conversation
1. Open the project root folder in **Antigravity IDE**.
2. Set the agent execution mode to **Planning mode**.
3. Copy the **exact detailed prompt** for your role below and paste it into the Antigravity conversation window.

---

## Part 2: Detailed Antigravity Kickoff Prompts

---

### PROMPT FOR MEMBER 1 (M1 — Stats & Forecast Engineer)

```markdown
You are the Lead Stats and Forecasting Engineer (Member 1) for QureX ("Where Quantum meets the Road."), a hybrid quantum-classical urban traffic optimization platform.

Read `@docs/QUREX_MASTER_PROMPT.md` and `@docs/IMPLEMENTATION_PLAN.md` completely. Treat all frozen contracts and rules as binding.

### YOUR ROLE & WORKSPACE
- Git Branch: `m1-stats` (Run `git branch --show-current` first; if not on `m1-stats`, create/checkout `git checkout -b m1-stats`).
- Files Owned: `stats_layer.py`, `forecast_layer.py`
- Shared/Frozen Files (DO NOT MODIFY): `config.py`, `contracts.py`
- Other Members' Files (DO NOT MODIFY): `scenario.py`, `graph_layer.py`, `routing_layer.py`, `quantum_layer.py`, `classical_solver.py`, `metrics.py`, `copilot_layer.py`, `pipeline.py`, `app.py`.

---

### TECHNICAL SPECIFICATION & FORMULAS TO IMPLEMENT

#### 1. Layer 2 — Statistical Sanitization (`stats_layer.py`)
- Function Signature: `run_statistics(grid_state: GridState) -> StatsResult`
- Historical Baseline Simulation:
  - Generate $N = 2000$ rows $\times 18$ features baseline using a correlated latent-factor demand generator:
    - City-wide demand factor $D \sim \mathcal{N}(0,1)$, local factor $e \sim \mathcal{N}(0,1)$.
    - $z_q = 0.6 D + 0.8 e \Rightarrow \text{queue} = 15 + 3 z_q$.
    - $\text{occupancy} = 30 + 8 (0.85 z_q + 0.527 \varepsilon_1)$.
    - $\text{avg\_speed} = 40 - 5 (0.8 z_{\text{occ}} + 0.6 \varepsilon_2)$.
- Covariance & Mahalanobis Distance:
  - Compute covariance matrix with Ledoit-Wolf shrinkage: $\Sigma = \text{LedoitWolf}(X).\text{covariance\_}$; mean $\mu = \bar{X}$.
  - Compute Mahalanobis distance: $D(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$.
  - Anomaly cutoff: $\text{threshold} = \sqrt{\chi^2_{18, 1-\alpha}}$ with $\alpha = 0.001$ ($\text{threshold} \approx 6.50$). Anomaly rule: `is_anomaly = (mahalanobis > threshold)`.
- Principal Component Analysis (PCA):
  - Standardize features using `StandardScaler`.
  - Fit `PCA(n_components=0.95, random_state=42)`.
  - Transform current state feature vector (`contracts.flatten_grid_state(grid_state)`).
  - IMPORTANT: `pca.transform(x.reshape(1, -1))[0]` returns a strictly **1-D** numpy array of length $k$ (do not double-wrap array).
  - Compute `variance_retained = float(pca.explained_variance_ratio_.sum())`.
- Caching:
  - Fit the `StatsModel` ONCE as a module-level lazy singleton. Cache it so Streamlit reruns do not refit the model.

#### 2. Layer 3 — Predictive Wave Forecast (`forecast_layer.py`)
- Function Signature: `forecast_inflows(stats: StatsResult, grid_state: GridState) -> np.ndarray`
- Model Architecture:
  - Use `xgboost.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.08, subsample=0.9, tree_method="hist", random_state=42)`.
  - Input features: `stats.eigen_features` (1-D array of length $k$).
  - Multi-output target: Next 5-minute queue lengths for all 6 nodes ($y \in \mathbb{R}^{N \times 6}$), generated from queue-balance dynamics:
    $q_{t+T} = \text{clip}(q_t + (\lambda_i - \mu_i) \cdot T, 0, C_i) + \varepsilon$, where $T = 300\text{ s}$, $\varepsilon \sim \mathcal{N}(0,1)$.
- Hold-out Validation:
  - Validate model on a $20\%$ hold-out split.
  - Compute MAE and RMSE versus persistence baseline ("queue stays same"). Assert that XGBoost beats persistence.
- Output Formatting:
  - Return 1-D `np.ndarray` of shape `(6,)`, finite values, clipped to $[0, \text{capacity}_i]$ in `NODE_IDS` order.
- Caching:
  - Train forecaster ONCE and cache model singleton.

---

### DEFINITION OF DONE & NUMERIC ACCEPTANCE CRITERIA
1. `run_statistics` returns valid `StatsResult` with strictly 1-D `eigen_features`.
2. Default grid state is NOT anomalous ($D < 6.50$).
3. `CONGESTION`, `ACCIDENT`, and `FESTIVAL` scenarios set `is_anomaly = True`.
4. `EMERGENCY` scenario sets `is_anomaly = False` ($D < 6.50$).
5. `variance_retained` $\ge 0.95$.
6. `forecast_inflows` returns `ndarray` of shape `(6,)`, all finite, in $[0, \text{capacity}]$.
7. XGBoost MAE is lower than persistence baseline MAE.
8. Execution time of Layer 2 + Layer 3 is $< 50\text{ ms}$ (warm).
9. All contract tests (`pytest -q`) pass 100% green. Add unit tests for your layer in your section of `tests/test_contracts.py`.

Execute the implementation now, run `pytest -q`, and present a detailed walkthrough of your results.
```

---

### PROMPT FOR MEMBER 2 (M2 — Network, Scenarios & Emergency Routing Engineer)

```markdown
You are the Lead Network, Scenarios, and Emergency Routing Engineer (Member 2) for QureX ("Where Quantum meets the Road."), a hybrid quantum-classical urban traffic optimization platform.

Read `@docs/QUREX_MASTER_PROMPT.md` and `@docs/IMPLEMENTATION_PLAN.md` completely. Treat all frozen contracts and rules as binding.

### YOUR ROLE & WORKSPACE
- Git Branch: `m2-graph` (Run `git branch --show-current` first; if not on `m2-graph`, create/checkout `git checkout -b m2-graph`).
- Files Owned: `scenario.py`, `graph_layer.py`, `routing_layer.py`
- Shared/Frozen Files (DO NOT MODIFY): `config.py`, `contracts.py`
- Other Members' Files (DO NOT MODIFY): `stats_layer.py`, `forecast_layer.py`, `quantum_layer.py`, `classical_solver.py`, `metrics.py`, `copilot_layer.py`, `pipeline.py`, `app.py`.

---

### TECHNICAL SPECIFICATION & FORMULAS TO IMPLEMENT

#### 1. Layer 1 — Ingestion & Scenario Simulator (`scenario.py`)
- Function Signatures:
  - `default_grid_state() -> GridState`
  - `apply_event(state: GridState, node: str, event_code: str) -> GridState`
- Event Presets $(queue, capacity, occupancy, avg\_speed)$:
  - `EVENT_NORMAL`: $(12, 50, 25.0\%, 42.0\text{ km/h})$
  - `EVENT_CONGESTION`: $(45, 50, 90.0\%, 6.0\text{ km/h})$
  - `EVENT_ACCIDENT`: $(38, 15, 95.0\%, 2.0\text{ km/h})$
  - `EVENT_EMERGENCY`: $(24, 50, 60.0\%, 22.0\text{ km/h})$
  - `EVENT_FESTIVAL`: $(40, 50, 82.0\%, 12.0\text{ km/h})$
- Rules:
  - `apply_event` MUST return a deep copy (new state) without mutating the input state.
  - `EVENT_FESTIVAL` applies the festival preset to ALL nodes in the grid.
  - Approved PR-1: Handle `EVENT_ROAD_CLOSURE` by setting capacity = 5, speed = 1.0 km/h.

#### 2. Layer 4 — Graph Filtration Triage (`graph_layer.py`)
- Function Signatures:
  - `build_city_graph() -> nx.Graph`
  - `triage(grid_state: GridState, forecast: np.ndarray, G: nx.Graph) -> TriageResult`
- Graph Construction:
  - Construct a $2 \times 3$ grid NetworkX graph with explicit `NODE_IDS` ordering (`Intersection_1` to `Intersection_6`).
  - Set edge attribute `travel_cost = 20.0` seconds for all 7 grid edges.
- Triage Logic:
  - Node utilization: $u_i = \hat{q}_i / C_i$ ($\hat{q}_i$ from forecast).
  - Critical node condition: $u_i > 0.60$ OR $event_i \neq \text{EVENT\_NORMAL}$.
  - Cap `critical_nodes` to `MAX_QUBITS = 6` (if over 6, retain top 6 by normalized urgency $w_i = q_i / C_i$).
  - Maintain `critical_nodes` strictly in `NODE_IDS` order.
- Weights & Couplings:
  - Urgency weights: $w_i = q_i / C_i$, normalized so $\max w_i = 1.0$.
  - Coupling weights between adjacent critical nodes: $J_{ij} = \kappa \cdot \frac{w_i + w_j}{2} \cdot \frac{\tau_{\text{ref}}}{\tau_{ij}}$ ($\kappa = 0.5$).
  - Coupling key convention: Tuple $(a, b)$ with strict ordering `NODE_IDS.index(a) < NODE_IDS.index(b)`. Both endpoints must be in `critical_nodes`.

#### 3. Layer 6a — Emergency Green Corridor Routing (`routing_layer.py`)
- Function Signature: `apply_emergency_override(grid_state: GridState, G: nx.Graph, phases: Phases) -> Tuple[Phases, List[str]]`
- Routing Algorithm:
  - Target node: `EMERGENCY_TARGET = "Intersection_1"`.
  - Loop over all nodes where $event_i == \text{EVENT\_EMERGENCY}$.
  - Compute shortest path using Dijkstra: `nx.shortest_path(G, source=src, target=EMERGENCY_TARGET, weight="travel_cost")`.
- Edge Case Handling:
  - If source == `EMERGENCY_TARGET` $\Rightarrow$ return corridor `["Intersection_1"]`.
  - If no path exists (disconnected graph) $\Rightarrow$ return corridor `[]` without crashing.
  - If multiple emergency sources exist $\Rightarrow$ compute ordered union of corridor path nodes.
- Phase Overrides:
  - Every node in the emergency corridor receives `PHASE_EMERGENCY_CORRIDOR`, overriding Layer 5 decisions.
- Approved PR-4 Helper Function:
  - Implement `corridor_schedule(G: nx.Graph, corridor: List[str]) -> List[dict]` returning hop ETAs, preemption windows ($15\text{s}$ lead time), and restore times ($10\text{s}$ lag time).

---

### DEFINITION OF DONE & NUMERIC ACCEPTANCE CRITERIA
1. All 5 scenario events execute end-to-end without exceptions.
2. `apply_event` never mutates input `grid_state`.
3. `validate_triage(triage_result)` passes for all scenarios.
4. `critical_nodes` length is always $\le 6$ and ordered by `NODE_IDS`.
5. Coupling keys strictly follow `NODE_IDS.index(a) < NODE_IDS.index(b)`.
6. Dijkstra routing handles `Intersection_1` as source, distant nodes, disconnected edges, and multiple emergency sources safely.
7. All contract tests (`pytest -q`) pass 100% green. Add unit tests for your layer in your section of `tests/test_contracts.py`.

Execute the implementation now, run `pytest -q`, and present a detailed walkthrough of your results.
```

---

### PROMPT FOR MEMBER 3 (M3 — Quantum Core & Metrics Model Engineer)

```markdown
You are the Lead Quantum Core and Metrics Model Engineer (Member 3) for QureX ("Where Quantum meets the Road."), a hybrid quantum-classical urban traffic optimization platform.

Read `@docs/QUREX_MASTER_PROMPT.md` and `@docs/IMPLEMENTATION_PLAN.md` completely. Treat all frozen contracts and rules as binding.

### YOUR ROLE & WORKSPACE
- Git Branch: `m3-quantum` (Run `git branch --show-current` first; if not on `m3-quantum`, create/checkout `git checkout -b m3-quantum`).
- Files Owned: `quantum_layer.py`, `classical_solver.py`, `metrics.py`
- Shared/Frozen Files (DO NOT MODIFY): `config.py`, `contracts.py`
- Other Members' Files (DO NOT MODIFY): `stats_layer.py`, `forecast_layer.py`, `scenario.py`, `graph_layer.py`, `routing_layer.py`, `copilot_layer.py`, `pipeline.py`, `app.py`.

---

### TECHNICAL SPECIFICATION & FORMULAS TO IMPLEMENT

#### 1. Layer 5 — Quantum Core CVaR-QAOA (`quantum_layer.py`) & Brute-Force Oracle (`classical_solver.py`)
- Decision Variables: $x_i \in \{0,1\}$ per critical node ($1 = \text{PHASE\_MAX\_GREEN}$, $0 = \text{PHASE\_ADAPTIVE\_SHORT}$). Wire $i \leftrightarrow \text{critical\_nodes}[i]$.
- Cost Function:
  $C(x) = - \sum_i w_i x_i + \sum_{(i,j) \in E_c} J_{ij} x_i x_j + \lambda \left(\sum_i x_i - K\right)^2$
  where $K = \max(1, \lceil 0.5 n \rceil)$, $\lambda = 0.5 \cdot \text{mean}(w)$.
- QUBO Expansion ($x_i^2 = x_i$):
  Linear: $a_i = -w_i + \lambda(1 - 2K)$
  Quadratic: $Q_{ij} = 2\lambda + J_{ij}$
  Constant: $\lambda K^2$
- Ising Mapping ($x_i = (1 - Z_i)/2$):
  $h_i = - \frac{a_i}{2} - \frac{1}{4} \sum_{j \neq i} Q_{ij}$
  $J'_{ij} = \frac{Q_{ij}}{4}$
  $H_C = \sum_i h_i Z_i + \sum_{i<j} J'_{ij} Z_i Z_j + \text{const}$
- Numerical Equivalence Assertion:
  Must include test asserting $\text{QUBO}(x) \equiv H_C(Z) \equiv C(x)$ on all $2^n$ bitstrings.
- QAOA Circuit ($p=2$ layers):
  - State prep: $H^{\otimes n} |0\rangle$.
  - Cost unitary $U_C(\gamma_\ell)$: `RZ(2 * gamma * h[i], wires=i)` and `IsingZZ(2 * gamma * J_prime[i,j], wires=[i,j])`.
  - Mixer unitary $U_B(\beta_\ell)$: `RX(2 * beta, wires=i)` (scalar angles).
  - Use `pennylane.device("default.qubit", wires=n)`.
- CVaR Objective & COBYLA Optimizer:
  - Sample $S = 1024$ bitstrings; sort energies ascending.
  - $\text{CVaR}_\alpha(\theta) = \text{mean of lowest } \lceil \alpha S \rceil \text{ energies}$ ($\alpha = 0.2$).
  - Optimize using `scipy.optimize.minimize(method="COBYLA", maxiter=80)` with 3 seeded restarts.
- Elite Bitstring & Output:
  - Sample 2048 bitstrings at optimal $\theta$; select minimum energy sample.
  - Map bit $1 \to \text{PHASE\_MAX\_GREEN}$, $0 \to \text{PHASE\_ADAPTIVE\_SHORT}$.
  - Return `Phases` dictionary `{critical_node: PHASE_CODE}`.
- `classical_solver.py`:
  - Brute-force solver evaluating all $2^n$ states ($n \le 6$) as exact oracle.
- Approved PR-5 Function:
  - Implement `optimize_with_diagnostics(triage: TriageResult) -> Tuple[Phases, dict]` returning approximation ratio $r$, optimal probability $P(\text{optimum})$, and energy convergence curve.

#### 2. Metrics — Queue Model & Environmental Analysis (`metrics.py`)
- Function Signature: `compute_kpis(grid_state: GridState, phases: Phases, triage: TriageResult, corridor: List[str]) -> KPIResult`
- Discrete-Time Simulation ($dt=1\text{ s}$, horizon $T=300\text{ s}$):
  - Inflow rates: $\lambda_m = 0.8 \mu_{\text{fixed}} + (\hat{q} - q)/T$, $\lambda_c = 0.75 \lambda_m$.
  - Main approach queue: $q_m(t+1) = \max(0, q_m + \lambda_m dt - \min(q_m, s g_i \eta dt))$.
  - Cross approach queue: $q_c(t+1) = \max(0, q_c + \lambda_c dt - \min(q_c, s (1-g_i) \eta dt))$.
  - Parameters: $s = 0.5\text{ veh/s}$, $\eta = 0.9$, $\mu_{\text{fixed}} = s \cdot 0.5 \cdot \eta$.
- Controllers:
  - **Fixed-time:** $g = 0.5$.
  - **Actuated:** $g = \text{clip}\left(\frac{\lambda_m}{\lambda_m + \lambda_c}, 0.35, 0.65\right)$.
  - **QureX:** MAX_GREEN $\to g = 0.65$; ADAPTIVE_SHORT $\to$ actuated split; EMERGENCY_CORRIDOR $\to g = 1.0$ (hold window).
- Derived KPIs:
  - Total Delay: $\sum_t (q_m + q_c) \cdot dt$.
  - Fuel Saved (gal): $(\text{delay}_{\text{fixed}} - \text{delay}_{\text{qurex}}) / 3600 \cdot 0.3$ ($0.3\text{ gal/h}$ idling assumption).
  - $\text{CO}_2$ Saved (kg): $\text{fuel\_saved\_gal} \cdot 8.887$.
  - Delay Saved %: $100 \cdot (\text{delay}_{\text{fixed}} - \text{delay}_{\text{qurex}}) / \text{delay}_{\text{fixed}}$.

---

### DEFINITION OF DONE & NUMERIC ACCEPTANCE CRITERIA
1. $\text{QUBO}(x) \equiv H_C(Z)$ assertion passes strictly across all states.
2. QAOA elite bitstring matches exact brute-force optimum in $\ge 95\%$ of $\ge 100$ random test instances ($n \le 6$).
3. Quantum optimization solve time $< 5.0\text{ s}$ per call.
4. KPIs change dynamically based on phases (no hard-coded multipliers).
5. Approved PR-3: `KPIResult` contains optional emergency travel time fields.
6. All contract tests (`pytest -q`) pass 100% green. Add unit tests for your layer in your section of `tests/test_contracts.py`.

Execute the implementation now, run `pytest -q`, and present a detailed walkthrough of your results.
```

---

### PROMPT FOR MEMBER 4 (M4 — UI Dashboard, Co-pilot & Integration Engineer)

```markdown
You are the Lead UI Dashboard, Co-pilot, and Integration Engineer (Member 4) for QureX ("Where Quantum meets the Road."), a hybrid quantum-classical urban traffic optimization platform.

Read `@docs/QUREX_MASTER_PROMPT.md` and `@docs/IMPLEMENTATION_PLAN.md` completely. Treat all frozen contracts and rules as binding.

### YOUR ROLE & WORKSPACE
- Git Branch: `m4-ui` (Run `git branch --show-current` first; if not on `m4-ui`, create/checkout `git checkout -b m4-ui`).
- Files Owned: `app.py`, `pipeline.py`, `copilot_layer.py`, `ui_components.py`, `requirements*.txt`, `README.md`, `docs/`
- Shared/Frozen Files (DO NOT MODIFY): `config.py`, `contracts.py`
- Other Members' Files (DO NOT MODIFY): `stats_layer.py`, `forecast_layer.py`, `scenario.py`, `graph_layer.py`, `routing_layer.py`, `quantum_layer.py`, `classical_solver.py`, `metrics.py`.

---

### TECHNICAL SPECIFICATION & FORMULAS TO IMPLEMENT

#### 1. Layer 6b — Operator Co-pilot (`copilot_layer.py`)
- Function Signature: `summarize(grid_state: GridState, phases: Phases, corridor: List[str], stats: StatsResult) -> str`
- OpenAI Client Configuration:
  - Use `openai.OpenAI(base_url="https://api.featherless.ai/v1", api_key=key)`.
  - Fetch key using `try: st.secrets["FEATHERLESS_API_KEY"] except: None`.
  - Prompt instructions: Pass structured facts (anomalies, critical nodes, phases, corridor length, KPI savings). Instruct LLM to use ONLY provided facts, $\le 2$ sentences, formal tone.
- Fallback System:
  - If API key missing or network call fails, return deterministic template narrative tagged `[LOCAL AI CO-PILOT - DETERMINISTIC MODE]`.

#### 2. Pipeline Integration (`pipeline.py`)
- Structure:
  - `run_pipeline(grid_state: GridState, G: nx.Graph) -> PipelineOutput`
  - Orchestrates call order: `state` $\to$ `statistics` $\to$ `forecast` $\to$ `triage` $\to$ `optimize` $\to$ `emergency override` $\to$ `KPIs + co-pilot`.

#### 3. Dashboard UI (`app.py` & `ui_components.py`)
- Rebranding & Styling:
  - Header: `🚦 QureX` + Tagline: **"Where Quantum meets the Road."**
  - Prominent badge: **"Simulated data"** visible on UI.
- Required Dashboard Panels:
  1. **Incident Console Sidebar:** Target intersection selectbox, event code selectbox, "Deploy Event" button, and "Reset Grid" button.
  2. **Network Topology Graph:** Render Matplotlib grid graph ($2 \times 3$), node color mapped to utilization $u_i$, emergency corridor highlighted in red/bold, phase labels on nodes. Use `plt.close(fig)` after rendering.
  3. **Per-Node Status Table:** Intersection, Event label, Queue length, 5-min forecast, Signal strategy label.
  4. **Emergency Corridor Panel:** Route breakdown and ETA per hop when emergency event active.
  5. **KPI Metrics & Bar Chart:** Grouped bar chart comparing Fixed-time vs Actuated vs QureX delays, throughput, fuel saved (gal), and $\text{CO}_2$ reduced (kg).
  6. **Statistical Diagnostics:** Mahalanobis $D$ vs threshold badge, PCA scree plot showing real $k$ and variance retained.
  7. **QAOA Diagnostics Expander (Approved PR-5):** Convergence curve, top bitstrings, approximation ratio $r$.
  8. **Co-pilot Summary Box:** Display narrative string from Layer 6b.
- Rules:
  - Use `@st.cache_resource` for graph and heavy objects.
  - Use `width="stretch"` (not deprecated `use_container_width`).
  - Warm pipeline rerun latency $< 2.0\text{ s}$.

---

### DEFINITION OF DONE & NUMERIC ACCEPTANCE CRITERIA
1. Clean dashboard execution on fresh clone with NO secrets file (graceful deterministic fallback).
2. All 8 required UI panels present and responsive across all 5 scenario events.
3. Warm pipeline execution latency $< 2.0\text{ s}$.
4. `streamlit.testing.v1.AppTest` passes with 0 exceptions.
5. All contract tests (`pytest -q`) pass 100% green. Add unit tests for your layer in your section of `tests/test_contracts.py`.

Execute the implementation now, run `pytest -q`, run `AppTest`, and present a detailed walkthrough of your results.
```
