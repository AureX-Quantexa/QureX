# QureX — Master Prompt for Google Antigravity
**Tagline: "Where Quantum meets the Road."**

> **HUMAN, READ FIRST (delete this box before pasting if you like)**
> 1. Put this file at `<project-root>/docs/QUREX_MASTER_PROMPT.md` (project root = the folder that contains `config.py`, inside your `AureX` folder).
> 2. Put the two files from `reference/` into `<project-root>/reference/`.
> 3. Open the `AureX` folder in Antigravity, set the agent to **Planning mode**, and paste the block under **"PASTE THIS"** (the short version), or paste this entire file.
> 4. The agent works in **STEPS 1 → 4**. It will STOP after Step 1 and Step 2 and wait for you. Reply `CONTINUE` (or your corrections).
> 5. Only after Step 4 do you (and each teammate) start the **BUILD phase** using the per-member prompts in Part 7.

---

## PASTE THIS (short kickoff prompt)

```
You are the lead engineer agent for QureX ("Where Quantum meets the Road."), a hybrid
quantum-classical urban traffic optimization platform. Read @docs/QUREX_MASTER_PROMPT.md
COMPLETELY (all parts) before doing anything, and treat it as binding.
Then execute STEP 1 only, and STOP for my reply. Do not write or modify project source files
in Steps 1-4. Work in Planning mode and produce artifacts (task list, implementation plan).
```

---

# PART 1 — MISSION AND ROLE

You are the **lead engineer agent** for **QureX**, a 4-person hackathon project. Four teammates (M1–M4) will each run their own Antigravity agent on their own git branch, so everything you produce must be precise enough that four different agents can work in parallel and merge **without integration errors**.

The repo was generated from a starter kit: frozen contracts (`config.py`, `contracts.py`), one stub module per layer, `pipeline.py`, a Streamlit `app.py`, and `tests/test_contracts.py`. The stubs already run end-to-end. The product is a **6-layer hybrid pipeline** (Part 3) that must **not be redesigned**.

Branding: **Project name QureX. Tagline "Where Quantum meets the Road."** The starter code says "QE-AUTOP" in titles and strings; that is replaced by QureX (UI text, comments, README) during BUILD. The parent folder is named `AureX`; **do not rename or move folders** (imports are flat: `from config import ...`). Locate the real project root yourself (the folder containing `config.py`). If the `AureX` vs `QureX` naming looks like a typo, mention it in one line and ask; do not act on it.

---

# PART 2 — NON-NEGOTIABLE RULES

1. **Do not change the 6 layers, their order, their responsibilities, or their logic** without my explicit permission. This includes `config.py`, `contracts.py`, function signatures, and file ownership. To request a change, use the **PERMISSION REQUEST** format in Appendix A and wait for my answer.
2. **Pre-approved implementation corrections (P1–P12, Appendix B)** are allowed. They implement the *intended* logic correctly; they do not redesign it. Anything not listed there needs permission.
3. **Never claim something works unless you ran it.** Show the command and the real output. If something fails, say so plainly.
4. **Honesty about the science.** All traffic data is *simulated*; label it "Simulated" in the UI. Do not hard-code improvement percentages. Do not claim proven "quantum advantage": with ≤6 qubits a classical brute-force solver is exact and faster, so it is used as the **correctness oracle**; the quantum part is a validated hybrid formulation that maps directly onto larger networks. The UI must say "Hybrid Quantum-Classical vs Baseline", not "Quantum Advantage".
5. **One owner per file** (Part 6). In BUILD, only edit files owned by the role I declare.
6. **Determinism:** seed everything from `config.RANDOM_SEED`. Cache expensive things (model fits, QAOA runs) so a Streamlit rerun is fast.
7. **Secrets:** never commit or print API keys. `.streamlit/secrets.toml` is git-ignored.
8. **Safety:** the LLM co-pilot only *narrates*; it never controls signals. Signals come only from Layers 4–6 logic.
9. **Terminal etiquette:** run commands only inside the project root; create/use a `.venv`; never delete or move files outside `scratch/` or `.venv`; ask before installing anything not in `requirements.txt`. Web/doc lookups are allowed for verifying APIs and constants.
10. **Cross-platform:** code must run on Windows, macOS and Linux with Python 3.10–3.12 (use `pathlib`, no shell-specific code).
11. **Layer numbering used everywhere (code comments, UI, docs):** L1 ingestion, L2 statistical sanitization, L3 forecasting, L4 graph triage, L5 quantum core, L6 enforcement + operator co-pilot.

---

# PART 3 — THE 6 LAYERS (FROZEN)

```
L1  URBAN GRID INGESTION            scenario.py                  -> grid_state
L2  STATISTICAL SANITIZATION (PCA)  stats_layer.py               -> StatsResult (Mahalanobis + eigen-features)
L3  PREDICTIVE WAVE FORECAST (XGB)  forecast_layer.py            -> np.ndarray (6,) 5-min queue forecast
L4  GRAPH FILTRATION TRIAGE         graph_layer.py               -> TriageResult (critical nodes, QUBO weights, couplings)
L5  QUANTUM CORE (CVaR-QAOA)        quantum_layer.py             -> {critical node: phase code}
L6  ENFORCEMENT + OPERATOR CO-PILOT routing_layer.py, copilot_layer.py  -> corridor override + narrative
    (metrics.py, classical_solver.py, pipeline.py, app.py support the layers)
```

Call order (in `pipeline.py`): state → statistics → forecast → triage → optimize → emergency override → KPIs + co-pilot → render.

Frozen interface (see `contracts.py`): `run_statistics(grid_state)->StatsResult`; `forecast_inflows(stats, grid_state)->ndarray(6,)`; `build_city_graph()->nx.Graph`; `triage(grid_state, forecast, G)->TriageResult`; `optimize(triage)->{node: phase}`; `apply_emergency_override(grid_state, G, phases)->(phases, corridor)`; `compute_kpis(grid_state, phases, triage, corridor)->KPIResult`; `summarize(grid_state, phases, corridor, stats)->str`.

---

# PART 4 — TECHNICAL SPECIFICATION (formulas and techniques)

Constants marked **(config)** live in `config.py`; constants marked **(local)** live at the top of the owning module. Verified reference scripts: `reference/stats_reference.py` and `reference/qaoa_reference.py` (tested with PennyLane 0.45.1, XGBoost 3.4.1, scikit-learn 1.8, SciPy 1.17, NumPy 2.4). **Adapt, do not blindly copy.**

## L1 — Ingestion and scenario simulator (M2, `scenario.py`)
Telemetry per node: `queue` (veh), `capacity` (veh of link storage), `occupancy` (%), `avg_speed` (km/h), `event` (code). Raw vector layout is defined **only** by `contracts.flatten_grid_state` (18 values, `NODE_IDS` order).

Event presets `(queue, capacity, occupancy, speed)`: NORMAL (12,50,25,42) · CONGESTION (45,50,90,6) · ACCIDENT (38,15,95,2) · EMERGENCY (24,50,60,22) · FESTIVAL all nodes (40,50,82,12).

Consistency sanity check (Greenshields): speed ≈ v_free·(1 − occupancy/100). Presets should roughly respect it.

## L2 — Statistical sanitization (M1, `stats_layer.py`)
**Historical baseline (simulated):** N = 2000 rows × 18 features from a correlated latent-factor generator: a city-wide demand factor `D~N(0,1)` and per-node factor `e_i~N(0,1)`; `z_q = 0.6·D + 0.8·e_i`; `queue = 15 + 3·z_q`; `occ = 30 + 8·(0.85·z_q + 0.527·ε₁)`; `speed = 40 − 5·(0.8·z_occ + 0.6·ε₂)`. Queue↑ ⇒ occupancy↑ ⇒ speed↓, so PCA has real structure to compress.

**Covariance with shrinkage:** `Σ = LedoitWolf(X).covariance_`; `μ = mean(X)`. (Plain `np.cov` + `pinv` is unstable when features are collinear.)

**Mahalanobis distance:**
```
D²(x) = (x − μ)ᵀ Σ⁻¹ (x − μ)          D(x) = √D²
```
Under approximate normality `D² ~ χ²_p` with p = 18. **Anomaly rule:** `D² > χ²_{p, 1−α}` with α = `ANOMALY_ALPHA` = 0.001 ⇒ `χ² ≈ 42.31`, so the D-threshold ≈ **6.50** (α=0.01 ⇒ 5.90; α=0.05 ⇒ 5.37). Compute with `scipy.stats.chi2.ppf`. Do **not** use fixed thresholds like 3.2/3.5: in 18 dimensions ordinary traffic sits near √18 ≈ 4.2 and would be flagged constantly. `StatsResult.mahalanobis = D`, `.threshold = √χ²`.

Expected behavior on the reference simulator (verify and report your own numbers): default state not anomalous; CONGESTION, ACCIDENT and FESTIVAL flagged; EMERGENCY **not** flagged (an ambulance is not a network fault: its variables move *along* the learned correlation).

**PCA:** standardize features (`StandardScaler`), then `PCA(n_components=0.95)`. Covariance eigen-decomposition `Σ = VΛVᵀ`; keep the smallest k with `Σ_{i≤k} λ_i / Σ λ_i ≥ 0.95`; feature vector `z = V_kᵀ (x − μ)/σ`. **Report the real k and real `explained_variance_ratio_.sum()`** (the reference gives k = 10 unstandardized, *not* 6; the UI must never hard-code "6 vectors / 95%"). `transform` returns 2-D: return `[0]` so `eigen_features` is **1-D** (a double-wrapped list makes a 3-D array and XGBoost rejects it).

Fit once (module-level lazy singleton), cache it, and never refit per Streamlit rerun.

## L3 — Predictive wave forecasting (M1, `forecast_layer.py`)
Model: `xgboost.XGBRegressor` with native multi-output (2-D `y`, 6 targets), `tree_method="hist"`, `n_estimators≈200`, `max_depth=3`, `learning_rate≈0.08`, `subsample≈0.9`, `random_state=RANDOM_SEED`, early stopping on a validation split. Input = `stats.eigen_features` (length k). Target = queue 5 minutes ahead, generated from the queue-balance dynamics
```
q_{t+T} = clip( q_t + (λ_i − μ_i)·T , 0 , C_i ) + ε,   T = 300 s,  ε ~ N(0, 1)
```
where `λ_i` is arrival rate and `μ_i = s·g·η` is service rate (s = 0.5 veh/s = 1800 veh/h, g = green ratio, η = lost-time efficiency ≈ 0.9). Report **MAE and RMSE on a hold-out set versus a persistence baseline** ("queue stays the same"); XGBoost must beat persistence or you must report that it does not. Output: `ndarray` shape `(6,)`, finite, clipped to `[0, capacity_i]`, `NODE_IDS` order. Label as trained on simulated data.

## L4 — Graph triage (M2, `graph_layer.py`)
Graph: 2×3 grid, row-major, explicit node order, edge attribute `travel_cost` = 20 s.

```
utilization    u_i = q̂_i / C_i                (q̂ = forecast)
critical       u_i > 0.60   OR   event_i ≠ NORMAL
urgency        w_i = q_i / C_i, then normalise so max_i w_i = 1
coupling       J_ij = κ · (w_i + w_j)/2 · (τ_ref/τ_ij),  κ = 0.5   for each edge (i,j) with BOTH ends critical
               (physical meaning: two adjacent intersections both on MAX_GREEN push a wave into each
                other and starve cross streets ⇒ spillback penalty)
```
`critical_nodes` is in `NODE_IDS` order and capped at `MAX_QUBITS` (if over, keep the highest `w_i`). Coupling keys are `(a, b)` with `NODE_IDS.index(a) < NODE_IDS.index(b)`. `qubo_weights` keys equal `critical_nodes`. The bottleneck graph is "highly compressed": only critical nodes and the edges among them reach L5.

## L5 — Quantum core: QUBO → Ising → CVaR-QAOA (M3, `quantum_layer.py`)
Decision variable `x_i ∈ {0,1}` per critical node: **1 = MAX_GREEN, 0 = ADAPTIVE_SHORT** (qubit i ↔ `critical_nodes[i]`).

**QUBO cost (minimize):**
```
C(x) = − Σ_i w_i x_i  +  Σ_{(i,j)∈E_c} J_ij x_i x_j  +  λ (Σ_i x_i − K)²
        reward serving      spillback conflict          budget: at most ~K nodes on max green so
        urgent nodes        between adjacent nodes      cross streets are not starved
K = max(1, ⌈0.5·n⌉),   λ = 0.5 · mean(w)              (local constants; document them)
```
**Expansion** (uses `x² = x`): `λ(Σx − K)² = λ[ (1−2K)Σ x_i + 2Σ_{i<j} x_i x_j + K² ]`, so
```
linear   a_i  = − w_i + λ(1 − 2K)
quad     Q_ij = 2λ + J_ij·[ (i,j) is an edge ]            for all pairs i<j
const    λK²
```
**Ising mapping** `x_i = (1 − Z_i)/2`, `x_i x_j = (1 − Z_i − Z_j + Z_i Z_j)/4`:
```
h_i   = − a_i/2 − (1/4) Σ_{j≠i} Q_ij
J'_ij = Q_ij / 4
H_C   = Σ_i h_i Z_i + Σ_{i<j} J'_ij Z_i Z_j + const        (const = λK² + Σa/2 + ΣQ/4)
```
The reference script **asserts numerically** that QUBO cost, this Ising energy and the direct cost agree on every bitstring. Keep such an assertion in tests.

**QAOA circuit** (p = 2 layers): start `H^{⊗n}|0⟩`; for each layer ℓ: cost `U_C(γ_ℓ)=e^{−iγ_ℓ H_C}` = `RZ(2γ_ℓ h_i)` on each wire then `IsingZZ(2γ_ℓ J'_ij)` on each coupled pair; mixer `U_B(β_ℓ)=e^{−iβ_ℓ ΣX_i}` = `RX(2β_ℓ)` on each wire. Each layer needs **scalar** angles (the original code passed a 2-element array to `Evolution` and `RX`; that is a bug). Do not use the deprecated `qml.Hamiltonian`; use the gate decomposition above (or `qml.dot`/`LinearCombination`). Pin the PennyLane version.

**CVaR objective** (Barkoutsos et al.): sample `S` bitstrings (S=1024) from the exact probabilities using a **seeded** NumPy RNG; energies `E_s`; sort ascending; `CVaR_α(θ) = mean of the lowest ⌈α·S⌉ energies`, α = 0.2. This focuses the optimizer on the best tail rather than the average.

**Classical optimizer:** `scipy.optimize.minimize(method="COBYLA", maxiter≈80)` on `CVaR_α(θ)` over `θ=(γ₁,γ₂,β₁,β₂)`; 3 seeded restarts (optionally initialize with a linear-ramp/TQA schedule); keep the best.

**Elite bitstring:** at optimal θ take 2048 samples, pick the minimum-energy sample (this is the "Elite Optimization Configuration Bitstring"). Map `1→PHASE_MAX_GREEN`, `0→PHASE_ADAPTIVE_SHORT`. Bit index i ↔ wire i ↔ `critical_nodes[i]` (wire 0 is the most significant bit of `format(idx, f"0{n}b")`).

**Validation vs classical (`classical_solver.py`):** brute-force all 2ⁿ states (n ≤ 6 ⇒ ≤ 64). Report *approximation ratio* `r = (E_max − E_found)/(E_max − E_min)`, whether the elite equals the brute-force optimum, and `P(optimum)` vs the uniform baseline `1/2ⁿ`. **Acceptance:** the elite equals the brute-force optimum in ≥ 95% of ≥ 100 random instances (n ≤ 6); otherwise tune (p, α, shots, restarts) and report honestly. Cap `n ≤ MAX_QUBITS`; `n = 0` returns `{}`. Cache by hash of `(critical_nodes, weights, couplings)`; a solve must take **< 5 s**.

`optimize(triage)` keeps the frozen signature (returns only phases). Diagnostics (energies, convergence curve, top bitstrings, r) are exposed through a *separate* function only if I approve **PR-5**.

## L6 — Enforcement and operator co-pilot
**Emergency green corridor (M2, `routing_layer.py`)** — Dijkstra: `nx.shortest_path(G, source, target, weight="travel_cost")`.
- Sources = nodes with `event == EVENT_EMERGENCY`; target = `config.EMERGENCY_TARGET`. `nx.shortest_path` takes **one** source, so loop over sources (or pick the nearest by `nx.shortest_path_length`); a list as `source` errors.
- Edge cases: source == target ⇒ corridor `[target]`; no path (e.g. closed edge) ⇒ corridor `[]` without crashing; multiple emergencies ⇒ ordered union of paths.
- Every corridor node gets `PHASE_EMERGENCY_CORRIDOR`, overriding L5's decision (safety override). ETA at hop k: `t_k = Σ travel_cost` along the path.
- Preemption timing (documented constants): open the green `lead = 15 s` before ETA, hold until `ETA + 10 s`, then **restore normal control**. (A timeline/schedule output needs **PR-4**.)

**Operator co-pilot (M4, `copilot_layer.py`):** OpenAI-compatible client to Featherless: `OpenAI(base_url="https://api.featherless.ai/v1", api_key=key)` (the original omitted the scheme; **verify base URL and an available model id in Featherless docs** or via the models endpoint before hard-coding a model). `timeout≈10 s`, `max_tokens≈120`, `temperature≈0.2`. Feed **structured facts** (event nodes, phases, corridor, D vs threshold, KPI numbers) and instruct: "Use only these facts; do not invent numbers; ≤ 2 sentences; formal." Read the key via `try: st.secrets[...] except: None`, because `st.secrets.get` can raise when no `secrets.toml` exists. If no key, invalid key or any error ⇒ **deterministic template fallback** built from the same facts, clearly tagged "[LOCAL AI CO-PILOT - DETERMINISTIC MODE]".

## Metrics — KPIs from a queue model (M3, `metrics.py`)
KPIs must be **computed from the chosen phases**, never from constants like `0.68 / 0.48` (the original code did that; a judge will ask). Label all outputs "Simulated estimate".

**Model insight you must respect:** with one signal serving two approaches, `dq_main + dq_cross = λ_m + λ_c − s·η` **regardless of the green split** as long as neither queue empties. So a split only helps by (a) not wasting green on an empty approach, and (b) network effects. Therefore include:
1. **Wasted green:** discharge = `min(queue, s·g·η)`.
2. **Downstream spillback:** flow from node i into adjacent node j is limited by j's free storage `C_j − q_j` (this is exactly what the L4/L5 couplings encode).

Discrete-time simulation, `dt = 1 s`, horizon `T = 300 s`, per node one *main* approach (the monitored queue) and one *cross* approach:
```
q_main(t+1) = max(0, q_main + λ_m − min(q_main, s·g_i·η))     (+ forwarded inflow from upstream, limited by free storage)
q_cross(t+1)= max(0, q_cross + λ_c − min(q_cross, s·(1−g_i)·η)),  λ_c = 0.75·λ_m,  q_cross(0) = 0.4·q_main(0)
λ_m = 0.8·μ_fixed + (q̂ − q)/T        (arrival rate consistent with the L3 forecast)
node delay (veh·s) = Σ_t (q_main + q_cross)·dt          s = 0.5 veh/s, η = 0.9, μ_fixed = s·0.5·η
```
Controllers: **Fixed-time** `g = 0.5` · **Actuated (rule-based)** `g = clip(λ_m/(λ_m+λ_c), 0.35, 0.65)` · **QureX** `MAX_GREEN → g = 0.65`; `ADAPTIVE_SHORT →` actuated split; `EMERGENCY_CORRIDOR →` main approach at `g = 1` for the hold window, then restored; `STANDARD_FIXED → 0.5`.

Derived KPIs:
```
throughput (veh/h)  = (3600/T) · Σ vehicles discharged in the window
idle fuel  (gal)    = (veh·s / 3600) · idle_rate        idle_rate ≈ 0.3 gal/h per vehicle (VERIFY against a cited source; state it as an assumption)
CO₂ (kg)            = fuel_gal · 8.887                  (EPA gasoline factor: 8,887 g CO₂ per gallon)
per-hour scaling    = ×(3600/T), labelled "if conditions persist"
delay saved %       = (total_fixed − total_qurex) / total_fixed
```
**Expected honesty:** in a quick probe of a simplified network model of this kind, gains vs fixed-time were single-digit to low-double-digit percent and only marginal vs a good actuated controller; the earlier hard-coded 32%–52% figures were invented. Your model will differ, but report whatever it honestly produces, run a small **sensitivity table** (vary `s`, `κ`, `λ_c/λ_m`), and never tune constants just to win. The **Emergency Green Corridor** travel-time saving (below) is usually the most robust, defensible headline.

**Emergency travel time (needs PR-3 to be shown):**
```
T_base = Σ edge travel_cost + Σ_k W_k ,   W_k = q_k/μ_fixed,k (clear the queue ahead) + 0.5·C·(1 − g)   (C = 90 s cycle)
T_corr = Σ edge travel_cost + Σ_k (residual ≈ 0–3 s)
saving = T_base − T_corr        disruption = extra cross-street veh·s while cross g = 0 for the hold window
```

## Dashboard (M4, `app.py`, `ui_components.py`)
Header: `🚦 QureX` + tagline **"Where Quantum meets the Road."** Sidebar: incident console (target node + event via `EVENT_LABELS`), Deploy, **Reset**. Required panels (map to the problem statement): road-network graph (node colour = utilization `u_i`, corridor highlighted, phase labels on nodes, matplotlib or Plotly; Folium only if PR-8 is approved) · per-node table (event, queue, forecast, current vs optimized signal) · emergency route with ETA per hop · waiting time · fuel and CO₂ · **fixed vs actuated vs QureX** grouped bar chart + totals · statistical diagnostics (D vs threshold, PCA scree with real k and variance) · QAOA diagnostics expander (only after PR-5) · co-pilot narrative · a visible **"Simulated data"** badge. Rules: `st.cache_resource` for graph/models; `width="stretch"` (not `use_container_width`); `plt.close(fig)` after every Matplotlib render; a full pipeline rerun < 2 s after warm-up (spinner for the first fit); no bare `except:`.

## Tests (everyone, `tests/`)
`tests/test_contracts.py` stays green and is never weakened. Each owner adds tests in their own section: M1 (thresholds, PCA k/variance, 1-D features, forecast beats persistence, determinism) · M2 (5 events, festival hits all nodes, coupling key order, Dijkstra edge cases, `apply_event` does not mutate) · M3 (QUBO≡Ising assertion, QAOA vs brute force ≥95%, `n=0`, `n=6`, KPIs change when phases change, no hard-coded multipliers) · M4 (co-pilot fallback with no secrets file, app smoke test with `streamlit.testing.v1.AppTest`).

---

# PART 5 — THE PROBLEM STATEMENT (verbatim; used in Step 4)

**Quantum-Enhanced Adaptive Urban Traffic Optimization.** Urban traffic congestion is a major challenge in growing cities. Traditional traffic signals often use fixed timings and cannot efficiently respond to changing traffic density, accidents, road closures, or emergency vehicles. Poor coordination between nearby intersections can also increase waiting time, fuel consumption, and carbon emissions.

The proposed system aims to develop a **Hybrid Quantum-Classical Traffic Optimization Platform** that dynamically manages traffic signals across multiple interconnected intersections. The system should model traffic flow as an optimization problem and explore techniques such as **QUBO and QAOA** to determine efficient signal timings while considering vehicle density, queue length, road capacity, pedestrian movement, and emergency vehicle priority. A major feature should be an **Emergency Green Corridor System** that reduces ambulance or emergency vehicle travel time while minimizing disruption to normal traffic.

**Core Features:** Multi-Intersection Traffic Network Model (≈4–8 connected intersections with traffic density, queue length, road capacity, signal status) · Quantum Optimization Engine (QUBO/Ising and QAOA or hybrid approach for signal timings) · Adaptive Traffic Signals (adjust green duration to changing conditions, not only fixed timings) · Emergency Green Corridor (prioritize emergency vehicles by dynamically modifying selected signals and **restoring normal traffic afterward**) · Dynamic Event Handling (at least one of: sudden congestion, accident, road closure, emergency arrival) · Environmental Analysis (waiting time, throughput, fuel, CO₂) · Classical Comparison (against fixed timing or rule-based control) · Interactive Dashboard (road network, traffic density, current and optimized signals, emergency route, queue length, waiting time, fuel and CO₂, classical vs quantum results).

**Objectives:** minimize waiting time · minimize queue length · reduce congestion · reduce emergency travel time · reduce fuel and emissions · maximize throughput.

**Suggested stack:** Qiskit/Aer, PennyLane, Python, NetworkX, SUMO or custom simulation, Streamlit, Folium/OpenStreetMap. **Expected outcome:** a working prototype demonstrating how hybrid quantum-classical optimization can improve multi-intersection flow, reduce congestion and emissions, and provide priority routing for emergency vehicles; it must **clearly explain where the quantum component is used and compare with a simple classical baseline.**

---

# PART 6 — ROLES AND FILE OWNERSHIP

| Role | Owns |
|---|---|
| **M1** Stats and forecasting | `stats_layer.py`, `forecast_layer.py` |
| **M2** Network, scenarios, routing | `scenario.py`, `graph_layer.py`, `routing_layer.py` |
| **M3** Quantum and metrics | `quantum_layer.py`, `classical_solver.py`, `metrics.py` |
| **M4** UI and integration | `app.py`, `pipeline.py`, `copilot_layer.py`, `ui_components.py`, `requirements*.txt`, `README.md`, `docs/` |
| **Frozen (all four)** | `config.py`, `contracts.py` |

Git: branches `m1-stats`, `m2-graph`, `m3-quantum`, `m4-ui`; merge to `main` only with `pytest -q` green; merge order M1 → M2 → M3, M4 runs the full app after each merge. If integration reveals a mismatch, the **producer** fixes it to match `contracts.py`.

---

# PART 7 — WORKFLOW (execute in order; obey the STOP gates)

Work in **Planning mode**. In Steps 1–4 you may create only: a `.venv`, a `scratch/` folder, and `docs/*.md` artifacts. **Do not modify any project source file in Steps 1–4.**

## STEP 1 — Orientation and MANUAL TASKS REPORT
1. Find the project root (the folder containing `config.py`) and report its path. Do not move anything.
2. Read every file, then create `.venv`, install `requirements.txt`, run `pytest -q` and `streamlit run app.py` (headless or via `AppTest`), and report the real results, Python version and OS.
3. Write a short **Understanding Report** (≤ 25 lines): confirm the 6 layers ↔ files ↔ contracts mapping; list any place where the repo differs from this prompt.
4. Write the **MANUAL TASKS REPORT**: everything *you cannot do* and a human must do. Group by category (accounts and keys · environment and installs · git and collaboration · decisions and approvals · data and constants to verify · demo and presentation prep). For each task give: **Task · Why an agent cannot do it · Suggested owner (M1–M4) · Needed before which step/phase · Exact how-to steps · How to confirm it is done**. Be specific to what you actually found (for example, installs that failed on this OS). Then, for each member M1–M4, write a **ready-to-copy message** containing only that member's tasks.
5. End Step 1 with at most 5 questions in one message (for example: demo deadline and time, each member's OS and Python version, real-world map location for PR-8, whether all four members have Featherless keys).
6. **STOP.** Wait for me to reply `CONTINUE`.

## STEP 2 — IMPLEMENTATION PLAN
Produce the `implementation_plan.md` artifact and copy it to `docs/IMPLEMENTATION_PLAN.md`. It must contain:
1. Per-member work packages: files, functions, formulas used (from Part 4), inputs/outputs, and a **definition of done with numeric acceptance criteria**.
2. A timeline scaled to my deadline: parallel build, a mid-way integration checkpoint (stubs replaced in the order M1 → M2 → M3, M4 runs the app after each), final testing of all five events, and demo rehearsal.
3. The integration protocol, the branch/merge plan, and the risk register (PennyLane/XGBoost install issues, QAOA runtime, API outage ⇒ deterministic fallback, judges asking about the KPI numbers, and so on).
4. A **gap analysis against the problem statement**. Known gaps in the current design: pedestrian movement; road-closure event; emergency travel-time metric and the restore-normal timeline; green *duration* in seconds (phases are currently binary); a road-network map. For each gap, file a **PERMISSION REQUEST** (Appendix A) using these proposals, each additive and backward-compatible:
   - **PR-1** Add `EVENT_ROAD_CLOSURE` (config) and handle it in `graph_layer`/`routing_layer` by removing edges and rerouting.
   - **PR-2** Pedestrian constraint via per-node config constants: `G_ped = 7 + W/1.2 s` (W = crossing width in m); ADAPTIVE_SHORT may not go below it. No telemetry change.
   - **PR-3** Add optional fields (with defaults) to `KPIResult`: `emergency_time_baseline_s`, `emergency_time_corridor_s`, `disruption_veh_s`.
   - **PR-4** Add `routing_layer.corridor_schedule(G, corridor)` (ETAs, preempt windows, restore times) without changing existing signatures.
   - **PR-5** Add `quantum_layer.optimize_with_diagnostics(triage)` returning `(phases, info)`.
   - **PR-6** Derive green durations in seconds (cycle `C = 90 s`, `g` from phase) for display; new constants in config.
   - **PR-7** Sensor noise/dropout toggle in the L1 console to showcase L2 (`scenario.add_sensor_noise`).
   - **PR-8** Folium/OpenStreetMap map layer (adds `streamlit-folium`; needs real coordinates from the team).
5. Recommendation for each PR (approve / defer) with impact and cost.
6. **STOP** and wait for my approval or edits.

## STEP 3 — FEASIBILITY VERIFICATION ("prove it can work")
In `scratch/feasibility/` (never in project source), run and report **real outputs and timings** for:
1. Clean install from `requirements.txt` in a fresh venv; record the exact installed versions of PennyLane, XGBoost, scikit-learn, SciPy, NumPy, Streamlit, NetworkX.
2. `reference/stats_reference.py`: PCA k and variance retained, D vs threshold for each of the five scenarios, forecast MAE vs persistence.
3. `reference/qaoa_reference.py`: QUBO≡Ising assertion passes, QAOA elite vs brute force, `P(optimum)` vs uniform, and wall-clock time. Then a stress test: 100 random instances with n = 1..6 (report the fraction where the elite equals the optimum, and the worst-case runtime).
4. A prototype of the queue model from Part 4 (metrics): show that KPIs change when phases change; report the honest fixed vs actuated vs hybrid numbers for all five events (this sets expectations for the team).
5. Dijkstra corridor on the 2×3 graph for every possible emergency source, including source == target.
6. Streamlit smoke test with the full pipeline in under 2 s after warm-up.
7. Featherless: check the docs for the base URL and a current model id. Only if a key exists, make one test call; otherwise confirm the deterministic fallback.

Finish with a **Feasibility Verdict**: PASS / PARTIAL / FAIL per item, evidence, and fixes for anything not PASS. If something fails, do not proceed silently. Propose the smallest fix and ask.

## STEP 4 — EXPLAIN THE PROBLEM STATEMENT
Only after Step 3, write `docs/PROBLEM_STATEMENT_EXPLAINED.md` and summarize it in chat:
1. **Plain-language explanation** of the problem and why fixed timers fail.
2. **Traceability matrix:** every requirement/feature/objective in Part 5 → which layer(s) and file(s) address it → status (Done / Planned / Needs PR-x) → how the demo shows it.
3. **Where exactly the quantum component is used** (L5 only: QUBO/Ising + CVaR-QAOA over the triaged bottleneck set), what the classical parts do, why the hybrid split makes sense, and an honest statement of limits (≤ 6 qubits, simulator, brute force is the oracle, no proven advantage claim).
4. **A 2-minute pitch script** and a list of the 8 toughest questions judges may ask, with honest answers (for example: "Are these savings real?", "Why not just brute force?").
Then say "Ready for BUILD" and **STOP**.

---

# PART 8 — BUILD PHASE (run later, one prompt per member, each in their own Antigravity on their own branch)

Common preamble for every member:
```
Read @docs/QUREX_MASTER_PROMPT.md and @docs/IMPLEMENTATION_PLAN.md. I am member <M#> on branch
<branch>. Run `git branch --show-current` first. Edit ONLY my files (Part 6). Keep every function
signature and contract exactly as frozen. Replace stub bodies with real logic per Part 4.
Apply only approved PRs. Run `pytest -q` before you finish. Show me real output. If you need a
change to config.py, contracts.py or another member's file, STOP and write a PERMISSION REQUEST.
Finish with a walkthrough.
```

**M1 — Stats and forecast.** Implement L2 and L3 per Part 4 (LedoitWolf, χ² threshold, standardize + PCA(0.95), real k/variance, 1-D features, cached singleton; XGBoost multi-output, hold-out MAE/RMSE vs persistence, cached). *Done when:* default not anomalous; congestion, accident and festival flagged; emergency not; variance ≥ 0.95; forecast shape (6,), finite, in `[0, capacity]`; beats persistence (or reported); warm call < 50 ms; tests added.

**M2 — Network, scenarios, routing.** Finalize L1 scenario presets (event codes only), `build_city_graph`, `triage` (u_i, w_i, J_ij, MAX_QUBITS cap, key ordering) and `apply_emergency_override` (loop over sources, edge cases, ETA hop list). *Done when:* all five events run; coupling keys ordered; `validate_triage` passes; emergency at Intersection_1, at a far node and with two emergencies all behave; a state is never mutated; tests added.

**M3 — Quantum and metrics.** Implement L5 per Part 4 (QUBO → Ising → CVaR-QAOA → elite bitstring, brute-force oracle, cache, seeds, pinned PennyLane) and `metrics.py` (queue model with wasted green and spillback, three controllers, throughput, fuel, CO₂, sensitivity table, no hard-coded multipliers). *Done when:* QUBO≡Ising asserted; elite = optimum in ≥ 95% of ≥ 100 random instances; solve < 5 s; KPIs respond to phase changes; report honest numbers.

**M4 — UI and integration.** Rebuild `app.py` cleanly as QureX (branding + tagline, all required panels, Reset, Simulated badge), `ui_components.py`, `copilot_layer.py` (Featherless with verified base URL/model, safe secrets, deterministic fallback), `requirements.txt` + lock file, README, demo script. After each merge run the full app and all five events. *Done when:* app runs on a clean clone; works with no secrets file; pipeline rerun < 2 s; all panels present; `AppTest` smoke test passes.

---

# APPENDIX A — PERMISSION REQUEST format
```
PERMISSION REQUEST PR-<n>
What:            <one sentence>
Why:             <problem-statement requirement or bug it addresses>
Files touched:   <list, including frozen files>
Contract impact: <none / additive with defaults / breaking>
Risk:            <low/med/high + why>
Alternative:     <a way to do it without changing anything frozen>
Recommendation:  <approve / defer>
```
Wait for my explicit "approved PR-n" before acting.

# APPENDIX B — Pre-approved implementation corrections (P1–P12)
- **P1** Events are detected by code, never by emoji or display strings.
- **P2** Node order is explicit (`NODE_IDS`), never `G.nodes()` luck.
- **P3** Mahalanobis uses Ledoit-Wolf covariance and a χ² quantile threshold (not fixed 3.2/3.5).
- **P4** PCA uses `n_components=0.95` after standardization; report the real k and variance.
- **P5** `eigen_features` is 1-D (no double wrapping).
- **P6** XGBoost is trained once, cached, native multi-output.
- **P7** QAOA has a real training loop, CVaR objective, `IsingZZ` couplings, scalar angles, no deprecated `qml.Hamiltonian`.
- **P8** Dijkstra loops over sources and handles source == target and no-path.
- **P9** KPIs are derived from a queue model of the chosen phases; fuel is dimensionally correct (veh·s → gal → kg CO₂).
- **P10** Featherless base URL includes scheme; secrets access is wrapped in try/except; deterministic fallback; `plt.close(fig)`; `width="stretch"`.
- **P11** Rebrand UI text/comments/README from QE-AUTOP to QureX with the tagline.
- **P12** Pinned lock file from a verified working install.

Anything not in this list, or any change to the 6 layers or their logic, requires a PERMISSION REQUEST.
