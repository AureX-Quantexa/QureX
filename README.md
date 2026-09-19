# QE-AUTOP — Team Repo

## Quick start (every member, once)
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                          # must be green
streamlit run app.py               # must open the stub dashboard
```

## Ownership — ONE OWNER PER FILE. Never edit a file you don't own; ask the owner.
| Member | Files |
|---|---|
| **M1** Stats & Forecast | `stats_layer.py`, `forecast_layer.py` |
| **M2** Network & Routing | `scenario.py`, `graph_layer.py`, `routing_layer.py` |
| **M3** Quantum & Metrics | `quantum_layer.py`, `classical_solver.py`, `metrics.py` |
| **M4** UI & Integration | `app.py`, `pipeline.py`, `copilot_layer.py`, `requirements*.txt`, `README.md` |
| **FROZEN (all 4)** | `config.py`, `contracts.py` |

## Rules
1. `config.py` / `contracts.py` are frozen. A change needs a group-chat message + 4 thumbs-up.
2. Keep your function **signatures** exactly as in the stubs. Replace the *body* only.
3. Layers exchange **codes** (`config.py`), never emoji/display strings.
4. Per-node arrays are always in `NODE_IDS` order.
5. Mismatch found at integration → the **producer** fixes it to match `contracts.py`.
6. `pytest -q` green before every push. Add tests for your layer in your own section of `tests/test_contracts.py`.
7. One branch per person: `m1-stats`, `m2-graph`, `m3-quantum`, `m4-ui`. Merge to `main` only when green. Pull `main` often.

## Pipeline order (see `pipeline.py`)
state → statistics → forecast → triage → optimize → emergency override → KPIs + co-pilot → render
