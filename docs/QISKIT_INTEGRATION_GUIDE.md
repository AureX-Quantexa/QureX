# QureX — Qiskit Aer & IBM Quantum Hardware Integration Guide

**Tagline:** "Where Quantum meets the Road."

This guide explains how to execute QureX's Layer 5 CVaR-QAOA quantum circuits on **IBM Qiskit Aer** or **Real IBM Quantum Hardware (QPUs)** using PennyLane's hardware-agnostic device architecture.

---

## 1. Overview & Architecture

In PennyLane, quantum circuits (`@qml.qnode`) are completely decoupled from the underlying hardware execution engine. You can seamlessly switch backends by changing the `qml.device` initialization line:

| Backend Target | Device String | Execution Hardware |
|---|---|---|
| **PennyLane Default** | `qml.device("default.qubit", wires=n)` | Pure Python statevector simulator |
| **Qiskit Aer** | `qml.device("qiskit.aer", wires=n)` | High-performance C++ Qiskit simulator |
| **Real IBM Quantum QPU** | `qml.device("qiskit.remote", wires=n, backend="ibm_brisbane", token=...)` | Real IBM Quantum Hardware via IBM Cloud API |

---

## 2. Step-by-Step Setup Guide

### Step 1: Install `pennylane-qiskit` & `qiskit-aer`
Install the official PennyLane-Qiskit plugin in your Python 3.12 environment:
```bash
pip install pennylane-qiskit qiskit-aer
```

### Step 2: Switch Backend to Qiskit Aer (`quantum_layer.py`)
In `quantum_layer.py`, locate the device definition inside `_solve_cvar_qaoa_cached`:

```python
# --- Option A: PennyLane Default Simulator
# dev = qml.device("default.qubit", wires=n)

# --- Option B: Qiskit Aer Simulator (C++ Engine)
dev = qml.device("qiskit.aer", wires=n)
```

PennyLane automatically translates the QAOA circuit gates (`RZ`, `IsingZZ`, `RX`) into Qiskit `QuantumCircuit` objects and dispatches them to Qiskit Aer.

---

## 3. Connecting to Real IBM Quantum Hardware

To execute QAOA circuits on actual IBM Quantum QPUs (e.g. `ibm_brisbane`, `ibm_kyiv`):

1. Create a free account at [quantum.ibm.com](https://quantum.ibm.com) and copy your API Token.
2. Add your token to `.streamlit/secrets.toml`:
   ```toml
   IBM_QUANTUM_TOKEN = "your_ibm_quantum_api_token_here"
   ```
3. Update `quantum_layer.py`:
   ```python
   import streamlit as st

   try:
       token = st.secrets["IBM_QUANTUM_TOKEN"]
       dev = qml.device(
           "qiskit.remote",
           wires=n,
           backend="ibm_brisbane",
           token=token
       )
   except KeyError:
       # Fallback to Qiskit Aer if token is missing
       dev = qml.device("qiskit.aer", wires=n)
   ```

---

## 4. Verification Test

Run the Qiskit Aer integration verification script:
```bash
python scratch/test_qiskit_device.py
```
**Expected Output:**
```text
Initializing PennyLane device with 'qiskit.aer' backend on 3 qubits...
Execution output from Qiskit Aer backend:
  Counts: {'000': 488, '111': 536}
Qiskit Aer integration test PASSED!
```

---

## 5. Presentation Pitch Script for Judges

When a hackathon judge asks: **"Can this run on Qiskit Aer or real IBM Quantum hardware?"**

> *"Yes! We engineered QureX using PennyLane's hardware-agnostic architecture. By initializing the device as `qml.device('qiskit.aer', wires=n)`, our QAOA optimization circuits are automatically translated into Qiskit quantum circuits and executed on IBM's Aer engine. Furthermore, with `qiskit.remote`, the exact same circuit runs directly on real IBM Quantum QPUs like `ibm_brisbane` without changing a single line of our quantum algorithm!"*
