# 🚦 QureX — Where Quantum meets the Road

## 🔗 Project Links & Resources
- 📊 **Pitch Deck (PPT):** [View Presentation](https://drive.google.com/drive/folders/1DZlM6dtM4epmjUq7cuXVPEsM0Ojo4yQP?usp=sharing)
- 🖥️ **Live Interactive Dashboard:** [QureX Streamlit App](https://hwty8gwssm9wxdf8ot7kar.streamlit.app/)
- 🎥 **SUMO Simulation Video:** [Watch Video Demonstration](https://drive.google.com/drive/folders/14NVYG-7kATyQwgaTkXdPFq76vS8FDZtB?usp=sharing)
  > **Video Description:** This video demonstrates the raw 3D Eclipse SUMO simulation running in the background. It showcases the drastic difference between traditional traffic routing and our quantum-optimized system. You can visually observe a massive traffic jam under classical fixed-timers being instantly resolved when the QureX algorithm engages, creating a perfectly synchronized "Green Wave" where cars move fluidly without stopping.

## 📖 The Problem Statement
Urban traffic congestion is a monumental crisis costing cities billions of dollars and millions of metric tons of wasted $\text{CO}_2$ emissions annually. Traditional traffic light systems operate on fixed timers or rudimentary actuated sensors that only look at one intersection at a time. They cannot dynamically coordinate across a grid, they fail to anticipate cascading traffic jams, and they do not provide a unified, prioritized "green wave" for emergency vehicles, leading to critical delays in life-saving scenarios.

## 💡 Our Proposed Solution
**QureX** is a cutting-edge Hybrid Quantum-Classical Urban Traffic Optimization Platform. We have designed a system that replaces isolated traffic timers with a global, intelligent coordinator. Our system ingests live grid data, detects anomalies, forecasts upcoming traffic waves, triages bottlenecks, and uses quantum optimization to perfectly synchronize traffic signals. 

This approach minimizes global network delay, cuts $\text{CO}_2$ emissions, and creates instant emergency corridors.

### 🏛️ The 6-Layer Architecture
Our solution is structured across a crisp, 6-layer pipeline that processes live traffic data in real-time:

1. **Layer 1: Urban Grid Ingestion**  
   *What it does:* Connects to the live traffic simulator (Eclipse SUMO) and pulls the exact queue lengths and capacities of every intersection.
2. **Layer 2: Statistical Sanitization (PCA)**  
   *What it does:* Uses Principal Component Analysis and Mahalanobis distance to instantly detect statistical anomalies, filtering out noisy data and identifying sudden traffic jams.
3. **Layer 3: Predictive Wave Forecast (XGBoost)**  
   *What it does:* Uses an XGBoost machine learning model to look 5 minutes into the future, predicting which intersections will become congested before the traffic wave even hits them.
4. **Layer 4: Graph Filtration Triage**  
   *What it does:* Maps the city as a mathematical graph and applies **dynamic threshold graph clustering** to isolate the most critical "bottleneck" intersections that are causing ripple effects across the grid.
5. **Layer 5: Quantum Core (CVaR-QAOA)**  
   *What it does:* Routes the critical bottlenecks to a Quantum Algorithm (QAOA) that explores thousands of combinatorial traffic light phases simultaneously, finding the mathematically optimal synchronization in milliseconds.
6. **Layer 6: Emergency Priority & COPS AI Assistant**  
   *What it does:* 
   - Uses the **Dijkstra algorithm** to calculate the absolute fastest route and overrides normal signals to instantly carve a "Green Wave" corridor for emergency vehicles.
   - Summarizes the entire system's mathematical outputs using an Interactive AI Assistant (COPS) that explains the traffic data simply and interactively to human operators.

### 🚀 Why is this better?
Instead of reacting to traffic that has already stopped, **QureX** anticipates traffic before it arrives. By utilizing Quantum Optimization (which solves complex routing exponentially faster than classical computers) and intelligent ML forecasting, QureX dramatically reduces vehicle waiting times, saves massive amounts of fuel, and ensures ambulances never hit a red light.

## 📊 Live Simulation Results (Classical vs Quantum)
Based on our real-time simulations, switching from a Classical fixed/actuated routing system to the QureX Quantum-optimized system yielded massive improvements across all Key Performance Indicators:

- **Vehicle Waiting Time (Delay):** Drastically reduced from **~840 seconds** (Classical) down to just **~42 seconds** (Quantum).
- **Traffic Throughput:** Maintained high efficiency, cleanly clearing bottlenecks and maintaining a smooth **~955+ vehicles/hour** throughput across the grid.
- **Fuel Consumption:** The elimination of stop-and-go congestion resulted in fuel savings of over **21+ gallons** per hour globally.
- **CO₂ Emissions:** The fuel efficiency directly prevented over **180+ kg** of CO₂ emissions from entering the atmosphere per hour.

---

## 💻 Setup and Run Instructions

Follow these simple steps to get QureX running on your local machine:

### 1. Prerequisites
- **Python 3.10+** installed on your machine.
- **Eclipse SUMO** traffic simulator installed (and added to your PATH).

### 2. Environment Setup
Clone the repository and install the required dependencies:
```bash
# Clone the repository
git clone https://github.com/AureX-Quantexa/QureX.git
cd QureX

# Create and activate a virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install all required packages
pip install -r requirements.txt
```

### 3. Run the System
To start the live simulation and open the interactive dashboard:

**Step A:** Open a terminal, activate your virtual environment, and run the background SUMO simulator:
```bash
python sumo_connector.py
```
*(Note: Do not click "Play" on the 3D SUMO window, let the Python script control it autonomously.)*

**Step B:** Open a **second** terminal, activate your virtual environment, and launch the Streamlit Dashboard:
```bash
streamlit run app.py
```

Your browser will automatically open the beautiful QureX Control Center!

---

> *"The road to the future is not paved with more asphalt, but with quantum algorithms that perfectly choreograph the dance of a million cars."*  
> — **QureX Team**
