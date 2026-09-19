import os
import sys
import time
import json
from dataclasses import asdict

from config import NODE_IDS, DEFAULT_CAPACITY, EVENT_NORMAL
from scenario import default_grid_state
from pipeline import run_pipeline

try:
    import traci
    HAS_TRACI = True
except ImportError:
    HAS_TRACI = False
    
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class MockTraCI:
    """Mock TraCI for when SUMO binaries fail to run on Windows."""
    def __init__(self):
        self.step = 0
        self.queues = {n: 12.0 for n in NODE_IDS}
        
        class Vehicle:
            def __init__(self):
                self.colors = {}
            def getIDList(self):
                return []
            def setSpeed(self, v, s):
                pass
            def setColor(self, v, c):
                self.colors[v] = c
            def getColor(self, v):
                return self.colors.get(v, (255, 255, 0, 255))
            def setShapeClass(self, v, c):
                pass
            def setWidth(self, v, w):
                pass
            def setLength(self, v, l):
                pass
            def setSpeedMode(self, v, m):
                pass
        
        class TrafficLight:
            def getIDList(self):
                return NODE_IDS
            def getControlledLanes(self, tls_id):
                return [f"{tls_id}_lane_1", f"{tls_id}_lane_2"]
            def setPhase(self, tls_id, phase_idx):
                pass
            def getPhase(self, tls_id):
                return 0
            def setProgram(self, tls_id, programID):
                pass
            def getProgram(self, tls_id):
                return "0"
                
        class Lane:
            def getLastStepHaltingNumber(self, lane_id):
                node = lane_id.split("_lane")[0]
                return self.queues.get(node, 12.0) / 2.0
                
            def __init__(self, parent):
                self.queues = parent.queues
                
        class Simulation:
            def __init__(self, parent):
                self.parent = parent
            def getMinExpectedNumber(self):
                return 1000 - self.parent.step
                
        class GUI:
            def setBackgroundColor(self, viewID, color):
                pass
            def trackVehicle(self, viewID, vehID):
                pass
            def setZoom(self, viewID, zoom):
                pass
                
        self.trafficlight = TrafficLight()
        self.lane = Lane(self)
        self.simulation = Simulation(self)
        self.gui = GUI()

    def start(self, cmd, label="default"):
        print(f"[MOCK] Starting SUMO with cmd: {cmd} label={label}")
        
    def simulationStep(self):
        self.step += 1
        self.simulation.step = self.step
        
    def close(self):
        print("[MOCK] Closing SUMO")

    def getConnection(self, label):
        return self

def start_sumo(gui=True):
    """Start two SUMO simulations: Classical and Quantum."""
    if not HAS_TRACI:
        print("TraCI not installed. Falling back to Mock.")
        engine = MockTraCI()
        return engine, engine
        
    sumo_binary = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe" if gui else r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"
    if not os.path.exists(sumo_binary):
        sumo_binary = "sumo-gui" if gui else "sumo"
    cfg_path = os.path.join("sumo", "qurex.sumocfg")
    
    try:
        traci.start([sumo_binary, "-c", cfg_path], label="classical")
        traci.start([sumo_binary, "-c", cfg_path], label="quantum")
        
        conn_classical = traci.getConnection("classical")
        conn_quantum = traci.getConnection("quantum")
        return conn_classical, conn_quantum
    except Exception as e:
        print(f"Failed to start SUMO binaries: {e}")
        engine = MockTraCI()
        return engine, engine

def get_sumo_state(conn):
    """Extract grid state from a specific running SUMO simulation connection."""
    state = default_grid_state()
    tls_ids = conn.trafficlight.getIDList()
    
    for i, node in enumerate(NODE_IDS):
        if i < len(tls_ids):
            tls_id = tls_ids[i]
            lanes = conn.trafficlight.getControlledLanes(tls_id)
            queue = 0
            for lane in set(lanes):
                queue += conn.lane.getLastStepHaltingNumber(lane)
                
            state[node]["queue"] = min(float(queue), float(DEFAULT_CAPACITY))
            state[node]["occupancy"] = min(queue * 2.0, 100.0)
            state[node]["avg_speed"] = max(40.0 - queue, 5.0)
            
    return state

def color_all_cars(conn, target_color):
    """Colors all normal cars while preserving the red accident and white ambulance."""
    try:
        for veh in conn.vehicle.getIDList():
            c = conn.vehicle.getColor(veh)
            if c not in [(255, 0, 0, 255), (255, 255, 255, 255)]:
                conn.vehicle.setColor(veh, target_color)
    except:
        pass

def cause_accident(conn, step):
    """Dynamically stops a car to simulate an accident in 3D."""
    veh_ids = conn.vehicle.getIDList()
    if veh_ids:
        veh = veh_ids[0]
        try:
            conn.vehicle.setSpeed(veh, 0.0)
            conn.vehicle.setColor(veh, (255, 0, 0, 255))
            conn.vehicle.setWidth(veh, 3.0) # Make it massive!
            conn.vehicle.setLength(veh, 10.0)
            conn.gui.trackVehicle("View #0", veh)
            conn.gui.setZoom("View #0", 3000)
            print(f"\\n💥 [EVENT] Accident occurred (Car {veh} broken down) at step {step}!")
        except:
            pass

def turn_car_into_ambulance(conn, step):
    """Dynamically converts an existing car into an ambulance in 3D."""
    veh_ids = conn.vehicle.getIDList()
    if veh_ids:
        veh = veh_ids[-1]
        try:
            conn.vehicle.setColor(veh, (255, 255, 255, 255))
            conn.vehicle.setShapeClass(veh, "emergency")
            conn.vehicle.setWidth(veh, 3.0) # Make it massive!
            conn.vehicle.setLength(veh, 12.0)
            # Force it to ignore speed limits
            conn.vehicle.setSpeedMode(veh, 0)
            conn.vehicle.setSpeed(veh, 30.0)
            conn.gui.trackVehicle("View #0", veh)
            conn.gui.setZoom("View #0", 1500)
            print(f"\\n🚑 [EVENT] Ambulance deployed (Car {veh}) at step {step}!")
        except:
            pass

def apply_phases_to_sumo(conn, phases, step):
    """Push QureX optimized phases back to a specific SUMO traffic lights connection."""
    tls_ids = conn.trafficlight.getIDList()
    
    from config import PHASE_MAX_GREEN, PHASE_EMERGENCY_CORRIDOR
    
    for i, node in enumerate(NODE_IDS):
        if i < len(tls_ids):
            tls_id = tls_ids[i]
            phase_decision = phases.get(node)
            
            if phase_decision in [PHASE_MAX_GREEN, PHASE_EMERGENCY_CORRIDOR]:
                # The Quantum AI takes over the traffic light.
                # We toggle it every 300 steps (30 seconds) to give the lanes a massive green wave to clear out!
                if step % 300 == 0:
                    try:
                        current = conn.trafficlight.getPhase(tls_id)
                        # Toggle between Green N/S (0) and Green E/W (2)
                        new_p = 2 if current == 0 else 0
                        conn.trafficlight.setPhase(tls_id, new_p)
                    except:
                        pass
            else:
                if conn.trafficlight.getProgram(tls_id) != "0":
                    conn.trafficlight.setProgram(tls_id, "0")

def run_loop(gui=True):
    """Main control loop."""
    conn_classical, conn_quantum = start_sumo(gui=gui)
    step = 0
    
    from graph_layer import build_city_graph
    G = build_city_graph()
    
    try:
        while conn_quantum.simulation.getMinExpectedNumber() > 0:
            conn_classical.simulationStep()
            conn_quantum.simulationStep()
            step += 1
            
            if step == 200:
                cause_accident(conn_quantum, step)
                cause_accident(conn_classical, step)
                
            if step == 400:
                turn_car_into_ambulance(conn_quantum, step)
                turn_car_into_ambulance(conn_classical, step)
                
            if step % 10 == 0:
                print(f"\\n--- SUMO Simulation Step {step} ---")
                
                # Violet for Classical, Yellow for Quantum
                color_all_cars(conn_classical, (148, 0, 211, 255))
                color_all_cars(conn_quantum, (255, 255, 0, 255))
                
                # Get states from both
                state_classical = get_sumo_state(conn_classical)
                state_quantum = get_sumo_state(conn_quantum)
                
                # Run pipeline ONLY on quantum state
                out = run_pipeline(state_quantum, G)
                
                # Fast KPI calculation for classical (Bypasses QAOA entirely!)
                from contracts import TriageResult
                import metrics
                c_kpis = metrics.compute_kpis(state_classical, {}, TriageResult([], {}, {}), [])
                
                print(f"Anomaly Detected: {out.stats.is_anomaly}")
                if out.stats.is_anomaly:
                    print(f"Critical Nodes Triaged: {out.triage.critical_nodes}")
                
                print("Quantum QAOA Optimized Phases:")
                for node, phase in out.phases.items():
                    print(f"  {node}: {phase}")
                    
                # 4. Apply QureX output back to the Quantum SUMO instance
                apply_phases_to_sumo(conn_quantum, out.phases, step)
                
                # Export live data for Streamlit
                export_data = {
                    "step": step,
                    "classical_kpis": asdict(c_kpis),
                    "quantum_kpis": out.kpis.__dict__,
                    "quantum_state": state_quantum,
                    "quantum_phases": out.phases,
                    "quantum_corridor": out.corridor,
                    "quantum_forecast": out.forecast,
                    "narrative": out.narrative
                }
                
                with open("live_sumo_state.json", "w") as f:
                    json.dump(export_data, f, cls=NumpyEncoder)
                    
            time.sleep(0.05)
            
    finally:
        try:
            conn_classical.close()
            conn_quantum.close()
        except:
            pass

if __name__ == "__main__":
    run_loop()
