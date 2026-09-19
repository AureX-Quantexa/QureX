"""
FROZEN SHARED CONSTANTS  (all 4 members must approve any change)

Rules:
  * Every per-node array in the project is indexed in NODE_IDS order.
  * Layers exchange CODES (strings below), never emoji or display text.
    Emoji / labels live ONLY in EVENT_LABELS / PHASE_LABELS and are used ONLY by the UI.
"""

# ---------------------------------------------------------------- topology
NODE_IDS = [f"Intersection_{i}" for i in range(1, 7)]
N_NODES = len(NODE_IDS)
GRID_ROWS, GRID_COLS = 2, 3            # 2x3 street grid, row-major:
#   Intersection_1  Intersection_2  Intersection_3
#   Intersection_4  Intersection_5  Intersection_6
EDGE_TRAVEL_COST = 20.0                # seconds, every road segment
EMERGENCY_TARGET = "Intersection_1"    # hospital / control-centre node

NODE_COORDS = {                        # GPS Coordinates for Folium map
    "Intersection_1": (40.7580, -73.9855),
    "Intersection_2": (40.7590, -73.9840),
    "Intersection_3": (40.7600, -73.9825),
    "Intersection_4": (40.7570, -73.9840),
    "Intersection_5": (40.7580, -73.9825),
    "Intersection_6": (40.7590, -73.9810),
}

# ---------------------------------------------------------------- telemetry
# Per-node telemetry keys, in this exact order.
FEATURES = ("queue", "occupancy", "avg_speed")
N_FEATURES = len(FEATURES)
N_RAW_FEATURES = N_NODES * N_FEATURES  # 18
# Flat vector layout (see contracts.flatten_grid_state):
#   [n1.queue, n1.occupancy, n1.avg_speed, n2.queue, n2.occupancy, ...]

DEFAULT_CAPACITY = 50

# ---------------------------------------------------------------- event codes
EVENT_NORMAL = "NORMAL"
EVENT_CONGESTION = "CONGESTION"
EVENT_ACCIDENT = "ACCIDENT"
EVENT_EMERGENCY = "EMERGENCY"
EVENT_FESTIVAL = "FESTIVAL"
EVENT_CODES = (EVENT_NORMAL, EVENT_CONGESTION, EVENT_ACCIDENT,
               EVENT_EMERGENCY, EVENT_FESTIVAL)

EVENT_LABELS = {                       # UI ONLY
    EVENT_NORMAL: "Normal Operations",
    EVENT_CONGESTION: "Sudden Traffic Congestion",
    EVENT_ACCIDENT: "💥 Severe Vehicle Accident",
    EVENT_EMERGENCY: "🚨 Emergency Vehicle (VVIP/Ambulance)",
    EVENT_FESTIVAL: "🎉 City Festival Peak",
}

# ---------------------------------------------------------------- phase codes
PHASE_MAX_GREEN = "MAX_GREEN"
PHASE_ADAPTIVE_SHORT = "ADAPTIVE_SHORT"
PHASE_EMERGENCY_CORRIDOR = "EMERGENCY_CORRIDOR"
PHASE_STANDARD_FIXED = "STANDARD_FIXED"   # default for any node not in `phases`
PHASE_CODES = (PHASE_MAX_GREEN, PHASE_ADAPTIVE_SHORT,
               PHASE_EMERGENCY_CORRIDOR, PHASE_STANDARD_FIXED)

PHASE_LABELS = {                       # UI ONLY
    PHASE_MAX_GREEN: "MAX GREEN WAVE PHASE",
    PHASE_ADAPTIVE_SHORT: "ADAPTIVE TIMING DECREASE",
    PHASE_EMERGENCY_CORRIDOR: "🚨 EMERGENCY FORCED PRIORITY WAVE",
    PHASE_STANDARD_FIXED: "STANDARD FIXED CLASSICAL TIMING",
}

# ---------------------------------------------------------------- tunables
RANDOM_SEED = 42
MAX_QUBITS = 6                         # M3 must never exceed this
TRIAGE_UTILIZATION_THRESHOLD = 0.15    # predicted_inflow / capacity
ANOMALY_ALPHA = 0.001                  # chi-square tail prob for Mahalanobis test
PCA_VARIANCE_TARGET = 0.95             # PCA(n_components=0.95)
