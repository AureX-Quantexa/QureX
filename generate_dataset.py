import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_traffic_dataset(filename="historical_traffic.csv", n_days=7, seed=42):
    """Generate a realistic 5-minute interval traffic dataset."""
    rng = np.random.default_rng(seed)
    
    start_time = datetime(2023, 10, 1, 0, 0)
    intervals = n_days * 24 * 12 # 5 min intervals
    
    timestamps = [start_time + timedelta(minutes=5 * i) for i in range(intervals)]
    
    # Base daily pattern (rush hours around 8 AM and 5 PM)
    def daily_factor(dt):
        hour = dt.hour + dt.minute / 60.0
        # Morning rush
        m_rush = np.exp(-0.5 * ((hour - 8.0) / 1.5)**2)
        # Evening rush
        e_rush = np.exp(-0.5 * ((hour - 17.5) / 2.0)**2)
        # Night lull
        base = 0.2 if hour < 5 or hour > 22 else 0.5
        # Weekend factor
        is_weekend = dt.weekday() >= 5
        wknd_mult = 0.6 if is_weekend else 1.0
        
        return base + (m_rush + e_rush * 1.2) * wknd_mult
        
    data = {"timestamp": timestamps}
    
    D = np.array([daily_factor(t) for t in timestamps])
    
    # 6 nodes
    for i in range(1, 7):
        node = f"Intersection_{i}"
        
        # Local factor
        e = rng.normal(0, 1.0, size=intervals)
        # High independent variance 'e' so single-node Emergency is normal
        # Low correlated variance 'D' so city-wide Festival is anomalous
        q = np.clip(12.0 + 2.0 * (D - 0.2) + 6.0 * e, 0, 50)
        occ = np.clip(25.0 + 10.0 * (D - 0.2) + 15.0 * e, 0, 100)
        spd = np.clip(42.0 - 5.0 * (D - 0.2) - 10.0 * e, 5, 60)
        
        data[f"{node}_queue"] = q.astype(int)
        data[f"{node}_occupancy"] = occ.round(1)
        data[f"{node}_avg_speed"] = spd.round(1)
        
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Successfully generated {filename} with {intervals} rows.")

if __name__ == "__main__":
    generate_traffic_dataset()
