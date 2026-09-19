"""OWNER: Member 1.  Layer 2 - Mahalanobis + PCA.
Statistical sanitization: checks for network-wide telemetry anomalies using
Ledoit-Wolf shrunk covariance Mahalanobis distance, and compresses the feature
space using standardized Principal Component Analysis (PCA).
"""
from typing import Optional

import numpy as np
import scipy.stats as st
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import pandas as pd

from config import (ANOMALY_ALPHA, N_NODES, PCA_VARIANCE_TARGET, RANDOM_SEED, NODE_IDS)
from contracts import GridState, StatsResult, flatten_grid_state


def load_historical_data(filename: str = "historical_traffic.csv") -> np.ndarray:
    """Load historical traffic data from CSV and format as (n, 18) array.
    Laid out as [queue, occupancy, avg_speed] x 6 nodes.
    """
    df = pd.read_csv(filename)
    n = len(df)
    
    # Extract columns in the correct order for the 18-element feature vector
    features = []
    for node in NODE_IDS:
        features.append(df[f"{node}_queue"].values)
        features.append(df[f"{node}_occupancy"].values)
        features.append(df[f"{node}_avg_speed"].values)
        
    X = np.stack(features, axis=1) # shape (n, 18)
    return X


class StatsModel:
    """Statistical model holding fitted parameters for Mahalanobis anomaly detection
    and PCA feature compression.
    """
    def __init__(self, X: np.ndarray, alpha: float = ANOMALY_ALPHA,
                 var_target: float = PCA_VARIANCE_TARGET, seed: int = RANDOM_SEED):
        self.X_train = X
        self.mu = X.mean(axis=0)
        # Ledoit-Wolf shrinkage provides well-conditioned, stable covariance inverse
        self.cov = LedoitWolf().fit(X).covariance_
        self.prec = np.linalg.inv(self.cov)
        self.d2_thr = float(st.chi2.ppf(1.0 - alpha, df=X.shape[1]))
        self.threshold = float(np.sqrt(self.d2_thr))

        # Standardize features prior to PCA
        self.scaler = StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)
        self.pca = PCA(n_components=var_target, random_state=seed).fit(X_scaled)
        self.variance_retained = float(self.pca.explained_variance_ratio_.sum())
        self.n_components = int(self.pca.n_components_)

    def score(self, x: np.ndarray) -> StatsResult:
        """Evaluate raw 18-element feature vector."""
        d = x - self.mu
        d2 = float(d @ self.prec @ d)
        mahalanobis = float(np.sqrt(max(0.0, d2)))
        is_anomaly = bool(d2 > self.d2_thr)

        # PCA transformation returns strictly 1-D array
        x_scaled = self.scaler.transform(x.reshape(1, -1))
        eigen_features = self.pca.transform(x_scaled)[0]

        return StatsResult(
            mahalanobis=mahalanobis,
            threshold=self.threshold,
            is_anomaly=is_anomaly,
            eigen_features=eigen_features,
            variance_retained=self.variance_retained,
        )


# Module-level lazy singleton cache
_STATS_MODEL: Optional[StatsModel] = None


def get_stats_model() -> StatsModel:
    """Retrieve or initialize the cached StatsModel singleton."""
    global _STATS_MODEL
    if _STATS_MODEL is None:
        X = load_historical_data("historical_traffic.csv")
        _STATS_MODEL = StatsModel(X, alpha=ANOMALY_ALPHA, var_target=PCA_VARIANCE_TARGET, seed=RANDOM_SEED)
    return _STATS_MODEL


def run_statistics(grid_state: GridState) -> StatsResult:
    """Layer 2 entrypoint: compute Mahalanobis distance, threshold test, and PCA eigen_features."""
    model = get_stats_model()
    x = flatten_grid_state(grid_state)
    return model.score(x)

