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

from config import (ANOMALY_ALPHA, N_NODES, PCA_VARIANCE_TARGET, RANDOM_SEED)
from contracts import GridState, StatsResult, flatten_grid_state


def simulate_history(n: int = 2000, seed: int = RANDOM_SEED) -> np.ndarray:
    """Correlated telemetry generator: queue up -> occupancy up -> speed down.
    Generates city-wide demand factor plus node-local factor.
    Returns X of shape (n, 18) laid out [queue, occupancy, avg_speed] x 6 nodes.
    """
    rng = np.random.default_rng(seed)
    D = rng.normal(size=(n, 1))                      # City-wide demand factor
    e = rng.normal(size=(n, N_NODES))                # Per-node local factor
    zq = 0.6 * D + 0.8 * e                           # Unit variance queue latent
    q = 15.0 + 3.0 * zq
    occ = 30.0 + 8.0 * (0.85 * zq + 0.527 * rng.normal(size=(n, N_NODES)))
    zo = (occ - 30.0) / 8.0
    spd = 40.0 - 5.0 * (0.8 * zo + 0.6 * rng.normal(size=(n, N_NODES)))
    return np.stack([q, occ, spd], axis=2).reshape(n, 18)


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
        X = simulate_history(n=2000, seed=RANDOM_SEED)
        _STATS_MODEL = StatsModel(X, alpha=ANOMALY_ALPHA, var_target=PCA_VARIANCE_TARGET, seed=RANDOM_SEED)
    return _STATS_MODEL


def run_statistics(grid_state: GridState) -> StatsResult:
    """Layer 2 entrypoint: compute Mahalanobis distance, threshold test, and PCA eigen_features."""
    model = get_stats_model()
    x = flatten_grid_state(grid_state)
    return model.score(x)

