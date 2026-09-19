"""
VERIFIED REFERENCE for Layer 2 (stats) + Layer 3 (forecast).  Owner: M1 adapts this into
stats_layer.py / forecast_layer.py.  Tested with numpy 2.4, scipy 1.17, scikit-learn 1.8, xgboost 3.4.
Adapt to the frozen contracts (StatsResult, forecast shape (6,)); do NOT copy blindly.
Everything here is SIMULATED data - label it that way in the UI.
"""
import numpy as np
import scipy.stats as st
import xgboost as xgb
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

SEED, N_NODES = 42, 6


def simulate_history(n=2000, seed=SEED):
    """Correlated telemetry: queue up -> occupancy up -> speed down; one city-wide demand factor
    plus a per-node factor.  Returns X of shape (n, 18) laid out [q,occ,spd] x 6 nodes."""
    rng = np.random.default_rng(seed)
    D = rng.normal(size=(n, 1))                      # city-wide demand factor
    e = rng.normal(size=(n, N_NODES))                # node-local factor
    zq = 0.6 * D + 0.8 * e                           # unit variance
    q = 15 + 3 * zq
    occ = 30 + 8 * (0.85 * zq + 0.527 * rng.normal(size=(n, N_NODES)))
    zo = (occ - 30) / 8
    spd = 40 - 5 * (0.8 * zo + 0.6 * rng.normal(size=(n, N_NODES)))
    return np.stack([q, occ, spd], axis=2).reshape(n, 18)


class StatsModel:
    def __init__(self, X, alpha=0.001, var_target=0.95):
        self.mu = X.mean(0)
        self.cov = LedoitWolf().fit(X).covariance_          # shrinkage: stable inverse
        self.prec = np.linalg.inv(self.cov)
        self.d2_thr = st.chi2.ppf(1 - alpha, df=X.shape[1])  # ~42.31 for p=18, alpha=.001
        self.pca = PCA(n_components=var_target, random_state=SEED).fit(X)

    def score(self, x):
        d = x - self.mu
        d2 = float(d @ self.prec @ d)
        return dict(mahalanobis=d2 ** 0.5, threshold=self.d2_thr ** 0.5, is_anomaly=d2 > self.d2_thr,
                    eigen_features=self.pca.transform(x.reshape(1, -1))[0],       # 1-D !
                    variance_retained=float(self.pca.explained_variance_ratio_.sum()))


def train_forecaster(X, pca, seed=SEED):
    """SIMULATED target: next-5-min queue = 1.15*queue + noise (replace with queue-balance dynamics)."""
    rng = np.random.default_rng(seed)
    Z = pca.transform(X)
    Y = X[:, ::3] * 1.15 + rng.normal(0, 1.0, (len(X), N_NODES))
    n_tr = int(0.8 * len(X))
    m = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.08, subsample=0.9,
                         random_state=seed)              # native multi-output (2-D y)
    m.fit(Z[:n_tr], Y[:n_tr])
    mae = np.abs(m.predict(Z[n_tr:]) - Y[n_tr:]).mean()
    persist = np.abs(X[n_tr:, ::3] - Y[n_tr:]).mean()     # naive baseline: "queue stays the same"
    return m, mae, persist


if __name__ == "__main__":
    X = simulate_history()
    sm = StatsModel(X)
    print(f"PCA components for 95%: {sm.pca.n_components_} | retained {sm.score(X[0])['variance_retained']:.3f}")
    base = np.array([[15, 30, 40]] * 6, float)

    def with_node1(v):
        b = base.copy(); b[0] = v; return b.ravel()

    cases = {"default": base.ravel(),
             "congestion(45,90,6)": with_node1([45, 90, 6]),
             "accident(38,95,2)": with_node1([38, 95, 2]),
             "emergency(24,60,22)": with_node1([24, 60, 22]),
             "festival(all 40,82,12)": np.tile([40, 82, 12], 6).astype(float)}
    for k, x in cases.items():
        r = sm.score(x)
        print(f"{k:24s} D={r['mahalanobis']:6.2f} thr={r['threshold']:.2f} anomaly={r['is_anomaly']}")
    m, mae, persist = train_forecaster(X, sm.pca)
    print(f"forecast MAE {mae:.2f} vs persistence baseline {persist:.2f}")
    f = m.predict(sm.score(cases['congestion(45,90,6)'])['eigen_features'].reshape(1, -1))[0]
    print("forecast shape:", f.shape)
