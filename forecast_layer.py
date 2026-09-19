"""OWNER: Member 1.  Layer 3 - XGBoost forecast.
Predictive wave forecast: multi-output XGBoost regression forecasting
5-minute ahead queue lengths across all 6 intersections from compressed PCA eigen-features,
trained on simulated queue-balance dynamics.
"""
from typing import Optional

import numpy as np
import xgboost as xgb

from config import DEFAULT_CAPACITY, NODE_IDS, RANDOM_SEED
from contracts import GridState, StatsResult
from stats_layer import get_stats_model


class ForecasterModel:
    """Multi-output XGBoost model forecasting 5-minute queue dynamics across the network."""
    def __init__(self, seed: int = RANDOM_SEED):
        stats_model = get_stats_model()
        X = stats_model.X_train
        scaler = stats_model.scaler
        pca = stats_model.pca

        # Use sliding window over historical dataset: predict q_{t+1} from features at t
        q_t = X[:, ::3]  # shape (n, 6)
        
        # We predict the next 5-min interval, so Y is q_t shifted by 1
        Y = q_t[1:]
        
        # Features projected onto PCA eigen-basis
        X_scaled = scaler.transform(X[:-1]) # Match length of Y
        Z = pca.transform(X_scaled)
        
        # Adjust n since we dropped 1 row
        n = len(Y)

        # 80/20 train/validation split
        n_train = int(0.8 * n)
        Z_train, Z_val = Z[:n_train], Z[n_train:]
        Y_train, Y_val = Y[:n_train], Y[n_train:]
        q_val = q_t[n_train:-1]

        self.regressor = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            tree_method="hist",
            random_state=seed,
        )
        self.regressor.fit(Z_train, Y_train)

        # Validation diagnostics vs naive persistence baseline ("queue stays same")
        preds_val = self.regressor.predict(Z_val)
        self.xgb_mae = float(np.abs(preds_val - Y_val).mean())
        self.xgb_rmse = float(np.sqrt(np.mean((preds_val - Y_val) ** 2)))
        self.persist_mae = float(np.abs(q_val - Y_val).mean())
        self.persist_rmse = float(np.sqrt(np.mean((q_val - Y_val) ** 2)))

        assert self.xgb_mae < self.persist_mae, (
            f"XGBoost MAE ({self.xgb_mae:.3f}) must outperform persistence baseline ({self.persist_mae:.3f})"
        )

    def predict(self, eigen_features: np.ndarray, capacities: np.ndarray) -> np.ndarray:
        """Predict next 5-minute queue lengths given 1-D eigen_features."""
        # Ensure 2-D input for XGBoost
        feat_2d = eigen_features.reshape(1, -1)
        pred_raw = self.regressor.predict(feat_2d)[0]
        # Clip to [0, capacity_i] per node
        return np.clip(pred_raw, 0.0, capacities).astype(float)


# Module-level lazy singleton cache
_FORECASTER_MODEL: Optional[ForecasterModel] = None


def get_forecaster_model() -> ForecasterModel:
    """Retrieve or initialize the cached ForecasterModel singleton."""
    global _FORECASTER_MODEL
    if _FORECASTER_MODEL is None:
        _FORECASTER_MODEL = ForecasterModel(seed=RANDOM_SEED)
    return _FORECASTER_MODEL


def forecast_inflows(stats: StatsResult, grid_state: GridState) -> np.ndarray:
    """Layer 3 entrypoint: return shape (6,) predicted 5-min queue per node, in NODE_IDS order."""
    model = get_forecaster_model()
    capacities = np.array([float(grid_state[n]["capacity"]) for n in NODE_IDS], dtype=float)
    return model.predict(stats.eigen_features, capacities)

