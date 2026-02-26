"""XGBoost model training and inference."""

import logging
from typing import Any

import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import polars as pl

log = logging.getLogger(__name__)


def train_model(
    data: pl.DataFrame, target_col: str = "power_mw", test_size: float = 0.2, **xgb_params: Any
) -> tuple[xgb.XGBRegressor, dict[str, float]]:
    """Train an XGBoost model and return it along with performance metrics."""

    # Simple feature selection: numeric columns except timestamp and target
    features = [
        col
        for col in data.columns
        if col not in ["timestamp", target_col]
        and data[col].dtype
        in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64, pl.Int8, pl.UInt8]
    ]

    X = data.select(features).to_pandas()
    y = data.select(target_col).to_pandas()

    # Time-based split would be better, but for a "simple" model we'll use random split for now
    # as per "simple model" instruction.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "objective": "reg:squarederror",
        "n_jobs": -1,
    }
    params.update(xgb_params)

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = float(mean_squared_error(y_test, y_pred))
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": mse**0.5,
    }

    log.info(f"Model trained. Metrics: {metrics}")

    return model, metrics


def predict(model: xgb.XGBRegressor, data: pl.DataFrame) -> pl.Series:
    """Run inference using a trained model."""
    features = model.get_booster().feature_names
    X = data.select(features).to_pandas()
    preds = model.predict(X)
    return pl.Series("predictions", preds)
