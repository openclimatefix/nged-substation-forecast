"""XGBoost model training and inference."""

import logging
from typing import Any

import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import polars as pl

log = logging.getLogger(__name__)


def train_model(
    data: pl.DataFrame,
    target_col: str = "power_mw",
    test_size: float = 0.2,
    time_split: bool = False,
    **xgb_params: Any,
) -> tuple[xgb.XGBRegressor, dict[str, Any]]:
    """Train an XGBoost model and return it along with performance metrics.

    Args:
        data: Input polars DataFrame.
        target_col: Target variable name.
        test_size: Fraction of data for testing.
        time_split: If True, uses the last `test_size` fraction of data for testing (preserving time order).
        **xgb_params: Additional XGBoost parameters.
    """
    # Simple feature selection: numeric columns except timestamp and target
    features = [
        col
        for col in data.columns
        if col not in ["timestamp", target_col, "ensemble_member", "h3_index"]
        and data[col].dtype
        in [
            pl.Float32,
            pl.Float64,
            pl.Int32,
            pl.Int64,
            pl.UInt32,
            pl.UInt64,
            pl.Int8,
            pl.UInt8,
            pl.Int16,
        ]
    ]

    # Drop rows with nulls (e.g. from lags)
    data = data.drop_nulls(subset=features + [target_col])

    if time_split:
        # Sort by timestamp and split
        data_sorted = data.sort("timestamp")
        split_idx = int(len(data_sorted) * (1 - test_size))

        train_df = data_sorted[:split_idx]
        test_df = data_sorted[split_idx:]

        X_train = train_df.select(features).to_pandas()
        y_train = train_df.select(target_col).to_pandas()
        X_test = test_df.select(features).to_pandas()
        y_test = test_df.select(target_col).to_pandas()
        test_timestamps = test_df["timestamp"]
    else:
        X = data.select(features).to_pandas()
        y = data.select(target_col).to_pandas()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        test_timestamps = None

    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "objective": "reg:squarederror",
        "n_jobs": -1,
        "early_stopping_rounds": 10,
        "tree_method": "hist",  # Memory efficient
    }
    params.update(xgb_params)

    model = xgb.XGBRegressor(**params)
    # Fit with a validation set for early stopping
    # We use X_test/y_test as eval_set because it's already split
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)

    mse = float(mean_squared_error(y_test, y_pred))
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": mse**0.5,
        "y_test": y_test,
        "y_pred": y_pred,
        "test_timestamps": test_timestamps,
        "features": features,
    }

    log.info(
        f"Model trained on {len(X_train)} rows. Best iteration: {model.get_booster().best_iteration}. MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}"
    )

    return model, metrics


def predict(model: xgb.XGBRegressor, data: pl.DataFrame, features: list[str]) -> pl.Series:
    """Run inference using a trained model."""
    X = data.select(features).to_pandas()
    preds = model.predict(X)
    return pl.Series("predictions", preds)
