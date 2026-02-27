"""Benchmark script to measure XGBoost model performance."""

import logging
import random
import polars as pl
from datetime import datetime, timezone
from xgboost_forecaster.data import get_substation_metadata, prepare_training_data
from xgboost_forecaster.model import train_model

# Configure logging to only show warnings/errors except for our results
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("benchmark")
log.setLevel(logging.INFO)


def run_benchmark(label: str, num_subs: int = 10):
    metadata = get_substation_metadata()
    available_subs = metadata["substation_name_in_location_table"].to_list()

    # Use a fixed seed for reproducible benchmarking
    random.seed(42)
    sample_subs = random.sample(available_subs, min(num_subs, len(available_subs)))

    test_start_time = datetime(2026, 2, 17, tzinfo=timezone.utc)

    # Prepare data
    all_data = prepare_training_data(sample_subs, metadata, use_lags=True)
    if all_data.is_empty():
        print(f"{label}: No data available")
        return

    # Time-based split
    train_data = all_data.filter(pl.col("timestamp") < test_start_time)
    test_data = all_data.filter(pl.col("timestamp") >= test_start_time)

    if train_data.is_empty() or test_data.is_empty():
        print(f"{label}: Insufficient split data")
        return

    # Train model
    model, metrics = train_model(train_data, time_split=False)
    features = metrics["features"]

    # Evaluate on test set
    X_test = test_data.select(features).to_pandas()
    y_test = test_data.select("power_mw").to_pandas()

    y_pred = model.predict(X_test)

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    print(f"--- {label} ---")
    print(f"Features: {features}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print("-" * (len(label) + 8))
    return mae, rmse


if __name__ == "__main__":
    run_benchmark("Baseline")
