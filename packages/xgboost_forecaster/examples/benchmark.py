"""Benchmark script to measure XGBoost model performance."""

import logging
import random
import sys
from datetime import datetime, timezone

import polars as pl
from xgboost_forecaster.data import get_substation_metadata, prepare_training_data
from xgboost_forecaster.model import train_model

# Configure logging to only show warnings/errors except for our results
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("benchmark")
log.setLevel(logging.INFO)


def run_benchmark(label: str, num_subs: int = 10, member_selection: str = "mean"):
    metadata = get_substation_metadata()
    available_subs = metadata["substation_name_in_location_table"].to_list()

    # Use a fixed seed for reproducible benchmarking
    random.seed(42)
    sample_subs = random.sample(available_subs, min(num_subs, len(available_subs)))

    test_start_time = datetime(2026, 2, 17, tzinfo=timezone.utc)

    # Prepare training data
    print(f"Loading training data ({member_selection})...")
    train_all = prepare_training_data(
        sample_subs, metadata, use_lags=True, member_selection=member_selection
    )
    train_data = train_all.filter(pl.col("timestamp") < test_start_time)

    # Prepare test data (Always use mean for evaluation consistency)
    print("Loading test data...")
    test_all = prepare_training_data(sample_subs, metadata, use_lags=True, member_selection="mean")
    test_data = test_all.filter(pl.col("timestamp") >= test_start_time)

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
    mode = sys.argv[1] if len(sys.argv) > 1 else "Baseline"
    num_subs = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    # Map modes
    member_map = {"Baseline": "mean", "Single": "single", "Exploded": "all"}

    selection = member_map.get(mode, "mean")
    run_benchmark(mode, num_subs=num_subs, member_selection=selection)
