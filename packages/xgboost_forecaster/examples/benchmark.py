"""Benchmark script to measure XGBoost model performance with various feature sets."""

import logging
import math
import random
import sys
from datetime import datetime, timezone
from typing import cast

import polars as pl
from xgboost_forecaster import (
    XGBoostForecaster,
    get_substation_metadata,
    prepare_training_data,
)

# Configure logging to only show warnings/errors except for our results
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("benchmark")
log.setLevel(logging.INFO)


def run_benchmark(
    label: str,
    num_subs: int = 10,
    member_selection: str = "mean",
    exclude_features: list[str] | None = None,
):
    metadata = get_substation_metadata()
    available_subs = metadata["substation_number"].to_list()

    # Use a fixed seed for reproducible benchmarking
    random.seed(42)
    sample_subs = random.sample(available_subs, min(num_subs, len(available_subs)))

    test_start_time = datetime(2026, 2, 17, tzinfo=timezone.utc)

    # Prepare training data
    print(f"Loading training data ({member_selection})...")
    train_all = prepare_training_data(
        sample_subs,
        metadata,
        use_lags=True,
        member_selection=member_selection,
    )
    if train_all.is_empty():
        print(f"{label}: No training data")
        return

    train_data = train_all.filter(pl.col("timestamp") < test_start_time)

    # Prepare test data
    print("Loading test data...")
    test_all = prepare_training_data(sample_subs, metadata, use_lags=True, member_selection="mean")
    if test_all.is_empty():
        print(f"{label}: No test data")
        return

    test_data = test_all.filter(pl.col("timestamp") >= test_start_time)

    if train_data.is_empty() or test_data.is_empty():
        print(f"{label}: Insufficient split data")
        return

    # Filter features if requested
    target_col = "power_mw"
    potential_features = [
        col
        for col in train_data.columns
        if col not in ["timestamp", target_col, "ensemble_member", "h3_index", "substation_number"]
        and train_data[col].dtype.is_numeric()
    ]

    features = potential_features
    if exclude_features:
        features = [f for f in features if f not in exclude_features]

    # Train with selected features
    forecaster = XGBoostForecaster()
    forecaster.train(train_data, target_col=target_col, feature_cols=features)

    # Evaluate on test set
    preds = forecaster.predict(test_data)
    y_test = test_data[target_col]

    mae = (y_test - preds).abs().mean()
    mean_sq_err = (y_test - preds).pow(2).mean()
    rmse = math.sqrt(cast(float, mean_sq_err)) if mean_sq_err is not None else 0.0

    print(f"--- {label} ---")
    print(f"Features ({len(features)}): {features}")
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

    # Experiment modes
    if mode == "Pure-Baseline":
        run_benchmark(
            mode,
            num_subs=num_subs,
            member_selection=selection,
            exclude_features=["wind_speed_10m", "wind_direction_10m", "windchill"],
        )
    elif mode == "WindChill-Only":
        run_benchmark(
            mode,
            num_subs=num_subs,
            member_selection=selection,
            exclude_features=["wind_speed_10m", "wind_direction_10m"],
        )
    elif mode == "WindSpeed-Only":
        run_benchmark(
            mode,
            num_subs=num_subs,
            member_selection=selection,
            exclude_features=["windchill", "wind_direction_10m"],
        )
    elif mode == "WindSpeed-Direction":
        run_benchmark(
            mode,
            num_subs=num_subs,
            member_selection=selection,
            exclude_features=[
                "wind_u_10m",
                "wind_v_10m",
                "wind_u_100m",
                "wind_v_100m",
                "windchill",
            ],
        )
    elif mode == "WindAll-Polar":
        run_benchmark(
            mode,
            num_subs=num_subs,
            member_selection=selection,
            exclude_features=["wind_u_10m", "wind_v_10m", "wind_u_100m", "wind_v_100m"],
        )
    else:
        run_benchmark(mode, num_subs=num_subs, member_selection=selection)
