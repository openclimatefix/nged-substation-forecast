"""Benchmark script to measure XGBoost model performance with various feature sets."""

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


def run_benchmark(
    label: str,
    num_subs: int = 10,
    member_selection: str = "mean",
    scale_to_uint8: bool = True,
    exclude_features: list[str] | None = None,
):
    metadata = get_substation_metadata()
    available_subs = metadata["substation_name_in_location_table"].to_list()

    # Use a fixed seed for reproducible benchmarking
    random.seed(42)
    sample_subs = random.sample(available_subs, min(num_subs, len(available_subs)))

    test_start_time = datetime(2026, 2, 17, tzinfo=timezone.utc)

    # Prepare training data
    print(f"Loading training data ({member_selection}, scale_to_uint8={scale_to_uint8})...")
    train_all = prepare_training_data(
        sample_subs,
        metadata,
        use_lags=True,
        member_selection=member_selection,
        scale_to_uint8=scale_to_uint8,
    )
    train_data = train_all.filter(pl.col("timestamp") < test_start_time)

    # Prepare test data
    print(f"Loading test data (scale_to_uint8={scale_to_uint8})...")
    test_all = prepare_training_data(
        sample_subs, metadata, use_lags=True, member_selection="mean", scale_to_uint8=scale_to_uint8
    )
    test_data = test_all.filter(pl.col("timestamp") >= test_start_time)

    if train_data.is_empty() or test_data.is_empty():
        print(f"{label}: Insufficient split data")
        return

    # Filter features if requested
    # We first get all available columns that would be used as features
    target_col = "power_mw"
    potential_features = [
        col
        for col in train_data.columns
        if col not in ["timestamp", target_col, "ensemble_member", "h3_index"]
        and train_data[col].dtype
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

    features = potential_features
    if exclude_features:
        features = [f for f in features if f not in exclude_features]

    # Train with selected features
    model, metrics = train_model(train_data.select(features + [target_col]), time_split=False)
    features = metrics["features"]

    # Evaluate on test set
    X_test = test_data.select(features).to_pandas()
    y_test = test_data.select("power_mw").to_pandas()

    y_pred = model.predict(X_test)

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

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
    scale = False if len(sys.argv) > 3 and sys.argv[3].lower() == "false" else True

    # Map modes
    member_map = {"Baseline": "mean", "Single": "single", "Exploded": "all"}
    selection = member_map.get(mode, "mean")

    # Experiment modes
    if mode == "Pure-Baseline":
        run_benchmark(
            mode,
            num_subs=num_subs,
            member_selection=selection,
            scale_to_uint8=scale,
            exclude_features=["wind_speed_10m", "wind_direction_10m", "windchill"],
        )
    elif mode == "WindChill-Only":
        run_benchmark(
            mode,
            num_subs=num_subs,
            member_selection=selection,
            scale_to_uint8=scale,
            exclude_features=["wind_speed_10m", "wind_direction_10m"],
        )
    elif mode == "WindSpeed-Only":
        run_benchmark(
            mode,
            num_subs=num_subs,
            member_selection=selection,
            scale_to_uint8=scale,
            exclude_features=["windchill", "wind_direction_10m"],
        )
    elif mode == "WindSpeed-Direction":
        run_benchmark(
            mode,
            num_subs=num_subs,
            member_selection=selection,
            scale_to_uint8=scale,
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
            scale_to_uint8=scale,
            exclude_features=["wind_u_10m", "wind_v_10m", "wind_u_100m", "wind_v_100m"],
        )
    else:
        run_benchmark(mode, num_subs=num_subs, member_selection=selection, scale_to_uint8=scale)
