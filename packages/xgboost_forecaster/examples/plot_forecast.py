"""Plot ensemble forecast vs ground truth for a substation."""

import logging
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import polars as pl
from xgboost_forecaster.data import (
    BASE_WEATHER_PATH,
    get_substation_metadata,
    load_substation_power,
    load_weather_data,
    prepare_training_data,
)
from xgboost_forecaster.model import predict, train_model

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


def plot_substation_ensemble_forecast(sub_name: str = "Lawford 33 11kv S Stn"):
    log.info(f"Fetching metadata and preparing data for {sub_name}...")
    metadata = get_substation_metadata()

    try:
        # 1. Prepare training data with lags
        # We'll use 80% for training
        all_data = prepare_training_data(sub_name, metadata, use_lags=True)
        if all_data.is_empty():
            log.error(f"No overlapping data found for {sub_name}.")
            return

        all_data = all_data.sort("timestamp")
        split_idx = int(len(all_data) * 0.8)
        train_data = all_data[:split_idx]
        test_data_actual = all_data[split_idx:]

        if test_data_actual.is_empty():
            log.error("Test data is empty.")
            return

        # 2. Train model
        log.info(f"Training model for {sub_name}...")
        model, metrics = train_model(train_data, time_split=False)
        features = metrics["features"]

        # 3. Find the most recent NWP run before the test period starts
        test_start_time = test_data_actual["timestamp"].min()
        weather_files = sorted(BASE_WEATHER_PATH.glob("*.parquet"))

        # Find the latest file where initialization is <= test_start_time
        latest_init_file = None
        latest_init_time = None
        for f in reversed(weather_files):
            # Filename is YYYY-MM-DDTHH.parquet
            file_init_time = datetime.strptime(f.stem, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)

            if file_init_time <= test_start_time:
                latest_init_file = f
                latest_init_time = file_init_time
                break

        if not latest_init_file or latest_init_time is None:
            log.error("No suitable NWP initialization found before test period.")
            return

        log.info(f"Using NWP initialization: {latest_init_time}")

        # 4. Load weather for that run without averaging ensembles
        sub_meta = metadata.filter(pl.col("substation_name_in_location_table") == sub_name)
        h3_index = sub_meta["h3_index"][0]

        # Load weather for the whole test period for this specific init run
        weather_ensemble = load_weather_data(
            [h3_index],
            start_date=latest_init_time.strftime("%Y-%m-%d"),
            end_date=all_data["timestamp"].max().strftime("%Y-%m-%d"),
            init_time=latest_init_time,
            average_ensembles=False,
        )

        if weather_ensemble.is_empty():
            log.error("Weather ensemble is empty for selected run.")
            return

        # 5. Prepare feature data for prediction
        # We need: Weather + Lags + Temporal features
        # Lags come from historical power data.
        parquet_file = sub_meta["parquet_filename"][0]
        power_for_lags = load_substation_power(parquet_file).sort("timestamp")

        # Create lag table
        lag_table = power_for_lags.with_columns(
            [
                pl.col("power_mw").shift(288).alias("power_mw_lag_1d"),
                pl.col("power_mw").shift(2016).alias("power_mw_lag_7d"),
            ]
        ).select(["timestamp", "power_mw_lag_1d", "power_mw_lag_7d"])

        # Join weather ensemble with lags
        pred_data = weather_ensemble.join(lag_table, on="timestamp", how="inner")

        # Temporal features
        pred_data = pred_data.with_columns(
            [
                pl.col("timestamp").dt.hour().alias("hour"),
                pl.col("timestamp").dt.weekday().alias("day_of_week"),
                pl.col("timestamp").dt.month().alias("month"),
            ]
        ).drop_nulls(subset=features)

        if pred_data.is_empty():
            log.error("Prediction data is empty after joining with lags.")
            return

        # 6. Predict for each ensemble member
        ensemble_members = pred_data["ensemble_member"].unique().sort()
        plt.figure(figsize=(12, 7))

        log.info(f"Generating predictions for {len(ensemble_members)} ensemble members...")
        for member in ensemble_members:
            member_data = pred_data.filter(pl.col("ensemble_member") == member).sort("timestamp")
            if member_data.is_empty():
                continue
            # Pass individual ensemble member data
            preds = predict(model, member_data, features)
            plt.plot(
                member_data["timestamp"],
                preds,
                color="skyblue",
                alpha=0.3,
                linewidth=1,
                label="Ensemble Forecast" if member == ensemble_members[0] else "",
            )

        # Plot Ground Truth
        # Filter ground truth to match the prediction period
        pred_timestamps = pred_data["timestamp"].unique().sort()
        actual_matching = test_data_actual.filter(pl.col("timestamp").is_in(pred_timestamps)).sort(
            "timestamp"
        )

        plt.plot(
            actual_matching["timestamp"],
            actual_matching["power_mw"],
            color="black",
            linewidth=2,
            label="Actual Ground Truth",
        )

        plt.title(
            f"Production-Realistic Ensemble Forecast: {sub_name}\n(NWP Init: {latest_init_time})"
        )
        plt.xlabel("Timestamp")
        plt.ylabel("Power (MW)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        filename = f"ensemble_forecast_{sub_name.replace(' ', '_').replace('/', '_')}.png"
        plt.savefig(filename)
        log.info(f"Plot saved to {filename}")

    except Exception as e:
        log.exception(f"Failed to plot ensemble forecast for {sub_name}: {e}")


if __name__ == "__main__":
    plot_substation_ensemble_forecast("Lawford 33 11kv S Stn")
