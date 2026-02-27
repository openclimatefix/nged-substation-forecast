"""Plot ensemble forecast vs ground truth for 5 substations on one figure."""

import logging
import random
from datetime import datetime, timedelta, timezone

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


def plot_multi_substation_forecast(num_subs: int = 5):
    log.info("Fetching substation metadata...")
    metadata = get_substation_metadata()

    available_subs = metadata["substation_name_in_location_table"].to_list()
    if len(available_subs) < num_subs:
        log.warning(f"Only {len(available_subs)} substations available.")
        num_subs = len(available_subs)

    sample_subs = random.sample(available_subs, num_subs)
    log.info(f"Selected substations: {sample_subs}")

    # Determine a common test start time and NWP run
    # Based on previous exploration, we have weather up to Feb 23rd and power up to Feb 24th.
    # We'll use Feb 17th as a common split point to ensure we have some training data (started Jan 27th)
    # and a decent forecast window.
    test_start_time = datetime(2026, 2, 17, tzinfo=timezone.utc)

    # Find the latest NWP run before test_start_time
    weather_files = sorted(BASE_WEATHER_PATH.glob("*.parquet"))
    latest_init_time = None
    for f in reversed(weather_files):
        file_init_time = datetime.strptime(f.stem, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
        if file_init_time <= test_start_time:
            latest_init_time = file_init_time
            break

    if not latest_init_time:
        log.error("No suitable NWP initialization found.")
        return

    log.info(f"Using common NWP initialization: {latest_init_time}")
    forecast_end_time = latest_init_time + timedelta(days=7)

    fig, axes = plt.subplots(num_subs, 1, figsize=(14, 4 * num_subs), sharex=True)
    if num_subs == 1:
        axes = [axes]

    for i, sub_name in enumerate(sample_subs):
        ax = axes[i]
        log.info(f"Processing {sub_name}...")
        try:
            # 1. Prepare data and split
            all_data = prepare_training_data(sub_name, metadata, use_lags=True).sort("timestamp")
            train_data = all_data.filter(pl.col("timestamp") < test_start_time)

            if train_data.is_empty():
                log.warning(f"No training data for {sub_name} before {test_start_time}")
                continue

            # 2. Train model
            model, model_meta = train_model(train_data, time_split=False)
            features = model_meta["features"]

            # 3. Load weather ensemble for the 7-day window
            sub_meta = metadata.filter(pl.col("substation_name_in_location_table") == sub_name)
            h3_index = sub_meta["h3_index"][0]

            weather_ensemble = load_weather_data(
                [h3_index],
                start_date=latest_init_time.strftime("%Y-%m-%d"),
                end_date=forecast_end_time.strftime("%Y-%m-%d"),
                init_time=latest_init_time,
                average_ensembles=False,
            )

            if weather_ensemble.is_empty():
                log.warning(f"No weather ensemble for {sub_name}")
                continue

            # 4. Prepare prediction data with lags
            parquet_file = sub_meta["parquet_filename"][0]
            power_for_lags = load_substation_power(parquet_file).sort("timestamp")
            lag_table = power_for_lags.with_columns(
                [
                    pl.col("power_mw").shift(288).alias("power_mw_lag_1d"),
                    pl.col("power_mw").shift(2016).alias("power_mw_lag_7d"),
                ]
            ).select(["timestamp", "power_mw_lag_1d", "power_mw_lag_7d"])

            pred_data = weather_ensemble.join(lag_table, on="timestamp", how="inner")
            pred_data = (
                pred_data.with_columns(
                    [
                        pl.col("timestamp").dt.hour().alias("hour"),
                        pl.col("timestamp").dt.weekday().alias("day_of_week"),
                        pl.col("timestamp").dt.month().alias("month"),
                    ]
                )
                .filter(pl.col("timestamp") <= forecast_end_time)
                .drop_nulls(subset=features)
            )

            if pred_data.is_empty():
                log.warning(f"No prediction data for {sub_name} in forecast window.")
                continue

            # 5. Predict and Plot Ensemble
            ensemble_members = pred_data["ensemble_member"].unique().sort()
            for member in ensemble_members:
                member_data = pred_data.filter(pl.col("ensemble_member") == member).sort(
                    "timestamp"
                )
                preds = predict(model, member_data, features)
                ax.plot(
                    member_data["timestamp"],
                    preds,
                    color="skyblue",
                    alpha=0.2,
                    linewidth=1,
                )

            # Ensemble Mean
            ensemble_mean = (
                pred_data.group_by("timestamp")
                .agg([pl.col(f).mean() for f in features])
                .sort("timestamp")
            )
            mean_preds = predict(model, ensemble_mean, features)
            ax.plot(
                ensemble_mean["timestamp"],
                mean_preds,
                color="blue",
                linewidth=2,
                label="Ensemble Mean Forecast",
            )

            # Ground Truth
            actual = power_for_lags.filter(
                (pl.col("timestamp") >= latest_init_time)
                & (pl.col("timestamp") <= forecast_end_time)
            )
            ax.plot(
                actual["timestamp"],
                actual["power_mw"],
                color="black",
                linewidth=2,
                linestyle="--",
                label="Actual Ground Truth",
            )

            ax.set_title(f"Substation: {sub_name}")
            ax.set_ylabel("Power (MW)")
            ax.legend(loc="upper right", fontsize="small")
            ax.grid(True, alpha=0.3)

        except Exception as e:
            log.exception(f"Failed to process {sub_name}: {e}")

    plt.xlabel("Timestamp")
    plt.suptitle(
        f"7-Day Ensemble Forecasts for 5 Substations\n(NWP Init: {latest_init_time})", fontsize=16
    )
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename = "multi_substation_forecast.png"
    plt.savefig(filename)
    log.info(f"Multi-substation plot saved to {filename}")


if __name__ == "__main__":
    plot_multi_substation_forecast(5)
