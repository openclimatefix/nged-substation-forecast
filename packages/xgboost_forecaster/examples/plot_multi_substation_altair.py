"""Plot ensemble forecast vs ground truth for 5 substations on one figure using Altair."""

import logging
import random
from datetime import datetime, timedelta, timezone

import altair as alt
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


def plot_multi_substation_ensemble_altair(num_subs: int = 5):
    log.info("Fetching substation metadata...")
    metadata = get_substation_metadata()

    # Randomly select substations
    available_subs = metadata["substation_name_in_location_table"].to_list()
    sample_subs = random.sample(available_subs, min(num_subs, len(available_subs)))
    log.info(f"Selected substations: {sample_subs}")

    # Set common test start time and forecast window
    test_start_time = datetime(2026, 2, 17, tzinfo=timezone.utc)

    # Find latest NWP run before test period
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

    log.info(f"Using NWP initialization: {latest_init_time}")
    forecast_end_time = latest_init_time + timedelta(days=7)

    # 1. Train ONE global model for all 5 substations
    log.info("Preparing training data for global model...")
    all_train_data = prepare_training_data(sample_subs, metadata, use_lags=True)

    if all_train_data.is_empty():
        log.error("Global training data is empty.")
        return

    # Split training data (before test_start_time)
    train_data = all_train_data.filter(pl.col("timestamp") < test_start_time)

    log.info(
        f"Training global model on {train_data.height} samples across {len(sample_subs)} substations..."
    )
    model, model_meta = train_model(train_data, time_split=False)
    features = model_meta["features"]

    # 2. Generate forecasts for each substation and ensemble member
    all_plot_data = []

    for sub_name in sample_subs:
        log.info(f"Generating forecast for {sub_name}...")
        sub_meta = metadata.filter(pl.col("substation_name_in_location_table") == sub_name)
        h3_index = sub_meta["h3_index"][0]
        sub_id = sub_meta["substation_number"][0]
        parquet_file = sub_meta["parquet_filename"][0]

        # Historical power for lags (5-min res)
        power_full = load_substation_power(parquet_file)
        if power_full.is_empty():
            log.warning(f"Skipping {sub_name} due to lack of sane power data.")
            continue

        power_full = power_full.sort("timestamp")

        # Weather ensemble (upsampled to 5-min)
        weather_ensemble = load_weather_data(
            [h3_index],
            start_date=latest_init_time.strftime("%Y-%m-%d"),
            end_date=all_train_data["timestamp"].max().strftime("%Y-%m-%d"),
            init_time=latest_init_time,
            average_ensembles=False,
            upsample_to_5min=True,
        )

        if weather_ensemble.is_empty():
            log.warning(
                f"Skipping {sub_name} due to lack of weather data for selected initialization."
            )
            continue

        # Historical power for lags (5-min res)
        # We must strictly only use power data from BEFORE latest_init_time for prediction lags
        power_for_lags = power_full.filter(pl.col("timestamp") <= latest_init_time)

        lag_table_7d = power_for_lags.select(
            [
                (pl.col("timestamp") + timedelta(days=7)).alias("timestamp"),
                pl.col("power_mw").alias("power_mw_lag_7d"),
            ]
        )
        lag_table_14d = power_for_lags.select(
            [
                (pl.col("timestamp") + timedelta(days=14)).alias("timestamp"),
                pl.col("power_mw").alias("power_mw_lag_14d"),
            ]
        )

        # Prepare prediction features
        pred_data = weather_ensemble.join(lag_table_7d, on="timestamp", how="inner").join(
            lag_table_14d, on="timestamp", how="left"
        )

        pred_data = (
            pred_data.with_columns(
                [
                    pl.lit(sub_id).alias("substation_id").cast(pl.Int32),
                    pl.col("timestamp").dt.hour().alias("hour"),
                    pl.col("timestamp").dt.weekday().alias("day_of_week"),
                    pl.col("timestamp").dt.month().alias("month"),
                ]
            )
            .filter(pl.col("timestamp") <= forecast_end_time)
            .drop_nulls(subset=features)
        )

        if pred_data.is_empty():
            continue

        # Predict for each member
        ensemble_members = pred_data["ensemble_member"].unique().sort()
        for member in ensemble_members:
            member_data = pred_data.filter(pl.col("ensemble_member") == member).sort("timestamp")
            preds = predict(model, member_data, features)

            member_plot_df = member_data.select(["timestamp"]).with_columns(
                power_mw=preds,
                substation=pl.lit(sub_name),
                type=pl.lit("forecast"),
                member=pl.lit(str(member)),
            )
            all_plot_data.append(member_plot_df)

        # Actuals
        actual = (
            power_full.filter(
                (pl.col("timestamp") >= latest_init_time)
                & (pl.col("timestamp") <= forecast_end_time)
            )
            .select(["timestamp", "power_mw"])
            .with_columns(
                substation=pl.lit(sub_name), type=pl.lit("actual"), member=pl.lit("truth")
            )
        )
        all_plot_data.append(actual)

    if not all_plot_data:
        log.error("No data for plotting.")
        return

    plot_df = pl.concat(all_plot_data).to_pandas()

    # 3. Create Altair Chart
    log.info("Generating Altair visualization...")

    # Base chart with data
    base = (
        alt.Chart(plot_df)
        .encode(x=alt.X("timestamp:T", title="Time"), y=alt.Y("power_mw:Q", title="Power (MW)"))
        .properties(width=800, height=200)
    )

    # Ensemble lines (thin, transparent)
    ensemble_lines = (
        base.transform_filter(alt.datum.type == "forecast")
        .mark_line(strokeWidth=0.5, opacity=0.3)
        .encode(color=alt.value("skyblue"), detail="member")
    )

    # Ground truth line (thick)
    actual_line = base.transform_filter(alt.datum.type == "actual").mark_line(
        strokeWidth=2, color="black"
    )

    # Combined chart with faceting
    chart = (
        (ensemble_lines + actual_line)
        .facet(row=alt.Row("substation:N", title="Substation"))
        .properties(title=f"7-Day Global Ensemble Forecast (NWP Init: {latest_init_time})")
        .resolve_scale(y="independent")
    )

    filename = "multi_substation_altair.json"
    chart.save(filename)
    log.info(f"Altair chart saved to {filename}")

    # Also try to save as html for easier viewing
    try:
        chart.save("multi_substation_altair.html")
        log.info("Altair chart also saved to multi_substation_altair.html")
    except Exception as e:
        log.warning(f"Could not save as HTML: {e}")


if __name__ == "__main__":
    plot_multi_substation_ensemble_altair(5)
