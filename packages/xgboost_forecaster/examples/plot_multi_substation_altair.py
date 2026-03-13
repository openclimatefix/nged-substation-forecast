"""Plot ensemble forecast vs ground truth for 5 substations on one figure using Altair."""

import logging
import random
from datetime import datetime, timedelta, timezone
from typing import cast

import altair as alt
import patito as pt
import polars as pl
from contracts.data_schemas import SimplifiedSubstationFlows
from xgboost_forecaster import (
    DataConfig,
    EnsembleSelection,
    XGBoostForecaster,
    get_substation_metadata,
    load_nwp_run,
    load_substation_power,
    prepare_training_data,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


def plot_multi_substation_ensemble_altair(num_subs: int = 5):
    log.info("Fetching substation metadata...")
    config = DataConfig()
    metadata = get_substation_metadata(config)

    # Randomly select substations
    available_subs = metadata["substation_number"].to_list()
    sample_subs = random.sample(available_subs, min(num_subs, len(available_subs)))
    log.info(f"Selected substations: {sample_subs}")

    # Set common test start time and forecast window
    test_start_time = datetime(2026, 2, 17, tzinfo=timezone.utc)

    # Find latest NWP run before test period
    weather_files = sorted(config.base_weather_path.glob("*.parquet"))
    latest_init_time = None
    for f in reversed(weather_files):
        file_init_time = datetime.strptime(f.stem.replace("Z", ""), "%Y-%m-%dT%H").replace(
            tzinfo=timezone.utc
        )
        if file_init_time <= test_start_time:
            latest_init_time = file_init_time
            break

    if not latest_init_time:
        log.error("No suitable NWP initialization found.")
        return

    latest_init_time = cast(datetime, latest_init_time)
    log.info(f"Using NWP initialization: {latest_init_time}")
    forecast_end_time = latest_init_time + timedelta(days=7)

    # 1. Train ONE global model for all 5 substations
    log.info("Preparing training data for global model...")
    start_date = datetime.strptime(weather_files[0].stem[:10], "%Y-%m-%d").date()
    end_date = datetime.strptime(weather_files[-1].stem[:10], "%Y-%m-%d").date()

    all_train_data = prepare_training_data(
        substation_numbers=sample_subs,
        metadata=metadata,
        start_date=start_date,
        end_date=end_date,
        config=config,
        selection=EnsembleSelection.MEAN,
        use_lags=True,
    )

    if all_train_data.is_empty():
        log.error("Global training data is empty.")
        return

    # Split training data (before test_start_time)
    train_data = all_train_data.filter(pl.col("timestamp") < test_start_time)

    log.info(
        f"Training global model on {train_data.height} samples across {len(sample_subs)} substations..."
    )
    forecaster = XGBoostForecaster()
    forecaster.train(train_data)

    # 2. Generate forecasts for each substation and ensemble member
    all_plot_data = []

    for sub_number in sample_subs:
        log.info(f"Generating forecast for {sub_number}...")
        sub_meta = metadata.filter(pl.col("substation_number") == sub_number)
        sub_name = sub_meta["substation_name_in_location_table"][0]

        # Historical power for lags
        power_full = load_substation_power(sub_number, config)
        if power_full.is_empty():
            log.warning(f"Skipping {sub_number} due to lack of sane power data.")
            continue

        power_full = power_full.sort("timestamp")

        # Weather ensemble (ALL members)
        raw_weather = load_nwp_run(
            latest_init_time.replace(tzinfo=None), [sub_meta["h3_res_5"][0]], config
        )
        from xgboost_forecaster.data import process_weather_data, join_features

        weather_ensemble = process_weather_data(raw_weather, EnsembleSelection.ALL, config)

        # Join weather ensemble with power lags and add features
        pred_data = join_features(
            cast(pt.DataFrame[SimplifiedSubstationFlows], power_full),
            weather_ensemble,
            sub_number,
            use_lags=True,
        )
        pred_data = pred_data.filter(pl.col("timestamp") <= forecast_end_time)

        if pred_data.is_empty():
            continue

        # Predict for each member
        ensemble_members = pred_data["ensemble_member"].unique().sort()
        for member in ensemble_members:
            member_data = pred_data.filter(pl.col("ensemble_member") == member).sort("timestamp")
            preds = forecaster.predict(member_data)

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
            .select(["timestamp", "power"])
            .rename({"power": "power_mw"})
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

    base = (
        alt.Chart(plot_df)
        .encode(x=alt.X("timestamp:T", title="Time"), y=alt.Y("power_mw:Q", title="Power (MW)"))
        .properties(width=800, height=200)
    )

    ensemble_lines = (
        base.transform_filter(alt.datum.type == "forecast")
        .mark_line(strokeWidth=0.5, opacity=0.3)
        .encode(color=alt.value("skyblue"), detail="member")
    )

    actual_line = base.transform_filter(alt.datum.type == "actual").mark_line(
        strokeWidth=2, color="black"
    )

    chart = (
        (ensemble_lines + actual_line)
        .facet(row=alt.Row("substation:N", title="Substation"))
        .properties(title=f"7-Day Global Ensemble Forecast (NWP Init: {latest_init_time})")
        .resolve_scale(y="independent")
    )

    filename = "multi_substation_altair.html"
    chart.save(filename)
    log.info(f"Altair chart saved to {filename}")


if __name__ == "__main__":
    plot_multi_substation_ensemble_altair(5)
