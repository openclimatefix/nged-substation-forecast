import logging
from datetime import datetime
from pathlib import Path

import dagster as dg
import polars as pl
from contracts.settings import Settings
from dagster import ResourceParam
from xgboost_forecaster import (
    DataConfig,
    EnsembleSelection,
    XGBoostForecaster,
    get_substation_metadata,
    prepare_inference_data,
    prepare_training_data,
)

log = logging.getLogger(__name__)


@dg.asset(deps=["live_primary_flows", "ecmwf_ens_forecast"])
def xgb_models(
    context: dg.AssetExecutionContext, settings: ResourceParam[Settings]
) -> dg.Output[list[Path]]:
    """Train XGBoost models for all substations."""
    power_path = settings.nged_data_path / "delta" / "live_primary_flows"
    if not power_path.exists():
        context.log.warning("No Delta table found.")
        return dg.Output([], metadata={"n_models": 0})

    substation_numbers = (
        pl.read_delta(str(power_path)).select("substation_number").unique().to_series().to_list()
    )

    context.log.info(f"Training models for {len(substation_numbers)} substations")

    data_config = DataConfig(
        base_power_path=power_path,
        base_weather_path=settings.nwp_data_path / "ECMWF" / "ENS",
    )
    metadata = get_substation_metadata(data_config)

    # We need a start and end date for training.
    # For now, let's use a fixed range.
    # TODO: This should be configurable so we can do expanding window time series cross-validation
    # (i.e. where we mimic what happens in production) and so that, when we train for production,
    # we use as much historical data as possible.
    start_date = datetime(2026, 2, 1).date()
    end_date = datetime(2026, 2, 28).date()

    df_all = prepare_training_data(
        substation_numbers=substation_numbers,
        metadata=metadata,
        start_date=start_date,
        end_date=end_date,
        config=data_config,
        selection=EnsembleSelection.MEAN,
        use_lags=True,
    )

    model_paths = []
    for substation_number in substation_numbers:
        try:
            # TODO: Move all the code in this `try` block into a new function.

            df = df_all.filter(pl.col("substation_number") == substation_number)

            if df.is_empty():
                context.log.warning(f"No data available for substation {substation_number}")
                continue

            # Train model
            forecaster = XGBoostForecaster()

            # Split into train/eval
            df = df.sort("timestamp")
            train_size = int(len(df) * 0.8)
            train_df = df.head(train_size)
            eval_df = df.tail(len(df) - train_size)

            target_col = "MW_or_MVA"
            feature_cols = [
                c for c in df.columns if c not in [target_col, "timestamp", "substation_number"]
            ]

            eval_set = [(eval_df, eval_df[target_col])]

            forecaster.train(
                df=train_df,
                target_col=target_col,
                feature_cols=feature_cols,
                eval_set=eval_set,
            )

            # Save model
            model_path = (
                settings.trained_ml_model_params_base_path
                / XGBoostForecaster.model_name_and_version()
                / f"{substation_number}.json"
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            forecaster.save(model_path)
            model_paths.append(model_path)

        except Exception as e:
            context.log.error(f"Failed to train model for {substation_number}: {e}")

    return dg.Output(
        model_paths,
        metadata={
            "n_models": len(model_paths),
            "substations": dg.MetadataValue.text(", ".join([p.stem for p in model_paths])),
        },
    )


@dg.asset(deps=["ecmwf_ens_forecast"])
def xgb_forecasts(
    context: dg.AssetExecutionContext, xgb_models: list[Path], settings: ResourceParam[Settings]
) -> dg.Output[pl.DataFrame]:
    """Generate forecasts for all substations using the trained XGBoost models."""
    all_forecasts = []
    for model_path in xgb_models:
        substation_number = int(model_path.stem)
        try:
            # Load model
            forecaster = XGBoostForecaster.load(model_path)

            data_config = DataConfig(
                base_power_path=settings.nged_data_path / "delta" / "live_primary_flows",
                base_weather_path=settings.nwp_data_path / "ECMWF" / "ENS",
            )
            metadata = get_substation_metadata(data_config)

            # TODO (p1): This MUST be changed to dynamically use the most recent NWP init!
            init_time = datetime(2026, 2, 17, 0)  # Example init time

            df = prepare_inference_data(
                substation_number=substation_number,
                init_time=init_time,
                metadata=metadata,
                config=data_config,
                use_lags=True,
            )

            if df.is_empty():
                context.log.warning(
                    f"No data available for forecasting substation {substation_number}"
                )
                continue

            # Make predictions
            preds = forecaster.predict(df)

            # Conform to PowerForecast contract
            forecast_df = df.select(
                [
                    pl.col("timestamp").alias("valid_time"),
                    pl.col("substation_number").alias("substation_id"),
                    pl.lit(preds).alias("MW_or_MVA").cast(pl.Float32),
                    pl.lit(init_time).alias("nwp_init_time").cast(pl.Datetime("us", "UTC")),
                    pl.lit(XGBoostForecaster.model_name_and_version())
                    .alias("power_fcst_model")
                    .cast(pl.Categorical),
                ]
            )
            all_forecasts.append(forecast_df)

        except Exception as e:
            context.log.error(f"Failed to generate forecast for {substation_number}: {e}")

    if not all_forecasts:
        return dg.Output(pl.DataFrame(), metadata={"n_forecasts": 0})

    final_df = pl.concat(all_forecasts)

    # Save combined forecast
    forecast_path = (
        settings.power_forecasts_data_path
        / XGBoostForecaster.model_name_and_version()
        / "all_substations.parquet"
    )
    forecast_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.write_parquet(forecast_path)

    return dg.Output(
        final_df,
        metadata={
            "path": dg.MetadataValue.path(forecast_path),
            "n_points": len(final_df),
            "n_substations": len(all_forecasts),
        },
    )
