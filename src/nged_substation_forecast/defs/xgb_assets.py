import logging
from datetime import datetime
from pathlib import Path

import dagster as dg
import polars as pl
from dagster import ResourceParam
from xgboost_forecaster import (
    DataConfig,
    XGBoostForecaster,
    get_substation_metadata,
    prepare_data_for_substation,
)

from contracts.settings import Settings

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

    model_paths = []
    for substation_number in substation_numbers:
        try:
            data_config = DataConfig(
                base_power_path=power_path,
                base_weather_path=settings.nwp_data_path / "ECMWF" / "ENS",
            )
            metadata = get_substation_metadata(data_config)

            df = prepare_data_for_substation(
                sub_number=substation_number,
                metadata=metadata,
                config=data_config,
                use_lags=True,
            )

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

            target_col = "power_mw"
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

            df = prepare_data_for_substation(
                sub_number=substation_number,
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
                    pl.lit(preds).alias("power_mw").cast(pl.Float32),
                    pl.lit(datetime.now()).alias("nwp_init_time").cast(pl.Datetime("us", "UTC")),
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
