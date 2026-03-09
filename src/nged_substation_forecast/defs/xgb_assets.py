import logging
from datetime import datetime
from pathlib import Path

import dagster as dg
import polars as pl
from xgboost_forecaster import (
    DataConfig,
    XGBoostForecaster,
    get_substation_metadata,
    prepare_data_for_substation,
)

from nged_substation_forecast.config_resource import NgedConfig

from .nged_assets import substation_names_def

log = logging.getLogger(__name__)


@dg.asset(partitions_def=substation_names_def, deps=["live_primary_parquet", "ecmwf_ens_forecast"])
def xgb_model(context: dg.AssetExecutionContext, config: NgedConfig) -> dg.Output[Path]:
    """Train an XGBoost model for a specific substation."""
    settings = config.to_settings()
    substation_name = context.partition_key

    data_config = DataConfig(
        base_power_path=settings.NGED_DATA_PATH / "parquet" / "live_primary_flows",
        base_weather_path=settings.NWP_DATA_PATH / "ECMWF" / "ENS",
    )
    metadata = get_substation_metadata(data_config)

    df = prepare_data_for_substation(
        sub_name=substation_name,
        metadata=metadata,
        config=data_config,
        use_lags=True,
    )

    if df.is_empty():
        raise ValueError(f"No data available for substation {substation_name}")

    # Train model
    forecaster = XGBoostForecaster()

    # Split into train/eval
    df = df.sort("timestamp")
    train_size = int(len(df) * 0.8)
    train_df = df.head(train_size)
    eval_df = df.tail(len(df) - train_size)

    target_col = "power_mw"
    feature_cols = [
        c
        for c in df.columns
        if c not in [target_col, "timestamp", "substation_name", "substation_id"]
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
        settings.TRAINED_ML_MODEL_PARAMS_BASE_PATH
        / XGBoostForecaster.model_name_and_version()
        / f"{substation_name}.json"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    forecaster.save(model_path)

    importance_df = forecaster.get_feature_importance()
    context.log.info(f"Top features for {substation_name}: {importance_df.head(5)}")

    return dg.Output(
        model_path,
        metadata={
            "path": dg.MetadataValue.path(model_path),
            "n_rows": len(df),
            "top_features": dg.MetadataValue.text(str(importance_df.head(5).to_dicts())),
        },
    )


@dg.asset(partitions_def=substation_names_def, deps=["ecmwf_ens_forecast"])
def xgb_forecast(
    context: dg.AssetExecutionContext, xgb_model: Path, config: NgedConfig
) -> dg.Output[pl.DataFrame]:
    """Generate a forecast using the trained XGBoost model."""
    settings = config.to_settings()
    substation_name = context.partition_key

    # Load model
    forecaster = XGBoostForecaster.load(xgb_model)

    data_config = DataConfig(
        base_power_path=settings.NGED_DATA_PATH / "parquet" / "live_primary_flows",
        base_weather_path=settings.NWP_DATA_PATH / "ECMWF" / "ENS",
    )
    metadata = get_substation_metadata(data_config)

    df = prepare_data_for_substation(
        sub_name=substation_name,
        metadata=metadata,
        config=data_config,
        use_lags=True,
    )

    if df.is_empty():
        raise ValueError(f"No data available for forecasting substation {substation_name}")

    # Make predictions
    preds = forecaster.predict(df)

    # Conform to PowerForecast contract
    forecast_df = df.select(
        [
            pl.col("timestamp").alias("valid_time"),
            pl.col("substation_id"),
            pl.lit(preds).alias("power_mw").cast(pl.Float32),
            pl.lit(datetime.now()).alias("nwp_init_time").cast(pl.Datetime("us", "UTC")),
            pl.lit(XGBoostForecaster.model_name_and_version())
            .alias("power_fcst_model")
            .cast(pl.Categorical),
        ]
    )

    # Save forecast
    forecast_path = (
        settings.POWER_FORECASTS_DATA_PATH
        / XGBoostForecaster.model_name_and_version()
        / f"{substation_name}.parquet"
    )
    forecast_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.write_parquet(forecast_path)

    return dg.Output(
        forecast_df,
        metadata={
            "path": dg.MetadataValue.path(forecast_path),
            "n_points": len(forecast_df),
        },
    )
