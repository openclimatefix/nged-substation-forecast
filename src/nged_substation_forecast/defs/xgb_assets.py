"""Dagster assets for XGBoost forecasting."""

import logging
from datetime import datetime
from pathlib import Path

import dagster as dg
import polars as pl
from xgboost_forecaster import (
    XGBoostForecaster,
    get_substation_metadata,
    prepare_data_for_substation,
)

from .nged_assets import substation_names_def

log = logging.getLogger(__name__)

# TODO: Configure these paths
MODEL_BASE_PATH = Path("data/models/xgboost")
FORECAST_BASE_PATH = Path("data/forecasts/xgboost")


@dg.asset(partitions_def=substation_names_def, deps=["live_primary_parquet", "ecmwf_ens_forecast"])
def xgb_model(context: dg.AssetExecutionContext) -> dg.Output[Path]:
    """Train an XGBoost model for a specific substation."""
    substation_name = context.partition_key

    metadata = get_substation_metadata()

    df = prepare_data_for_substation(
        sub_name=substation_name,
        metadata=metadata,
        use_lags=True,
    )

    if df.is_empty():
        raise ValueError(f"No data available for substation {substation_name}")

    # Train model
    forecaster = XGBoostForecaster()

    # Split into train/eval (simple temporal split)
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
    model_path = MODEL_BASE_PATH / f"{substation_name}.json"
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
def xgb_forecast(context: dg.AssetExecutionContext, xgb_model: Path) -> dg.Output[pl.DataFrame]:
    """Generate a forecast using the trained XGBoost model."""
    substation_name = context.partition_key

    # Load model
    forecaster = XGBoostForecaster.load(xgb_model)

    # Prepare inference data
    # For now, we'll just use the last few days of data to "forecast" (mocking a real forecast)
    # In reality, this would use future weather data.
    metadata = get_substation_metadata()
    df = prepare_data_for_substation(
        sub_name=substation_name,
        metadata=metadata,
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
            pl.lit("xgboost_v1.0.0").alias("power_fcst_model").cast(pl.Categorical),
        ]
    )

    # Save forecast
    forecast_path = FORECAST_BASE_PATH / f"{substation_name}.parquet"
    forecast_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.write_parquet(forecast_path)

    return dg.Output(
        forecast_df,
        metadata={
            "path": dg.MetadataValue.path(forecast_path),
            "n_points": len(forecast_df),
        },
    )
