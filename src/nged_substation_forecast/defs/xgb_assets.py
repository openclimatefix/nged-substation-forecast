import logging
from datetime import datetime
from typing import cast

import dagster as dg
import mlflow
import patito as pt
import polars as pl
from contracts.data_schemas import ProcessedNwp
from contracts.settings import Settings
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from xgboost import XGBRegressor

from xgboost_forecaster.model import XGBoostForecaster

log = logging.getLogger(__name__)


def load_hydra_config(model_name: str) -> dict:
    """Load the Hydra configuration for a specific model."""
    if not GlobalHydra.instance().is_initialized():
        initialize(version_base=None, config_path="../../../conf")
    cfg = compose(config_name="config", overrides=[f"model={model_name}"])
    return dict(cfg)


@dg.asset(
    ins={
        "scada": dg.AssetIn("combined_actuals"),
    },
    deps=["ecmwf_ens_forecast"],
    compute_kind="python",
    group_name="models",
)
def train_xgboost(
    context: dg.AssetExecutionContext,
    scada: pl.DataFrame,
    settings: dg.ResourceParam[Settings],
):
    """Train the XGBoost baseline model."""
    model_name = "xgboost_baseline"
    full_cfg = load_hydra_config(model_name)
    model_cfg = full_cfg["model"]

    # Slicing the data based on Hydra config
    train_start = full_cfg["data_split"]["train_start"]
    train_end = full_cfg["data_split"]["train_end"]

    # Filter SCADA data temporally
    scada_df = scada.filter(pl.col("timestamp").is_between(train_start, train_end)).lazy()

    # Load weather data from disk (since it's partitioned and we want a range)
    weather_path = settings.nwp_data_path / "ECMWF" / "ENS"
    weather_df = pl.scan_parquet(weather_path / "*.parquet").filter(
        pl.col("valid_time").is_between(train_start, train_end)
    )

    # Join SCADA and weather data
    joined_df = cast(
        pl.DataFrame,
        scada_df.rename({"timestamp": "valid_time", "substation_number": "substation_id"})
        .join(
            weather_df.rename({"h3_index": "substation_id"}),
            on=["valid_time", "substation_id"],
        )
        .collect(),
    )

    # Prepare features and target
    X = joined_df.select(
        pl.all().exclude(["MW", "MVA", "MVAr", "ingested_at", "valid_time", "substation_id"])
    ).to_pandas()
    y = joined_df.select("MW").to_pandas()

    # Train the model
    model = XGBRegressor(**model_cfg.get("hyperparameters", {}))
    model.fit(X, y)

    # Log using native MLflow flavor
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(full_cfg)
        mlflow.xgboost.log_model(model, artifact_path="model")

        context.add_output_metadata(
            {
                "mlflow_run_id": run.info.run_id,
                "power_fcst_model_name": model_name,
            }
        )

    return model


@dg.asset(
    ins={
        "model": dg.AssetIn("train_xgboost"),
    },
    deps=["ecmwf_ens_forecast"],
    compute_kind="python",
    group_name="models",
)
def evaluate_xgboost(
    context: dg.AssetExecutionContext,
    model: XGBRegressor,
    settings: dg.ResourceParam[Settings],
):
    """Evaluate the XGBoost baseline model and generate forecasts."""
    model_name = "xgboost_baseline"
    full_cfg = load_hydra_config(model_name)

    # Prepare inference data using the test split
    test_start = full_cfg["data_split"]["test_start"]
    test_end = full_cfg["data_split"]["test_end"]

    # Load weather data from disk
    weather_path = settings.nwp_data_path / "ECMWF" / "ENS"
    weather_test = cast(
        pt.DataFrame[ProcessedNwp],
        pl.scan_parquet(weather_path / "*.parquet")
        .filter(pl.col("valid_time").is_between(test_start, test_end))
        .collect(),
    )

    # Wrap in our Forecaster class
    forecaster = XGBoostForecaster(model)

    # Generate forecasts
    predictions_df = forecaster.predict(weather_ecmwf_ens_0_25=weather_test)

    # Add metadata for Delta Lake
    now = datetime.now()
    year_month = now.strftime("%Y-%m")

    results_df = predictions_df.with_columns(
        power_fcst_model_name=pl.lit(model_name).cast(pl.Categorical),
        power_fcst_init_time=pl.lit(now).cast(pl.Datetime("us", "UTC")),
        power_fcst_init_year_month=pl.lit(year_month).cast(pl.String),
        nwp_init_time=pl.lit(now).cast(pl.Datetime("us", "UTC")),
    )

    # Add to dynamic partitions
    context.instance.add_dynamic_partitions("model_partitions", [model_name])

    context.add_output_metadata(
        {
            "num_rows": len(results_df),
            "power_fcst_model_name": model_name,
        }
    )
    return results_df
