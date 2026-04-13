import dagster as dg
import mlflow
import patito as pt
import polars as pl
from typing import cast
from datetime import date
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from pydantic import Field, field_validator

from contracts.hydra_schemas import NwpModel, TrainingConfig
from contracts.data_schemas import (
    PowerForecast,
    PowerTimeSeries,
    TimeSeriesMetadata,
)
from contracts.settings import PROJECT_ROOT, Settings
from ml_core.utils import evaluate_and_save_model, train_and_log_model
from xgboost_forecaster.model import XGBoostForecaster
from ..utils import create_dagster_type_from_patito_model, scan_delta_table

PowerForecastDagsterType = create_dagster_type_from_patito_model(PowerForecast)


class XGBoostConfig(dg.Config):
    """Configuration for the XGBoost model assets."""

    train_start: str | None = Field(
        default=None, description="Start date for training in ISO format (YYYY-MM-DD)."
    )
    train_end: str | None = Field(
        default=None, description="End date for training in ISO format (YYYY-MM-DD)."
    )
    test_start: str | None = Field(
        default=None, description="Start date for testing in ISO format (YYYY-MM-DD)."
    )
    test_end: str | None = Field(
        default=None, description="End date for testing in ISO format (YYYY-MM-DD)."
    )
    time_series_ids: list[int] | None = Field(
        default=None, description="Optional list of substation IDs to include."
    )
    allow_empty_time_series: bool = Field(
        default=True,
        description="Allow the pipeline to continue if no healthy substations are found.",
    )

    @field_validator("train_start", "train_end", "test_start", "test_end")
    @classmethod
    def validate_date_string(cls, v: str | None) -> str | None:
        """Validate that the date string is in ISO format."""
        if v is not None:
            try:
                date.fromisoformat(v)
            except ValueError as e:
                raise ValueError(
                    f"Invalid date format for '{v}'. Expected ISO format (YYYY-MM-DD)."
                ) from e
        return v


def load_hydra_config(model_name: str) -> TrainingConfig:
    """Load the Hydra configuration for a specific model."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=[f"model={model_name}"])
        cfg_dict = cast(dict, OmegaConf.to_container(cfg, resolve=True))
        return TrainingConfig(**cfg_dict)


def _apply_config_overrides(config: TrainingConfig, dg_config: XGBoostConfig) -> TrainingConfig:
    """Apply Dagster configuration overrides to the Hydra configuration."""
    if dg_config.train_start:
        config.data_split.train_start = date.fromisoformat(dg_config.train_start)
    if dg_config.train_end:
        config.data_split.train_end = date.fromisoformat(dg_config.train_end)
    if dg_config.test_start:
        config.data_split.test_start = date.fromisoformat(dg_config.test_start)
    if dg_config.test_end:
        config.data_split.test_end = date.fromisoformat(dg_config.test_end)
    return config


def _get_filtered_time_series_data(
    context: dg.AssetExecutionContext,
    config: XGBoostConfig,
    settings: dg.ResourceParam[Settings],
) -> tuple[pt.LazyFrame[PowerTimeSeries], pt.DataFrame[TimeSeriesMetadata], list[int]]:
    """Load and filter time series data and metadata lazily.

    This helper optimizes the data preparation pipeline by applying filters
    lazily and avoiding unnecessary materialization of the entire dataset.

    Args:
        context: Asset execution context.
        config: XGBoost configuration.
        settings: Global settings.

    Returns:
        A tuple containing (filtered_power_time_series, filtered_metadata, time_series_ids).
        The list of time_series_ids contains the IDs of healthy substations.
    """
    # Load time series metadata
    time_series_metadata = pl.read_parquet(
        settings.nged_data_path / "parquet" / "time_series_metadata.parquet"
    )

    # Use scan_delta_table to ensure we have the full range.
    power_time_series = scan_delta_table(
        str(settings.nged_data_path / "delta" / "cleaned_power_time_series")
    )

    # Apply substation ID filter early if provided
    if config.time_series_ids:
        power_time_series = power_time_series.filter(
            pl.col("time_series_id").is_in(config.time_series_ids)
        )

    # Compute healthy substations lazily.
    # We join with the main dataset to ensure we only keep healthy substations
    # that are also within our requested ID set (if any).
    valid_ids_lf = (
        power_time_series.filter(pl.col("power").is_not_null()).select("time_series_id").unique()
    )

    # Materialize only the list of healthy substation IDs
    time_series_ids = (
        cast(pl.DataFrame, valid_ids_lf.collect()).get_column("time_series_id").to_list()
    )

    # Filter the main dataset lazily using an inner join to push down filters
    power_time_series_filtered = power_time_series.join(
        valid_ids_lf, on="time_series_id", how="inner"
    )

    # Filter metadata to target substations
    time_series_metadata_filtered = time_series_metadata.filter(
        pl.col("time_series_id").is_in(time_series_ids)
    )

    return (
        cast(pt.LazyFrame[PowerTimeSeries], power_time_series_filtered),
        cast(pt.DataFrame[TimeSeriesMetadata], time_series_metadata_filtered),
        time_series_ids,
    )


def _prepare_xgboost_inputs(
    context: dg.AssetExecutionContext,
    config: XGBoostConfig,
    settings: dg.ResourceParam[Settings],
    model_name: str = "xgboost",
):
    """
    Centralizes data preparation and configuration loading for XGBoost assets.

    This helper function ensures consistency between training and evaluation assets
    by centralizing the logic for:
    - Setting up MLflow tracking.
    - Loading and overriding Hydra configurations.
    - Loading power time series and metadata.
    - Identifying healthy substations.
    - Filtering data to target substations.

    Returns:
        A tuple containing (hydra_config, power_time_series_filtered, time_series_metadata_filtered, time_series_ids).
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    hydra_config = load_hydra_config(model_name)
    hydra_config = _apply_config_overrides(hydra_config, config)

    # Load and filter data using the helper
    power_time_series_filtered, time_series_metadata_filtered, time_series_ids = (
        _get_filtered_time_series_data(context, config, settings)
    )

    # Validate that we have substations if required
    if not time_series_ids and not config.allow_empty_time_series:
        raise ValueError("No healthy substations available for training/evaluation.")

    return hydra_config, power_time_series_filtered, time_series_metadata_filtered, time_series_ids


@dg.asset(
    ins={
        "nwp": dg.AssetIn("processed_nwp_data"),
    },
    deps=["cleaned_power_time_series"],
    compute_kind="python",
    group_name="models",
)
def train_xgboost(
    context: dg.AssetExecutionContext,
    config: XGBoostConfig,
    settings: dg.ResourceParam[Settings],
    nwp: pl.LazyFrame,
):
    """Train the XGBoost model on cleaned substation data.

    This asset reads cleaned actuals data where problematic values (stuck sensors,
    insane values) have been replaced with null. The XGBoost forecaster will drop
    rows with null target values AFTER feature engineering is complete.

    This asset focuses on orchestrating the training process, delegating data
    preparation to `_prepare_xgboost_inputs`.
    """
    hydra_config, power_time_series_filtered, time_series_metadata_filtered, time_series_ids = (
        _prepare_xgboost_inputs(context, config, settings)
    )

    # If no substations are available, return an empty model gracefully
    if not time_series_ids:
        context.log.warning(
            "No healthy substations available for training. Returning an untrained model."
        )
        return XGBoostForecaster()

    context.log.info(f"time_series_metadata_filtered shape: {time_series_metadata_filtered.shape}")
    context.log.info(f"time_series_ids: {time_series_ids}")

    # Option A: Train on the control member (ensemble_member == 0)
    # This avoids non-linearity issues and distribution shift.
    nwp_train = nwp.filter(pl.col("ensemble_member") == 0)

    return train_and_log_model(
        context=context,
        model_name="xgboost",
        trainer=XGBoostForecaster(),
        config=hydra_config,
        nwps={NwpModel.ECMWF_ENS_0_25DEG: nwp_train},
        power_time_series=power_time_series_filtered,
        time_series_metadata=time_series_metadata_filtered,
    )


@dg.asset(
    ins={
        "model": dg.AssetIn("train_xgboost"),
        "nwp": dg.AssetIn("processed_nwp_data"),
    },
    deps=["cleaned_power_time_series"],
    compute_kind="python",
    group_name="models",
    dagster_type=PowerForecastDagsterType,
)
def evaluate_xgboost(
    context: dg.AssetExecutionContext,
    config: XGBoostConfig,
    settings: dg.ResourceParam[Settings],
    model: XGBoostForecaster,
    nwp: pl.LazyFrame,
):
    """Evaluate the XGBoost model and generate forecasts.

    This asset evaluates the trained model on cleaned actuals data. The evaluation:
    - Uses `cleaned_power_time_series` instead of `combined_actuals` to ensure evaluation
      on physically plausible data only.

    This asset focuses on orchestrating the evaluation process, delegating data
    preparation to `_prepare_xgboost_inputs`.
    """
    hydra_config, power_time_series_filtered, time_series_metadata_filtered, time_series_ids = (
        _prepare_xgboost_inputs(context, config, settings)
    )

    # If no substations are available, return an empty forecast gracefully
    if not time_series_ids:
        context.log.warning(
            "No healthy substations available for evaluation. Returning an empty forecast dataframe."
        )
        return pl.DataFrame(schema=PowerForecast.dtypes)

    # Log shapes and ensemble members for debugging
    context.log.info(f"Number of target substations: {len(time_series_ids)}")
    context.log.info(f"time_series_metadata_filtered shape: {time_series_metadata_filtered.shape}")
    num_ensemble_members = cast(
        pl.DataFrame, nwp.select("ensemble_member").unique().collect()
    ).height
    context.log.info(f"Number of NWP ensemble members: {num_ensemble_members}")

    # The Dagster asset now returns the full XGBoostForecaster instance
    forecaster = model
    forecaster.config = hydra_config.model

    context.log.info("Starting evaluation...")
    result = evaluate_and_save_model(
        context=context,
        model_name="xgboost",
        forecaster=forecaster,
        config=hydra_config,
        nwps={NwpModel.ECMWF_ENS_0_25DEG: nwp},
        power_time_series=power_time_series_filtered,
        time_series_metadata=time_series_metadata_filtered,
    )
    context.log.info("Evaluation complete.")
    return result
