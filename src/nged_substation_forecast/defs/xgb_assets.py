import dagster as dg
import mlflow
import polars as pl
from typing import cast
from datetime import date
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from pydantic import Field, field_validator

from contracts.hydra_schemas import NwpModel, TrainingConfig
from contracts.data_schemas import PowerForecast
from contracts.settings import PROJECT_ROOT, Settings
from ml_core.utils import evaluate_and_save_model, train_and_log_model
from xgboost_forecaster.model import XGBoostForecaster

from .data_cleaning_assets import get_cleaned_actuals_lazy


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
    substation_ids: list[int] | None = Field(
        default=None, description="Optional list of substation IDs to include."
    )
    allow_empty_substations: bool = Field(
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


def _get_target_substations(
    config: XGBoostConfig,
    healthy_substations: list[int],
    context: dg.AssetExecutionContext,
) -> list[int]:
    """Intersects requested substation IDs with healthy/active ones and logs the result.

    This helper centralizes the intersection and validation logic for target substations,
    ensuring consistent behavior across training and evaluation assets.

    Strategy:
    ---------
    1. If `config.substation_ids` is provided, filter to only healthy substations.
    2. Otherwise, use all healthy substations.

    Args:
        config: XGBoost configuration with optional substation_ids override.
        healthy_substations: List of substations with healthy telemetry.
        context: Asset execution context for logging.

    Returns:
        List of substation IDs to use for training/evaluation.
    """
    if config.substation_ids:
        # Use provided substation IDs, filtered to only healthy ones
        sub_ids = [s for s in config.substation_ids if s in healthy_substations]
        context.log.info(f"Filtered requested substations to {len(sub_ids)} healthy ones.")
    else:
        # Fallback: use all healthy substations
        sub_ids = healthy_substations
        context.log.info(f"Using {len(sub_ids)} healthy substations.")

    if not sub_ids and not config.allow_empty_substations:
        raise ValueError("No healthy substations available for training/evaluation.")

    return sub_ids


@dg.asset(
    ins={
        "nwp": dg.AssetIn("processed_nwp_data"),
    },
    deps=["cleaned_actuals"],
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

    Key Design Decisions:
    --------------------
    - Uses `cleaned_actuals` instead of `combined_actuals` to ensure the model
      trains only on physically plausible data points.
    - `_get_target_substations` now uses `time_series_metadata` as an efficient
      fallback instead of scanning the entire actuals dataset.
    - The `healthy_substations` dependency has been removed as it's no longer needed.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model_name = "xgboost"
    hydra_config = load_hydra_config(model_name)
    hydra_config = _apply_config_overrides(hydra_config, config)

    # Load time series metadata
    time_series_metadata = pl.read_parquet(
        settings.nged_data_path / "parquet" / "time_series_metadata.parquet"
    )

    # Use get_cleaned_actuals_lazy to ensure we have the full training range.
    # This function serves as the single source of truth for accessing cleaned actuals.
    substation_power_flows = get_cleaned_actuals_lazy(settings, context)

    # Identify healthy substations using lazy evaluation to avoid massive eager collection
    # Note: The actuals may still have nulls; these will be dropped after feature engineering
    healthy_substations = (
        cast(
            pl.DataFrame,
            substation_power_flows.filter(pl.col("power").is_not_null())
            .select("time_series_id")
            .unique()
            .collect(),
        )
        .get_column("time_series_id")
        .to_list()
    )

    # Filter to target substations using metadata as efficient fallback
    sub_ids = _get_target_substations(config, healthy_substations, context)

    # If no substations are available, return an empty model gracefully
    if not sub_ids:
        context.log.warning(
            "No healthy substations available for training. Returning an untrained model."
        )
        return XGBoostForecaster()

    # Filter the actuals data to target substations
    substation_power_flows_filtered = substation_power_flows.filter(
        pl.col("time_series_id").is_in(sub_ids)
    )
    # context.log.info(f"substation_power_flows_filtered: {substation_power_flows_filtered}")

    # Filter metadata to target substations
    time_series_metadata_filtered = time_series_metadata.filter(
        pl.col("time_series_id").is_in(sub_ids)
    )
    context.log.info(f"time_series_metadata_filtered shape: {time_series_metadata_filtered.shape}")
    context.log.info(f"sub_ids: {sub_ids}")

    # Option A: Train on the control member (ensemble_member == 0)
    # This avoids non-linearity issues and distribution shift.
    nwp_train = nwp.filter(pl.col("ensemble_member") == 0)

    return train_and_log_model(
        context=context,
        model_name=model_name,
        trainer=XGBoostForecaster(),
        config=hydra_config,
        nwps={NwpModel.ECMWF_ENS_0_25DEG: nwp_train},
        substation_power_flows=substation_power_flows_filtered,
        time_series_metadata=time_series_metadata_filtered,
    )


@dg.asset(
    ins={
        "model": dg.AssetIn("train_xgboost"),
        "nwp": dg.AssetIn("processed_nwp_data"),
    },
    deps=["cleaned_actuals"],
    compute_kind="python",
    group_name="models",
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
    - Uses `cleaned_actuals` instead of `combined_actuals` to ensure evaluation
      on physically plausible data only.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model_name = "xgboost"
    hydra_config = load_hydra_config(model_name)
    hydra_config = _apply_config_overrides(hydra_config, config)

    # Use get_cleaned_actuals_lazy to ensure we have the full evaluation range.
    # This function serves as the single source of truth for accessing cleaned actuals.
    substation_power_flows = get_cleaned_actuals_lazy(settings, context)

    # Identify healthy substations using lazy evaluation to avoid massive eager collection
    healthy_substations = (
        cast(
            pl.DataFrame,
            substation_power_flows.filter(pl.col("power").is_not_null())
            .select("time_series_id")
            .unique()
            .collect(),
        )
        .get_column("time_series_id")
        .to_list()
    )

    # Filter to target substations using metadata as efficient fallback
    sub_ids = _get_target_substations(config, healthy_substations, context)

    # Load time series metadata
    time_series_metadata = pl.read_parquet(
        settings.nged_data_path / "parquet" / "time_series_metadata.parquet"
    )

    # If no substations are available, return an empty forecast gracefully
    if not sub_ids:
        context.log.warning(
            "No healthy substations available for evaluation. Returning an empty forecast dataframe."
        )
        return pl.DataFrame(schema=PowerForecast.dtypes)

    # Filter the actuals data to target substations
    substation_power_flows_filtered = substation_power_flows.filter(
        pl.col("time_series_id").is_in(sub_ids)
    )

    # Filter metadata to target substations
    time_series_metadata_filtered = time_series_metadata.filter(
        pl.col("time_series_id").is_in(sub_ids)
    )

    # The Dagster asset now returns the full XGBoostForecaster instance
    forecaster = model
    forecaster.config = hydra_config.model

    return evaluate_and_save_model(
        context=context,
        model_name=model_name,
        forecaster=forecaster,
        config=hydra_config,
        nwps={NwpModel.ECMWF_ENS_0_25DEG: nwp},
        substation_power_flows=substation_power_flows_filtered,
        time_series_metadata=time_series_metadata_filtered,
    )
