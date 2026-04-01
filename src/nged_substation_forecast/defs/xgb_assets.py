import dagster as dg
import polars as pl
from typing import cast
from datetime import date
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from pydantic import Field, field_validator

from contracts.hydra_schemas import NwpModel, TrainingConfig
from contracts.data_schemas import PowerForecast
from ml_core.utils import evaluate_and_save_model, train_and_log_model

from xgboost_forecaster.model import XGBoostForecaster


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
    if not GlobalHydra.instance().is_initialized():
        initialize(version_base=None, config_path="../../../conf")
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
    substation_metadata: pl.DataFrame | None = None,
) -> list[int]:
    """Intersects requested substation IDs with healthy/active ones and logs the result.

    This helper centralizes the intersection and validation logic for target substations,
    ensuring consistent behavior across training and evaluation assets.

    Strategy:
    ---------
    1. If `config.substation_ids` is provided, filter to only healthy substations.
    2. Otherwise, use `substation_metadata` as a fallback to identify active substations.
       This is more efficient than scanning the actuals data and uses the authoritative
       metadata source that reflects current network configuration.

    Args:
        config: XGBoost configuration with optional substation_ids override.
        healthy_substations: List of substations with healthy telemetry.
        context: Asset execution context for logging.
        substation_metadata: Optional SubstationMetadata DataFrame. If provided and
                             no config.substation_ids given, this is used as a fallback
                             to identify active substations (url is not null).

    Returns:
        List of substation IDs to use for training/evaluation.
    """
    if config.substation_ids:
        # Use provided substation IDs, filtered to only healthy ones
        sub_ids = [s for s in config.substation_ids if s in healthy_substations]
        context.log.info(f"Filtered requested substations to {len(sub_ids)} healthy ones.")
    else:
        # Fallback: use metadata to determine active substations
        # This is more efficient than scanning actuals data for healthy substations
        if substation_metadata is not None and len(substation_metadata) > 0:
            # Get substations with live telemetry (url is not null) from metadata
            valid_substation_ids = (
                substation_metadata.filter(pl.col("url").is_not_null())
                .get_column("substation_number")
                .unique()
            )
            # Intersect with healthy substations
            sub_ids = [s for s in valid_substation_ids if s in healthy_substations]
        else:
            # Last resort: use all healthy substations
            sub_ids = healthy_substations

        context.log.info(f"Using {len(sub_ids)} substations from metadata/healthy fallback.")

    if not sub_ids and not config.allow_empty_substations:
        raise ValueError("No healthy substations available for training/evaluation.")

    return sub_ids


@dg.asset(
    ins={
        "nwp": dg.AssetIn("processed_nwp_data"),
        "substation_power_flows": dg.AssetIn("cleaned_actuals"),
        "substation_metadata": dg.AssetIn("substation_metadata"),
    },
    compute_kind="python",
    group_name="models",
)
def train_xgboost(
    context: dg.AssetExecutionContext,
    config: XGBoostConfig,
    nwp: pl.LazyFrame,
    substation_power_flows: pl.LazyFrame,
    substation_metadata: pl.DataFrame,
):
    """Train the XGBoost model on cleaned substation data.

    This asset reads cleaned actuals data where problematic values (stuck sensors,
    insane values) have been replaced with null. The XGBoost forecaster will drop
    rows with null target values AFTER feature engineering is complete.

    Key Design Decisions:
    --------------------
    - Uses `cleaned_actuals` instead of `combined_actuals` to ensure the model
      trains only on physically plausible data points.
    - `_get_target_substations` now uses `substation_metadata` as an efficient
      fallback instead of scanning the entire actuals dataset.
    - The `healthy_substations` dependency has been removed as it's no longer needed.
    """
    model_name = "xgboost"
    hydra_config = load_hydra_config(model_name)
    hydra_config = _apply_config_overrides(hydra_config, config)

    # Collect to get a DataFrame for metadata operations
    # Note: The actuals may still have nulls; these will be dropped after feature engineering
    from typing import cast as type_cast

    substation_power_flows_df: pl.DataFrame = type_cast(
        pl.DataFrame, substation_power_flows.collect()
    )
    healthy_substations: list[int] = []
    if len(substation_power_flows_df) > 0:
        healthy_substations = (
            substation_power_flows_df.filter(pl.col("MW_or_MVA").is_not_null())
            .get_column("substation_number")
            .unique()
            .to_list()
        )

    # Filter to target substations using metadata as efficient fallback
    sub_ids = _get_target_substations(config, healthy_substations, context, substation_metadata)

    # If no substations are available, return an empty model gracefully
    if not sub_ids:
        context.log.warning(
            "No healthy substations available for training. Returning an untrained model."
        )
        return XGBoostForecaster()

    # Filter the actuals data to target substations
    substation_power_flows_filtered = substation_power_flows.filter(
        pl.col("substation_number").is_in(sub_ids)
    )
    substation_metadata_filtered = substation_metadata.filter(
        pl.col("substation_number").is_in(sub_ids)
    )

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
        substation_metadata=substation_metadata_filtered,
    )


@dg.asset(
    ins={
        "model": dg.AssetIn("train_xgboost"),
        "nwp": dg.AssetIn("processed_nwp_data"),
        "substation_power_flows": dg.AssetIn("cleaned_actuals"),
        "substation_metadata": dg.AssetIn("substation_metadata"),
    },
    compute_kind="python",
    group_name="models",
)
def evaluate_xgboost(
    context: dg.AssetExecutionContext,
    config: XGBoostConfig,
    model: XGBoostForecaster,
    nwp: pl.LazyFrame,
    substation_power_flows: pl.LazyFrame,
    substation_metadata: pl.DataFrame,
):
    """Evaluate the XGBoost model and generate forecasts.

    This asset evaluates the trained model on cleaned actuals data. The evaluation:
    - Uses `cleaned_actuals` instead of `combined_actuals` to ensure evaluation
      on physically plausible data only.
    - Uses `substation_metadata` for determining active substations instead
      of relying on the removed `healthy_substations` dependency.
    """
    model_name = "xgboost"
    hydra_config = load_hydra_config(model_name)
    hydra_config = _apply_config_overrides(hydra_config, config)

    # Collect to determine healthy substations from the actuals data
    substation_power_flows_df: pl.DataFrame = __import__("typing").cast(
        pl.DataFrame, substation_power_flows.collect()
    )
    healthy_substations: list[int] = []
    if len(substation_power_flows_df) > 0:
        healthy_substations = (
            substation_power_flows_df.filter(pl.col("MW_or_MVA").is_not_null())
            .get_column("substation_number")
            .unique()
            .to_list()
        )

    # Filter to target substations using metadata as efficient fallback
    sub_ids = _get_target_substations(config, healthy_substations, context, substation_metadata)

    # If no substations are available, return an empty forecast gracefully
    if not sub_ids:
        context.log.warning(
            "No healthy substations available for evaluation. Returning an empty forecast dataframe."
        )
        return pl.DataFrame(schema=PowerForecast.dtypes)

    # Filter the actuals data to target substations
    substation_power_flows_filtered = substation_power_flows.filter(
        pl.col("substation_number").is_in(sub_ids)
    )
    substation_metadata_filtered = substation_metadata.filter(
        pl.col("substation_number").is_in(sub_ids)
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
        substation_metadata=substation_metadata_filtered,
    )
