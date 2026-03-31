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


@dg.asset(
    ins={
        "nwp": dg.AssetIn("processed_nwp_data"),
        "substation_power_flows": dg.AssetIn("combined_actuals"),
        "substation_metadata": dg.AssetIn("substation_metadata"),
        "healthy_substations": dg.AssetIn("healthy_substations"),
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
    healthy_substations: list[int],
):
    """Train the XGBoost model."""
    model_name = "xgboost"
    hydra_config = load_hydra_config(model_name)
    hydra_config = _apply_config_overrides(hydra_config, config)

    # Filter to healthy substations and optional manual selection
    if config.substation_ids:
        sub_ids = [s for s in config.substation_ids if s in healthy_substations]
    else:
        sub_ids = healthy_substations

    # If no healthy substations are found, we can either crash or return an empty model.
    # Returning an empty model allows the pipeline to continue gracefully in test
    # environments or during telemetry outages.
    if not sub_ids:
        if config.allow_empty_substations:
            context.log.warning(
                "No healthy substations available for training. Returning an untrained model."
            )
            return XGBoostForecaster()
        raise ValueError("No healthy substations available for training.")

    if config.substation_ids:
        context.log.info(f"Filtered requested substations to {len(sub_ids)} healthy ones.")

    substation_power_flows_filtered = substation_power_flows.filter(
        pl.col("substation_number").is_in(sub_ids)
    )
    substation_metadata = substation_metadata.filter(pl.col("substation_number").is_in(sub_ids))

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
        substation_metadata=substation_metadata,
    )


@dg.asset(
    ins={
        "model": dg.AssetIn("train_xgboost"),
        "nwp": dg.AssetIn("processed_nwp_data"),
        "substation_power_flows": dg.AssetIn("combined_actuals"),
        "substation_metadata": dg.AssetIn("substation_metadata"),
        "healthy_substations": dg.AssetIn("healthy_substations"),
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
    healthy_substations: list[int],
):
    """Evaluate the XGBoost model and generate forecasts."""
    model_name = "xgboost"
    hydra_config = load_hydra_config(model_name)
    hydra_config = _apply_config_overrides(hydra_config, config)

    # Filter to healthy substations and optional manual selection
    if config.substation_ids:
        sub_ids = [s for s in config.substation_ids if s in healthy_substations]
    else:
        sub_ids = healthy_substations

    # If no healthy substations are found, we can either crash or return an empty forecast.
    # Returning an empty dataframe allows the pipeline to continue gracefully in test
    # environments or during telemetry outages.
    if not sub_ids:
        if config.allow_empty_substations:
            context.log.warning(
                "No healthy substations available for evaluation. Returning an empty forecast dataframe."
            )
            return pl.DataFrame(schema=PowerForecast.dtypes)
        raise ValueError("No healthy substations available for evaluation.")

    if config.substation_ids:
        context.log.info(f"Filtered requested substations to {len(sub_ids)} healthy ones.")

    substation_power_flows_filtered = substation_power_flows.filter(
        pl.col("substation_number").is_in(sub_ids)
    )
    substation_metadata = substation_metadata.filter(pl.col("substation_number").is_in(sub_ids))

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
        substation_metadata=substation_metadata,
    )
