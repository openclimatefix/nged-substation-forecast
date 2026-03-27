import dagster as dg
import polars as pl
from typing import cast
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from contracts.hydra_schemas import NwpModel, TrainingConfig
from ml_core.utils import evaluate_and_save_model, train_and_log_model

from xgboost_forecaster.model import XGBoostForecaster


def load_hydra_config(model_name: str) -> TrainingConfig:
    """Load the Hydra configuration for a specific model."""
    if not GlobalHydra.instance().is_initialized():
        initialize(version_base=None, config_path="../../../conf")
    cfg = compose(config_name="config", overrides=[f"model={model_name}"])
    cfg_dict = cast(dict, OmegaConf.to_container(cfg, resolve=True))
    return TrainingConfig(**cfg_dict)


@dg.asset(
    ins={
        "nwp": dg.AssetIn("processed_nwp_data"),
        "substation_power_flows": dg.AssetIn("combined_actuals"),
        "substation_metadata": dg.AssetIn("substation_metadata"),
    },
    compute_kind="python",
    group_name="models",
)
def train_xgboost(
    context: dg.AssetExecutionContext,
    nwp: pl.LazyFrame,
    substation_power_flows: pl.LazyFrame,
    substation_metadata: pl.DataFrame,
):
    """Train the XGBoost model."""
    model_name = "xgboost"
    config = load_hydra_config(model_name)

    # Option A: Train on the control member (ensemble_member == 0)
    # This avoids non-linearity issues and distribution shift.
    nwp_train = nwp.filter(pl.col("ensemble_member") == 0)

    return train_and_log_model(
        context=context,
        model_name=model_name,
        trainer=XGBoostForecaster(),
        config=config,
        nwps={NwpModel.ECMWF_ENS_0_25DEG: nwp_train},
        substation_power_flows=substation_power_flows,
        substation_metadata=substation_metadata,
    )


@dg.asset(
    ins={
        "model": dg.AssetIn("train_xgboost"),
        "nwp": dg.AssetIn("processed_nwp_data"),
        "substation_power_flows": dg.AssetIn("combined_actuals"),
        "substation_metadata": dg.AssetIn("substation_metadata"),
    },
    compute_kind="python",
    group_name="models",
)
def evaluate_xgboost(
    context: dg.AssetExecutionContext,
    model,
    nwp: pl.LazyFrame,
    substation_power_flows: pl.LazyFrame,
    substation_metadata: pl.DataFrame,
):
    """Evaluate the XGBoost model and generate forecasts."""
    model_name = "xgboost"
    config = load_hydra_config(model_name)

    # Note: We must inject config into the forecaster so it can build features
    forecaster = XGBoostForecaster(model)
    forecaster.config = config

    return evaluate_and_save_model(
        context=context,
        model_name=model_name,
        forecaster=forecaster,
        config=config,
        nwps={NwpModel.ECMWF_ENS_0_25DEG: nwp},
        substation_power_flows=substation_power_flows,
        substation_metadata=substation_metadata,
    )
