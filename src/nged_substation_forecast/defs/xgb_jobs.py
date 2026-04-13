import logging
from collections.abc import Iterable
from datetime import date, timedelta
from typing import Any, cast

import dagster as dg
import polars as pl
from contracts.hydra_schemas import DataSplitConfig, NwpModel, TrainingConfig
from ml_core.utils import evaluate_and_save_model, train_and_log_model
from xgboost_forecaster.model import XGBoostForecaster

from .xgb_assets import load_hydra_config

log = logging.getLogger(__name__)


class CVConfig(dg.Config):
    """Configuration for cross-validation."""

    start_date: str
    end_date: str
    fold_size_months: int = 1


@dg.op(out=dg.DynamicOut())
def generate_expanding_windows(
    config: CVConfig,
) -> Iterable[dg.DynamicOutput[TrainingConfig]]:
    """Generate expanding windows for cross-validation.

    Args:
        config: Cross-validation configuration.

    Yields:
        DynamicOutput of TrainingConfig.
    """
    start = date.fromisoformat(config.start_date)
    end = date.fromisoformat(config.end_date)

    # Use polars to generate a range of dates at month intervals
    date_range = (
        pl.date_range(start, end, interval=f"{config.fold_size_months}mo", eager=True)
        .cast(pl.Date)
        .to_list()
    )

    # We need at least two dates to form a train/test pair
    if len(date_range) < 2:
        log.warning("Date range too short for cross-validation. Yielding single fold.")
        base_config = load_hydra_config("xgboost")
        base_config.data_split = DataSplitConfig(
            train_start=start,
            train_end=end,
            test_start=end,
            test_end=end + timedelta(days=30),
        )
        yield dg.DynamicOutput(base_config, mapping_key="fold_0")
        return

    for i in range(1, len(date_range)):
        train_end = date_range[i]
        # The test period is from train_end to the next date in the range (or end)
        test_end = date_range[i + 1] if i + 1 < len(date_range) else end

        base_config = load_hydra_config("xgboost")
        base_config.data_split = DataSplitConfig(
            train_start=start,
            train_end=train_end,
            test_start=train_end,
            test_end=test_end,
        )

        yield dg.DynamicOutput(
            base_config,
            mapping_key=f"fold_{i}",
        )


@dg.op
def train_cv_fold(
    context: dg.OpExecutionContext,
    config: Any,
    nwp: pl.LazyFrame,
    power_time_series: pl.LazyFrame,
) -> None:
    """Train and evaluate a single cross-validation fold.

    Args:
        context: Dagster context.
        config: Training configuration for this fold.
        nwp: Processed NWP data.
        power_time_series: Historical power flow data.
    """
    config = cast(TrainingConfig, config)
    model_name = f"xgboost_cv_fold_{config.data_split.train_end}"

    # Option A: Train on the control member (ensemble_member == 0)
    nwp_train = nwp.filter(pl.col("ensemble_member") == 0)

    # 1. Train
    model = train_and_log_model(
        context=context,
        model_name=model_name,
        trainer=XGBoostForecaster(),
        config=config,
        nwps={NwpModel.ECMWF_ENS_0_25DEG: nwp_train},
        power_time_series=power_time_series,
    )

    # 2. Evaluate (on all members)
    forecaster = XGBoostForecaster(model)
    forecaster.config = config.model

    evaluate_and_save_model(
        context=context,
        model_name=model_name,
        forecaster=forecaster,
        config=config,
        nwps={NwpModel.ECMWF_ENS_0_25DEG: nwp},
        power_time_series=power_time_series,
    )


@dg.job
def xgboost_cv_job() -> None:
    """Job to perform expanding window cross-validation for XGBoost models."""
    # Note: In a real Dagster job, we'd need to provide the assets as inputs.
    # This is a simplified version.
    pass


xgboost_integration_job = dg.define_asset_job(
    name="xgboost_integration_job",
    selection=dg.AssetSelection.assets(
        "cleaned_actuals",
        "all_nwp_data",
        "processed_nwp_data",
        "train_xgboost",
        "evaluate_xgboost",
        "forecast_vs_actual_plot",
    ),
)
