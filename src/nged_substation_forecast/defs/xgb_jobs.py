import logging
from collections.abc import Iterable
from datetime import date

import dagster as dg
import mlflow
import polars as pl
from contracts.settings import Settings
from mlflow.models import ModelSignature
from mlflow.types import ParamSchema, ParamSpec
from xgboost_forecaster import XGBoostPyFuncWrapper

from .xgb_assets import XGBoostTrainingParams, train_xgboost_models_for_range

log = logging.getLogger(__name__)


class CVConfig(dg.Config):
    """Configuration for cross-validation."""

    start_date: str
    end_date: str
    fold_size_months: int = 1


@dg.op(out=dg.DynamicOut())
def generate_expanding_windows(
    config: CVConfig,
) -> Iterable[dg.DynamicOutput[XGBoostTrainingParams]]:
    """Generate expanding windows for cross-validation.

    Args:
        config: Cross-validation configuration.

    Yields:
        DynamicOutput of XGBoostTrainingParams.
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
        yield dg.DynamicOutput(
            XGBoostTrainingParams(
                train_start_date=start.isoformat(),
                train_end_date=end.isoformat(),
            ),
            mapping_key="fold_0",
        )
        return

    for i in range(1, len(date_range)):
        train_end = date_range[i]
        # The test period is from train_end to the next date in the range (or end)
        test_end = date_range[i + 1] if i + 1 < len(date_range) else end

        fold_params = XGBoostTrainingParams(
            train_start_date=start.isoformat(),
            train_end_date=train_end.isoformat(),
            test_end_date=test_end.isoformat(),
        )

        yield dg.DynamicOutput(
            fold_params,
            mapping_key=f"fold_{i}",
        )


@dg.op
def train_cv_fold(
    context: dg.OpExecutionContext,
    params: XGBoostTrainingParams,
    settings: dg.ResourceParam[Settings],
) -> None:
    """Train a single cross-validation fold and log to MLflow.

    Args:
        context: Dagster context.
        params: Training parameters for this fold.
        settings: Application settings.
    """
    # We use a nested MLflow run to group folds together
    with mlflow.start_run(
        run_name=f"fold_{params.train_end_date}",
        nested=True,
    ):
        mlflow.log_params(
            {
                "train_start_date": params.train_start_date,
                "train_end_date": params.train_end_date,
                "test_end_date": params.test_end_date,
            }
        )

        artifacts = train_xgboost_models_for_range(context, params, settings)

        if not artifacts:
            context.log.warning(f"No models trained for fold ending {params.train_end_date}")
            return

        signature = ModelSignature(
            inputs=None,
            outputs=None,
            params=ParamSchema(
                [
                    ParamSpec(name="nwp_init_time", dtype="datetime", default=None),
                    ParamSpec(name="power_fcst_model", dtype="string", default=None),
                ]
            ),
        )

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=XGBoostPyFuncWrapper(),
            artifacts=artifacts,
            signature=signature,
        )


@dg.job
def xgboost_cv_job() -> None:
    """Job to perform expanding window cross-validation for XGBoost models."""
    generate_expanding_windows().map(train_cv_fold)
