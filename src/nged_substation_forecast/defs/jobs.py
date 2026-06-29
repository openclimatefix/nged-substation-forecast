"""Manually launched Dagster jobs for experiment lifecycle management.

These are jobs (not assets) because they manage MLflow state and the ``cv_experiment_folds``
dynamic partition set rather than producing a data artifact. Their ops use ``OpExecutionContext``
(not ``AssetExecutionContext``) because they call ``context.instance.add_dynamic_partitions``,
which needs the Dagster instance.
"""

from typing import Any, Literal, cast

import hydra
import mlflow
from contracts.hydra_schemas import CvConfig, load_cv_config
from contracts.settings import PROJECT_ROOT, Settings
from dagster import Config, OpExecutionContext, job, op
from ml_core._cv_helpers import CV_PARTITION_KEY_SEPARATOR, flatten_config
from ml_core._mlflow_runs import get_or_create_experiment, get_or_create_parent_run
from ml_core.base_forecaster import BaseForecaster, BaseForecasterConfig
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf
from pydantic import Field

from nged_substation_forecast.defs.cv_assets import CV_EXPERIMENT_FOLDS_NAME


class RegisterExperimentConfig(Config):
    """Run config for ``register_experiment_job``."""

    experiment_name: str = Field(
        description=(
            "Human-readable, unique name; becomes the MLflow experiment name and the partition"
            " key prefix. For sweeps, generate programmatically, e.g. 'xgboost_lr0p01_depth3'."
        ),
    )
    base_model_config: str = Field(
        description="Path relative to PROJECT_ROOT, e.g. 'conf/model/xgboost.yaml'.",
    )
    config_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Key-value overrides merged onto the base YAML's model_params,"
            " e.g. {'selected_features': ['lag_1h'], 'n_estimators': 300}."
        ),
    )
    run_mode: Literal["smoke_test", "full_cv", "register_only"] = Field(
        default="smoke_test",
        description=(
            "smoke_test: add the non-leaderboard dev folds; full_cv / register_only: add the"
            " leaderboard folds from conf/cv/default.yaml. No assets are materialised by the job"
            " itself."
        ),
    )
    description: str = Field(default="", description="Stored as an MLflow experiment tag.")


def _resolve_forecaster_config(
    base_model_config: str,
    config_overrides: dict[str, Any],
    experiment_name: str,
) -> tuple[type[BaseForecaster], BaseForecasterConfig]:
    """Build the concrete forecaster class + config from a model YAML and overrides.

    Loads the base model YAML, merges ``config_overrides`` onto its ``model_params``, then
    instantiates the ``BaseForecasterConfig`` subclass via Hydra. The forecaster class is resolved
    too, for its ``MODEL_NAME`` (used as the ``model_family`` tag).

    Args:
        base_model_config: Path relative to ``PROJECT_ROOT`` of the base model YAML.
        config_overrides: Overrides merged onto ``model_params`` (whole-value replacement; lists
            are replaced, not extended).
        experiment_name: Stamped onto the resolved config's ``experiment_name`` field.

    Returns:
        A ``(forecaster_cls, forecaster_config)`` tuple.
    """
    cfg = OmegaConf.merge(
        OmegaConf.load(PROJECT_ROOT / base_model_config),
        {"model_params": config_overrides},
    )
    forecaster_cls = cast(type[BaseForecaster], hydra.utils.get_class(cfg._target_))
    forecaster_config: BaseForecasterConfig = hydra.utils.instantiate(cfg.model_params)
    forecaster_config.experiment_name = experiment_name
    return forecaster_cls, forecaster_config


def _class_target(obj: type | object) -> str:
    """Return the fully-qualified import path of a class (or an instance's class).

    Used to stamp the forecaster and config class identity onto the MLflow experiment so assets
    can reconstruct them later via ``hydra.utils.get_class`` (see ``load_experiment_forecaster``).
    """
    cls = obj if isinstance(obj, type) else type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _fold_ids_for_run_mode(run_mode: str, cv_config: CvConfig) -> list[str]:
    """Return the fold ids a run mode expands to (read from the CV config, never hard-coded).

    Selection keys off the ``leaderboard`` flag, not fold position: ``smoke_test`` uses the
    non-leaderboard dev folds; ``full_cv`` and ``register_only`` use the leaderboard folds.
    """
    if run_mode == "smoke_test":
        return [fold.fold_id for fold in cv_config.folds if not fold.leaderboard]
    return cv_config.leaderboard_fold_ids


@op
def register_experiment(context: OpExecutionContext, config: RegisterExperimentConfig) -> None:
    """Create the MLflow experiment + parent run and add the experiment's CV partition keys.

    Idempotent: re-running with the same ``experiment_name`` resolves the existing experiment,
    parent run, and partition keys rather than duplicating them. Materialises no assets — the
    user materialises ``trained_cv_model`` / ``cv_power_forecasts`` for the new partitions
    afterwards.
    """
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    forecaster_cls, forecaster_config = _resolve_forecaster_config(
        config.base_model_config, config.config_overrides, config.experiment_name
    )

    experiment_id = get_or_create_experiment(config.experiment_name)
    client = MlflowClient()
    client.set_experiment_tag(experiment_id, "config", forecaster_config.model_dump_json())
    client.set_experiment_tag(experiment_id, "description", config.description)
    # Stamp class identity so assets can reconstruct the exact forecaster + config subclass from
    # MLflow alone (the config JSON above carries no class identifier). See
    # load_experiment_forecaster.
    client.set_experiment_tag(experiment_id, "forecaster_target", _class_target(forecaster_cls))
    client.set_experiment_tag(experiment_id, "config_target", _class_target(forecaster_config))

    parent_run_id = get_or_create_parent_run(experiment_id)
    with mlflow.start_run(run_id=parent_run_id):
        mlflow.log_params(flatten_config(forecaster_config))
        mlflow.set_tags(
            {
                "model_family": forecaster_cls.MODEL_NAME,
                "weather_source": forecaster_config.weather_source,
            }
        )

    cv_config = load_cv_config(settings.cv_config_path)
    fold_ids = _fold_ids_for_run_mode(config.run_mode, cv_config)
    partition_keys = [
        f"{config.experiment_name}{CV_PARTITION_KEY_SEPARATOR}{fold_id}" for fold_id in fold_ids
    ]
    context.instance.add_dynamic_partitions(CV_EXPERIMENT_FOLDS_NAME, partition_keys)

    context.add_output_metadata(
        {
            "experiment_name": config.experiment_name,
            "mlflow_experiment_id": experiment_id,
            "run_mode": config.run_mode,
            "n_partition_keys": len(partition_keys),
            "partition_keys": str(partition_keys),
        }
    )


@job
def register_experiment_job() -> None:
    """Register a new experiment: create its MLflow records and CV partition keys."""
    register_experiment()
