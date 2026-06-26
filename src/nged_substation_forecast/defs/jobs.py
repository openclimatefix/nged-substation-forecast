"""Manually launched Dagster jobs for experiment lifecycle management.

These are jobs (not assets) because they manage MLflow state and the ``cv_experiment_folds``
dynamic partition set rather than producing a data artifact. Their ops use ``OpExecutionContext``
(not ``AssetExecutionContext``) because they call ``context.instance.add_dynamic_partitions``,
which needs the Dagster instance.

Currently houses ``register_experiment_job`` (§4.3). ``retire_experiment_job`` (§4.3.1) is added
in a later phase.
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
            "Hydra-style key-value overrides applied on top of the base YAML's model_params,"
            " e.g. {'selected_features': ['lag_1h'], 'n_estimators': 300}."
        ),
    )
    run_mode: Literal["smoke_test", "full_cv", "register_only"] = Field(
        default="smoke_test",
        description=(
            "smoke_test: add only the earliest fold; full_cv / register_only: add every canonical"
            " fold from conf/cv/default.yaml. No assets are materialised by the job itself."
        ),
    )
    description: str = Field(default="", description="Stored as an MLflow experiment tag.")


def _resolve_forecaster_config(
    base_model_config: str,
    config_overrides: dict[str, Any],
    experiment_name: str,
) -> tuple[type[BaseForecaster], BaseForecasterConfig]:
    """Build the concrete forecaster class + config from a model YAML and overrides.

    Loads the base model YAML, applies the Hydra-style ``config_overrides`` onto its
    ``model_params``, then instantiates the ``BaseForecasterConfig`` subclass via Hydra. The
    forecaster class is resolved too, for its ``MODEL_NAME`` (used as the ``model_family`` tag).

    Args:
        base_model_config: Path relative to ``PROJECT_ROOT`` of the base model YAML.
        config_overrides: Dotted-key overrides applied onto ``model_params``.
        experiment_name: Stamped onto the resolved config's ``experiment_name`` field.

    Returns:
        A ``(forecaster_cls, forecaster_config)`` tuple.
    """
    cfg = OmegaConf.load(PROJECT_ROOT / base_model_config)
    for key, value in config_overrides.items():
        OmegaConf.update(cfg.model_params, key, value, merge=False, force_add=True)
    forecaster_cls = cast(type[BaseForecaster], hydra.utils.get_class(cfg._target_))
    forecaster_config: BaseForecasterConfig = hydra.utils.instantiate(cfg.model_params)
    forecaster_config.experiment_name = experiment_name
    return forecaster_cls, forecaster_config


def _fold_ids_for_run_mode(run_mode: str, cv_config: CvConfig) -> list[str]:
    """Return the fold ids a run mode expands to (read from the CV config, never hard-coded).

    ``smoke_test`` uses only the earliest fold; ``full_cv`` and ``register_only`` use every
    canonical fold.
    """
    if run_mode == "smoke_test":
        return [cv_config.fold_ids[0]]
    return cv_config.fold_ids


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
    """Register a new experiment: create its MLflow records and CV partition keys (§4.3)."""
    register_experiment()
