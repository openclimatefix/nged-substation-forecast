"""Idempotent MLflow run-resolution helpers, used by every CV/registration code path.

Single concern: resolving MLflow runs **by tag**. The CV assets (``trained_cv_model``,
``cv_power_forecasts``, ``metrics``) and ``register_experiment_job`` run in separate processes
(and, on retries, at separate times), so a live ``Run`` handle cannot be passed between them.
Instead, every code path discovers or creates the run it needs by tag and resumes it by ID.

Each helper returns an **ID string**, never an open run handle. The caller wraps the returned ID
in ``with mlflow.start_run(run_id=...)`` to log and close the run within its own process. Because
lookup is by tag, every helper is safe to call from any process and idempotent under Dagster
retries (a re-run resumes the same run rather than duplicating it).

The tracking URI is set by the caller (``mlflow.set_tracking_uri(...)``); these helpers simply
use ``mlflow`` / ``MlflowClient``, which honour whatever URI is in effect. This is what lets the
helpers run against a file-based MLflow in tests with no server.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final, cast

import hydra
import mlflow
from mlflow.tracking import MlflowClient

from ml_core.base_forecaster import BaseForecaster, BaseForecasterConfig

CV_PARENT_RUN_NAME: Final[str] = "cv_summary"
"""Run name of an experiment's parent run, which carries the aggregate leaderboard metrics."""


def load_experiment_forecaster(
    experiment_name: str,
) -> tuple[type[BaseForecaster], BaseForecasterConfig]:
    """Reconstruct the forecaster class + resolved config from an experiment's MLflow tags.

    ``register_experiment`` stamps three tags on the experiment: the config JSON, plus the
    fully-qualified import paths of the forecaster class and its config class (``forecaster_target``
    / ``config_target``). The config JSON alone carries no class identity, so both targets are
    needed to deserialise it into the correct ``BaseForecasterConfig`` subclass. The caller is
    responsible for setting the tracking URI (``mlflow.set_tracking_uri``) beforehand.

    Args:
        experiment_name: The MLflow experiment name (also the partition-key prefix).

    Returns:
        A ``(forecaster_cls, forecaster_config)`` tuple — the same pair
        ``register_experiment`` resolved, reconstructed from the stored tags.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"No MLflow experiment named {experiment_name!r}.")
    tags = experiment.tags
    forecaster_cls = cast(type[BaseForecaster], hydra.utils.get_class(tags["forecaster_target"]))
    config_cls = cast(type[BaseForecasterConfig], hydra.utils.get_class(tags["config_target"]))
    forecaster_config = config_cls.model_validate_json(tags["config"])
    return forecaster_cls, forecaster_config


def get_or_create_experiment(experiment_name: str) -> str:
    """Return the MLflow experiment id for ``experiment_name``, creating it if absent.

    This is the self-healing fallback path; the canonical creator is
    ``register_experiment_job``, which also stamps the experiment's ``config``/``description``
    tags. Created here untagged so a stray asset call cannot race the job into a tagless,
    half-registered experiment.

    Args:
        experiment_name: Human-readable, unique experiment name.

    Returns:
        The MLflow experiment id.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(experiment_name)


def get_or_create_parent_run(experiment_id: str) -> str:
    """Return the experiment's parent run id (tag ``cv_role=parent``), creating it if absent.

    The parent run holds the experiment's aggregate (leaderboard) metrics and the flattened
    config params. Resolved by tag so any process finds the same run.

    Args:
        experiment_id: The MLflow experiment id.

    Returns:
        The parent run id.
    """
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.cv_role = 'parent'",
        max_results=1,
    )
    if runs:
        return runs[0].info.run_id
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=CV_PARENT_RUN_NAME,
        tags={"cv_role": "parent"},
    ) as run:
        return run.info.run_id


def get_or_create_fold_run(experiment_id: str, parent_run_id: str, fold_id: str) -> str:
    """Return the fold's child run id (tags ``cv_role=fold, fold_id=...``), creating it if absent.

    The fold run holds that fold's per-fold params and metrics. It is created **nested** under
    the experiment's parent run, so the MLflow UI groups folds beneath ``cv_summary``. Resolved
    by ``(cv_role, fold_id)`` so any process — and any Dagster retry of the fold — finds the same
    run.

    Args:
        experiment_id: The MLflow experiment id.
        parent_run_id: The experiment's parent run id (from ``get_or_create_parent_run``).
        fold_id: The fold identifier (e.g. ``"2022"``).

    Returns:
        The fold run id.
    """
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.cv_role = 'fold' and tags.fold_id = '{fold_id}'",
        max_results=1,
    )
    if runs:
        return runs[0].info.run_id
    # Resume the parent so the new run nests beneath it (MLflow nests under the active run).
    # experiment_id must be passed explicitly: resuming a run does not switch the active
    # experiment, so without it the child would land in the default experiment ("0").
    with mlflow.start_run(run_id=parent_run_id):
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=fold_id,
            nested=True,
            tags={"cv_role": "fold", "fold_id": fold_id},
        ) as fold_run:
            return fold_run.info.run_id


@dataclass(frozen=True)
class PromotableRun:
    """One MLflow fold run (``cv_role=fold``) — a valid ``promoted_model`` promotion candidate."""

    run_id: str
    experiment_name: str
    fold_id: str
    start_time: datetime


def list_promotable_runs() -> list[PromotableRun]:
    """List every fold run (``cv_role=fold``) across all MLflow experiments, newest first.

    A read-only convenience for the ``promotable_model_runs`` asset
    (``defs/production_assets.py``), which logs this as a metadata table in the Dagster UI so a
    ``promoted_model`` promotion candidate's run id can be copy-pasted into that asset's
    launchpad rather than retyped from memory. The champion is still picked by eye off the MLflow
    leaderboard; this only lists candidates. The caller is responsible for setting the tracking
    URI (``mlflow.set_tracking_uri``) beforehand.
    """
    client = MlflowClient()
    runs = [
        PromotableRun(
            run_id=run.info.run_id,
            experiment_name=experiment.name,
            fold_id=run.data.tags.get("fold_id", "unknown"),
            start_time=datetime.fromtimestamp(run.info.start_time / 1000, tz=timezone.utc),
        )
        for experiment in client.search_experiments()
        for run in client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.cv_role = 'fold'",
            max_results=1000,
        )
    ]
    return sorted(runs, key=lambda run: run.start_time, reverse=True)
