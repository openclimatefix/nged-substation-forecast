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

from typing import Final

import mlflow
from mlflow.tracking import MlflowClient

CV_PARENT_RUN_NAME: Final[str] = "cv_summary"
"""Run name of an experiment's parent run, which carries the aggregate leaderboard metrics."""


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
