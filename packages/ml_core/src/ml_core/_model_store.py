"""Model-artifact storage: MLflow's artifact store fronted by a local-disk cache.

Single concern: getting trained model binaries to and from MLflow, with a local cache keyed by
the (immutable) MLflow run id. This is the *minimal* MLflow integration — ``log_artifacts`` /
``download_artifacts`` wrapped around the existing ``BaseForecaster.save`` / ``load``. It does
**not** use ``mlflow.pyfunc``, the model registry, or custom flavors.

The local cache is what keeps the live service running even when MLflow is unreachable: once a
model is cached, ``load_model`` reads it straight from disk and never contacts MLflow. The cache
key is the run id, which is immutable, so a cached model never goes stale and never needs
invalidation.
"""

import tempfile
from pathlib import Path

import mlflow

from ml_core.base_forecaster import BaseForecaster

_ARTIFACT_PATH: str = "model"
"""Sub-path under an MLflow run's artifact root where the model directory is stored."""


def save_model(forecaster: BaseForecaster, run_id: str) -> None:
    """Save a trained forecaster's artifacts onto the given MLflow run.

    Writes the forecaster to a temp directory via its own ``save`` (``.ubj`` per
    ``time_series_id`` + ``meta.json``), then uploads that directory to the run's artifact store
    under ``model/``.

    Args:
        forecaster: The trained forecaster to persist.
        run_id: The MLflow run id (resolved by tag elsewhere) to attach the artifacts to.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        forecaster.save(Path(tmp_dir))
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifacts(tmp_dir, artifact_path=_ARTIFACT_PATH)


def load_model(
    run_id: str,
    forecaster_cls: type[BaseForecaster],
    cache_base_path: Path,
) -> BaseForecaster:
    """Load a trained forecaster for ``run_id``, serving from the local cache when possible.

    On a **cache hit** (``{cache_base_path}/{run_id}/model`` already exists) the model is loaded
    straight from disk, so MLflow is never contacted — this is the production-resilience path. On
    a **cache miss** the artifacts are downloaded from the run into the cache, then loaded.

    Args:
        run_id: The MLflow run id the model was saved under.
        forecaster_cls: The ``BaseForecaster`` subclass to reconstruct (its ``load`` reads the
            saved ``meta.json``).
        cache_base_path: Root of the local cache; the model lives at ``{cache_base_path}/{run_id}``.

    Returns:
        The reconstructed, trained forecaster.
    """
    run_cache_dir = cache_base_path / run_id
    model_dir = run_cache_dir / _ARTIFACT_PATH
    if not model_dir.exists():
        run_cache_dir.mkdir(parents=True, exist_ok=True)
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=_ARTIFACT_PATH,
            dst_path=str(run_cache_dir),
        )
    return forecaster_cls.load(model_dir)
