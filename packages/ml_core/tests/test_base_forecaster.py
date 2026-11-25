"""Tests for the MLflow model-artifact plumbing, against file-based MLflow.

Covers ``BaseForecaster.save_to_mlflow``/``load_from_mlflow`` and the frozen metadata copy that
travels in the same archive — they are tested together here because they share one on-the-wire
format and one fake forecaster.

That fake (rather than a concrete model) keeps the tests focused on the shared behaviour —
archive upload/download — and free of any model-library dependency. It writes one file per
``time_series_id``, which is what makes a *shrinking* population observable as files that must
vanish from the run.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import ml_core.base_forecaster
import mlflow
import patito as pt
import polars as pl
import pytest
from contracts.config_schemas import class_target
from contracts.ml_schemas import AllFeatures
from contracts.power_schemas import PowerForecast, TimeSeriesMetadata
from ml_core.base_forecaster import (
    TRAINED_METADATA_FILENAME,
    BaseForecaster,
    BaseForecasterConfig,
    _download_and_unpack_model,
    load_trained_metadata,
)
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

pytestmark = pytest.mark.integration


class _FakeForecaster(BaseForecaster):
    """Minimal forecaster: a string payload plus one file per trained ``time_series_id``."""

    MODEL_NAME = "fake"
    CONFIG_CLASS = BaseForecasterConfig

    def __init__(
        self,
        model_params: BaseForecasterConfig,
        payload: str = "",
        series: Sequence[int] = (),
    ) -> None:
        super().__init__(model_params)
        self.payload = payload
        self._series = sorted(series)

    @property
    def trained_time_series_ids(self) -> list[int]:
        return self._series

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "payload.txt").write_text(self.payload)
        (path / "meta.json").write_text(
            json.dumps(
                {
                    "model_params": self.model_params.model_dump(mode="json"),
                    "trained_time_series_ids": self._series,
                    "model_class": class_target(self),
                }
            )
        )
        for ts_id in self._series:
            (path / f"{ts_id}.part").write_text(f"model for {ts_id}")

    @classmethod
    def load(cls, path: Path) -> Self:
        meta = json.loads((path / "meta.json").read_text())
        return cls(
            cls.CONFIG_CLASS.model_validate(meta["model_params"]),
            payload=(path / "payload.txt").read_text(),
            series=meta["trained_time_series_ids"],
        )

    def train(
        self, data: pt.LazyFrame[AllFeatures], time_series_ids: list[int]
    ) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    def predict(
        self, data: pt.LazyFrame[AllFeatures], *, fold_id: str = "live"
    ) -> pt.DataFrame[PowerForecast]:
        raise NotImplementedError  # pragma: no cover - unused


def test_trained_time_series_ids_is_abstract() -> None:
    """A subclass that omits ``trained_time_series_ids`` cannot be instantiated."""

    class _MissingPopulation(BaseForecaster):
        MODEL_NAME = "missing"

        def save(self, path: Path) -> None: ...

        @classmethod
        def load(cls, path: Path) -> Self:
            raise NotImplementedError

        def train(self, data: pt.LazyFrame[AllFeatures], time_series_ids: list[int]) -> None: ...

        def predict(
            self, data: pt.LazyFrame[AllFeatures], *, fold_id: str = "live"
        ) -> pt.DataFrame[PowerForecast]:
            raise NotImplementedError

    with pytest.raises(TypeError):
        # Deliberately instantiate an abstract class to check the runtime contract.
        _MissingPopulation(BaseForecasterConfig(selected_features=set()))  # ty: ignore[abstract-class-instantiation]


def _metadata_for(
    series: Sequence[int], area_wkt: bool = False
) -> pt.DataFrame[TimeSeriesMetadata]:
    """A metadata table for ``series``, carrying only the columns any consumer reads.

    ``set_model`` rather than ``validate``, as the assets themselves do, so a partial frame is
    enough. ``area_wkt`` adds the one column ``write_trained_metadata`` is expected to drop.
    """
    frame = pl.DataFrame(
        {
            "time_series_id": pl.Series(series, dtype=pl.Int32),
            "h3_res_5": pl.Series([100 + i for i in range(len(series))], dtype=pl.UInt64),
        }
    )
    if area_wkt:
        frame = frame.with_columns(area_wkt=pl.lit("POLYGON((0 0,1 1,1 0,0 0))"))
    return pt.DataFrame(frame).set_model(TimeSeriesMetadata)


def _save(
    run_id: str,
    payload: str,
    series: Sequence[int] = (),
    selected_features: set[str] | None = None,
) -> None:
    """Save a ``_FakeForecaster`` carrying ``payload``, ``series`` and ``selected_features``.

    ``selected_features`` defaults to empty, which every version of the code parses; pass a
    retired feature name to build a run this code can no longer serve.
    """
    forecaster = _FakeForecaster(
        BaseForecasterConfig(selected_features=selected_features or set()),
        payload=payload,
        series=series,
    )
    forecaster.save_to_mlflow(run_id, time_series_metadata=_metadata_for(series))


def _artifact_file_paths(run_id: str) -> list[str]:
    """Every artifact *file* reachable in the run, recursively, as artifact-root-relative paths."""
    client = MlflowClient()
    paths: list[str] = []
    dirs_to_walk = [""]
    while dirs_to_walk:
        for info in client.list_artifacts(run_id, dirs_to_walk.pop()):
            if info.is_dir:
                dirs_to_walk.append(info.path)
            else:
                paths.append(info.path)
    return sorted(paths)


@pytest.fixture
def saved_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A file-based MLflow run id with a saved _FakeForecaster (payload 'hello-model')."""
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(f"file://{tmp_path / 'mlruns'}")
    experiment_id = mlflow.create_experiment("base_forecaster_test")
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
    _save(run_id, "hello-model", series=[10, 20, 30])
    return run_id


def test_save_load_round_trip(saved_run: str) -> None:
    loaded = _FakeForecaster.load_from_mlflow(saved_run)
    assert loaded.payload == "hello-model"
    assert loaded.trained_time_series_ids == [10, 20, 30]


def test_the_model_is_stored_as_a_single_archive_artifact(saved_run: str) -> None:
    """The run holds exactly one model file — the archive — not a directory of model files.

    This is the property that makes a re-upload replace rather than merge (issue #470): MLflow
    overwrites a same-name single artifact but merges a directory upload.

    The assertion covers the run's *whole* artifact listing, which is deliberately stricter than
    the invariant: nothing else logs artifacts to a fold run today, and a second artifact turning
    up here is worth a deliberate look rather than a silent pass, since any *per-model* file
    logged alongside the archive would reopen the merge problem for that file.
    """
    assert _artifact_file_paths(saved_run) == ["model.tar.gz"]


def test_re_saving_to_the_same_run_is_reflected_on_the_next_load(saved_run: str) -> None:
    """Re-saving into a reused run must be what the next load returns (issue #197).

    CV fold runs are reused across re-materialisations, so the same ``run_id`` can hold a
    different model after re-training. ``load_from_mlflow`` has no local cache (issue #469
    removed it — see ``load_from_mlflow``'s docstring), so this is really a round-trip test, but
    it is worth keeping explicit: a future re-introduction of caching must not silently
    reintroduce the staleness bug this guards against.
    """
    assert _FakeForecaster.load_from_mlflow(saved_run).payload == "hello-model"

    # Re-train into the *same* run, exactly as a re-materialised fold does.
    _save(saved_run, "retrained-model", series=[10, 20, 30])

    assert _FakeForecaster.load_from_mlflow(saved_run).payload == "retrained-model"


def test_re_saving_a_smaller_model_leaves_no_trace_of_the_dropped_series(
    saved_run: str, tmp_path: Path
) -> None:
    """Re-training a reused run on a smaller population must not leave the dropped series behind.

    The regression this issue exists to prevent (issue #470). A directory upload
    (``log_artifacts``) *merges*, and MLflow has no public artifact-delete API, so series 30's
    file would survive in the run forever, downloaded on every subsequent load. Uploading one
    same-name archive replaces it instead.
    """
    _save(saved_run, "smaller-model", series=[10, 20])

    # Nothing in the run's artifact store mentions the dropped series...
    assert _artifact_file_paths(saved_run) == ["model.tar.gz"]

    # ...and nothing the download path materialises does either.
    assert _FakeForecaster.load_from_mlflow(saved_run).trained_time_series_ids == [10, 20]

    dest = _download_and_unpack_model(run_id=saved_run, work_dir=tmp_path, remedy="")
    assert sorted(p.name for p in dest.iterdir()) == [
        "10.part",
        "20.part",
        "meta.json",
        "payload.txt",
        TRAINED_METADATA_FILENAME,
    ]


def test_loading_a_run_with_no_archive_says_what_to_do_about_it(saved_run: str) -> None:
    """A run holding no model archive fails with an actionable message, not MLflow's raw one.

    MLflow's own error says only that the path was not found, which gives an operator
    re-materialising a fold nothing to act on. (``saved_run`` is depended on for the tracking URI
    it sets up, not for the model it holds.)
    """
    with mlflow.start_run(experiment_id=mlflow.create_experiment("empty_run")) as run:
        empty_run_id = run.info.run_id

    with pytest.raises(MlflowException, match="re-materialise `trained_cv_model`"):
        _FakeForecaster.load_from_mlflow(empty_run_id)


def test_the_archive_carries_the_trained_metadata_without_area_wkt(
    saved_run: str, tmp_path: Path
) -> None:
    """The archive must carry the rows a live asset locates its series by — and only those.

    ``area_wkt`` is 98.5% of the metadata table's bytes and nothing reads it, so it must not ride
    along into every fold's archive.
    """
    _FakeForecaster(
        BaseForecasterConfig(selected_features=set()), payload="located", series=[10, 20]
    ).save_to_mlflow(saved_run, time_series_metadata=_metadata_for([10, 20], area_wkt=True))

    dest = _download_and_unpack_model(run_id=saved_run, work_dir=tmp_path, remedy="")

    metadata = load_trained_metadata(dest)
    assert metadata["time_series_id"].to_list() == [10, 20]
    assert "h3_res_5" in metadata.columns
    assert "area_wkt" not in metadata.columns


def test_save_to_mlflow_hands_over_metadata_that_still_carries_its_patito_model(
    saved_run: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``write_trained_metadata`` is annotated to take a ``TimeSeriesMetadata`` frame.

    The rows reach it through an eager ``filter``, which returns a plain frame, so the model is
    only attached if ``save_to_mlflow`` re-attaches it.
    """
    received_models: list[object] = []
    original = ml_core.base_forecaster.write_trained_metadata

    def spy(*, model_dir: Path, time_series_metadata: pt.DataFrame[TimeSeriesMetadata]) -> None:
        received_models.append(getattr(time_series_metadata, "model", None))
        original(model_dir=model_dir, time_series_metadata=time_series_metadata)

    monkeypatch.setattr(ml_core.base_forecaster, "write_trained_metadata", spy)

    _save(saved_run, "re-saved", series=[10, 20])

    assert received_models == [TimeSeriesMetadata]


def test_load_trained_metadata_says_what_to_do_about_a_directory_without_it(
    tmp_path: Path,
) -> None:
    """A model directory predating the frozen copy names the remedy, not just the missing path."""
    tmp_path.joinpath("empty_model").mkdir()

    with pytest.raises(FileNotFoundError, match="deploy that run"):
        load_trained_metadata(tmp_path / "empty_model")


@pytest.mark.parametrize("model_release", [-1, 32768])
def test_model_release_outside_the_int16_range_is_rejected(model_release: int) -> None:
    with pytest.raises(ValueError, match="model_release"):
        BaseForecasterConfig(selected_features=set(), model_release=model_release)


@pytest.mark.parametrize("model_release", [0, 32767])
def test_model_release_at_the_int16_bounds_is_accepted(model_release: int) -> None:
    config = BaseForecasterConfig(selected_features=set(), model_release=model_release)

    assert config.model_release == model_release
