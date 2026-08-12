"""Tests for the MLflow model-artifact plumbing, against file-based MLflow.

Covers ``BaseForecaster.save_to_mlflow``/``load_from_mlflow`` and the third function that touches
the same archive layout, ``ml_core._production_helpers.fetch_model_artifacts`` — they are tested
together here because they share one on-the-wire format and one fake forecaster.

That fake (rather than a concrete model) keeps the tests focused on the shared behaviour —
archive upload/download — and free of any model-library dependency. It writes one file per
``time_series_id``, mirroring ``XGBoostForecaster``'s one-``.ubj``-per-series layout, which is
what makes a *shrinking* population observable as files that must vanish from the run.
"""

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Self

import mlflow
import patito as pt
import pytest
from contracts.config_schemas import class_target
from contracts.ml_schemas import AllFeatures
from contracts.power_schemas import PowerForecast
from ml_core._production_helpers import fetch_model_artifacts
from ml_core.base_forecaster import BaseForecaster, BaseForecasterConfig
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

pytestmark = pytest.mark.integration


class _FakeForecaster(BaseForecaster):
    """Minimal forecaster: a string payload plus one file per trained ``time_series_id``."""

    MODEL_NAME = "fake"
    MODEL_VERSION = 1
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


class _MetaWithoutModelParams(_FakeForecaster):
    """A forecaster whose saved record omits ``model_params`` altogether."""

    def save(self, path: Path) -> None:
        super().save(path)
        meta = json.loads((path / "meta.json").read_text())
        del meta["model_params"]
        (path / "meta.json").write_text(json.dumps(meta))


class _MetaWithUndeclaredHyperparameter(_FakeForecaster):
    """A forecaster saved while the code still declared a hyper-parameter that has since gone."""

    def save(self, path: Path) -> None:
        super().save(path)
        meta = json.loads((path / "meta.json").read_text())
        meta["model_params"]["dropout_rate"] = 0.1
        (path / "meta.json").write_text(json.dumps(meta))


class _MetaJsonMissing(_FakeForecaster):
    """A forecaster whose saved record omits ``meta.json`` altogether."""

    def save(self, path: Path) -> None:
        super().save(path)
        (path / "meta.json").unlink()


def test_trained_time_series_ids_is_abstract() -> None:
    """A subclass that omits ``trained_time_series_ids`` cannot be instantiated."""

    class _MissingPopulation(BaseForecaster):
        MODEL_NAME = "missing"
        MODEL_VERSION = 1

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
        _MissingPopulation(BaseForecasterConfig(selected_features=set()))


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
    forecaster.save_to_mlflow(run_id)


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
    different model after re-training. ``load_from_mlflow`` has no local cache (issue #469 removed
    it — see ``load_from_mlflow``'s docstring), so this is really a round-trip test, but it is
    worth keeping explicit: a future re-introduction of caching must not silently reintroduce the
    staleness bug this guards against.
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
    file would survive in the run forever — downloaded on every subsequent load and baked into
    the production container image. Uploading one same-name archive replaces it instead.
    """
    _save(saved_run, "smaller-model", series=[10, 20])

    # Nothing in the run's artifact store mentions the dropped series...
    assert _artifact_file_paths(saved_run) == ["model.tar.gz"]

    # ...and nothing the download path materialises does either, on either consumer.
    assert _FakeForecaster.load_from_mlflow(saved_run).trained_time_series_ids == [10, 20]

    dest = tmp_path / "production_model"
    fetch_model_artifacts(saved_run, dest)
    assert not (dest / "30.part").exists()
    assert sorted(p.name for p in dest.iterdir()) == [
        "10.part",
        "20.part",
        "meta.json",
        "payload.txt",
        "promotion.json",
    ]


def test_loading_a_run_with_no_archive_says_what_to_do_about_it(saved_run: str) -> None:
    """A run holding no model archive fails with an actionable message, not MLflow's raw one.

    MLflow's own error says only that the path was not found, which gives an operator
    re-materialising a fold nothing to act on. (``saved_run`` is depended on for the
    tracking URI it sets up, not for the model it holds.)
    """
    with mlflow.start_run(experiment_id=mlflow.create_experiment("empty_run")) as run:
        empty_run_id = run.info.run_id

    with pytest.raises(MlflowException, match="re-materialise `trained_cv_model`"):
        _FakeForecaster.load_from_mlflow(empty_run_id)


def test_fetch_model_artifacts_unpacks_the_archive_into_dest(
    saved_run: str, tmp_path: Path
) -> None:
    """The production download path lands a plain model directory, not an archive file."""
    dest = tmp_path / "production_model"
    fetch_model_artifacts(saved_run, dest)

    assert not (dest / "model.tar.gz").exists()
    assert _FakeForecaster.load(dest).payload == "hello-model"
    assert json.loads((dest / "promotion.json").read_text())["mlflow_run_id"] == saved_run


def _save_stale_vocabulary(run_id: str) -> None:
    """Save a model naming a feature a rename retired."""
    _save(run_id=run_id, payload="stale", series=[10], selected_features={"local_utc_offset"})


def _save_without_model_params(run_id: str) -> None:
    """Save a record with no ``model_params``, which no forecaster here can build a config from."""
    _MetaWithoutModelParams(
        model_params=BaseForecasterConfig(selected_features=set()),
        payload="featureless",
        series=[10],
    ).save_to_mlflow(run_id)


def _save_with_an_undeclared_hyperparameter(run_id: str) -> None:
    """Save a model whose stored config carries a key the config class no longer declares."""
    _MetaWithUndeclaredHyperparameter(
        model_params=BaseForecasterConfig(selected_features=set()),
        payload="undeclared",
        series=[10],
    ).save_to_mlflow(run_id)


def _save_without_meta_json(run_id: str) -> None:
    """Save a run whose archive holds the model files but no record describing them."""
    _MetaJsonMissing(
        model_params=BaseForecasterConfig(selected_features=set()),
        payload="recordless",
        series=[10],
    ).save_to_mlflow(run_id)


@pytest.mark.parametrize(
    ("save_unservable", "expected"),
    [
        (_save_stale_vocabulary, "local_utc_offset"),
        (_save_without_model_params, "model_params"),
        (_save_with_an_undeclared_hyperparameter, "model_params"),
        (_save_without_meta_json, "no meta.json"),
    ],
)
def test_fetch_model_artifacts_keeps_the_previous_model_when_the_new_one_is_unservable(
    saved_run: str,
    tmp_path: Path,
    save_unservable: Callable[[str], None],
    expected: str,
) -> None:
    """A promotion this code could not serve must not displace the champion already in place.

    The check runs before the atomic swap, so ``dest`` is untouched and the outgoing champion keeps
    serving instead of the service breaking at its next tick. The cases are the ways a saved record
    outlives the code that wrote it: a retired feature name, a hyper-parameter this code no longer
    declares, no config at all, and no record at all.
    """
    dest = tmp_path / "production_model"
    fetch_model_artifacts(run_id=saved_run, dest=dest)

    # Each parametrised case gets its own tmp_path, so its own tracking store: one name is enough.
    with mlflow.start_run(experiment_id=mlflow.create_experiment("unservable")) as run:
        unservable_run_id = run.info.run_id
    save_unservable(unservable_run_id)

    with pytest.raises(ValueError, match=expected) as exc_info:
        fetch_model_artifacts(run_id=unservable_run_id, dest=dest)

    # Which run was refused, so an operator knows what to re-train rather than what to re-download.
    assert unservable_run_id in str(exc_info.value)
    assert _FakeForecaster.load(dest).payload == "hello-model"
    assert json.loads((dest / "promotion.json").read_text())["mlflow_run_id"] == saved_run


def test_fetch_model_artifacts_keeps_the_previous_model_when_the_run_holds_no_model(
    saved_run: str, tmp_path: Path
) -> None:
    """A run id naming a run with no model saved to it must not displace the champion either.

    This is the likeliest operator slip of the three, because a mistyped or stale run id names a
    real run that simply never held a model. It is also the only refusal that happens in the
    download rather than in ``_check_meta_is_servable``, so it needs its own case to pin that
    ``dest`` survives it.

    The message must send the operator back to the run id they chose. Re-materialising
    ``trained_cv_model`` is the remedy for the *other* caller of the same download helper, and
    would be no help to someone who mistyped a run id here.
    """
    dest = tmp_path / "production_model"
    fetch_model_artifacts(run_id=saved_run, dest=dest)

    with mlflow.start_run(experiment_id=mlflow.create_experiment("modelless")) as run:
        modelless_run_id = run.info.run_id

    with pytest.raises(MlflowException, match="check the run id"):
        fetch_model_artifacts(run_id=modelless_run_id, dest=dest)

    assert _FakeForecaster.load(dest).payload == "hello-model"
