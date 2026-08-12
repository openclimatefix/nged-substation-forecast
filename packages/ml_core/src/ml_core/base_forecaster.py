"""`BaseForecaster`: the interface every forecasting model implements.

Also holds `BaseForecasterConfig`, which carries a trained model's experiment identity.
"""

import tarfile
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Final, Self

import mlflow
import patito as pt
from contracts.ml_schemas import AllFeatures
from contracts.power_schemas import PowerForecast
from mlflow.exceptions import MlflowException
from pydantic import BaseModel, ConfigDict, field_serializer

from ml_core.features import FeatureEngineer, TabularFeatureEngineer

_MLFLOW_MODEL_ARTIFACT: Final[str] = "model.tar.gz"
"""Name of the single artifact, at an MLflow run's artifact root, holding the saved model.

The whole model directory is uploaded as **one archive file** rather than as a directory of
files, because MLflow's directory upload (``log_artifacts``) *merges* into a run's artifact
store instead of replacing it, while re-logging a single artifact of the same name overwrites
it (both verified empirically against MLflow 3.15.1). CV fold runs are reused across
re-materialisations, so a directory upload lets a re-training on a **smaller** population leave
the dropped series' files behind in the run forever — MLflow exposes no public artifact-delete
API to clean them up. One replaceable archive makes that accumulation impossible. See
<https://openclimatefix.github.io/nged-substation-forecast/architecture/ml-orchestration/#one-archive-file-not-a-directory-of-files>.

A run holding exactly *one* model file is what the fix rests on: logging a second per-model
artifact alongside this archive would reopen the merge problem for that artifact.
"""

_ARCHIVE_COMPRESSLEVEL: Final[int] = 1
"""gzip level used for the model archive — the fastest, not ``tarfile``'s default of 9.

Measured on 40 real XGBoost boosters (500 estimators, depth 6, 24 features; 77 MB of ``.ubj``):
level 1 takes 0.7 s for a 2.7x reduction, level 9 takes 14.9 s for 3.5x. Level 9 would put ~15
minutes of pure CPU on every ``trained_cv_model`` materialisation at V2 scale (~2,500 series)
to save a fifth of the bytes. Boosters are already dense, and the archive is transient — it
exists to be one replaceable object, not to be small — so the fastest level is the right trade.
"""


def _archive_model_dir(model_dir: Path, archive_path: Path) -> None:
    """Write everything in ``model_dir`` into a gzipped tar at ``archive_path``.

    Members are stored by bare filename (no leading directory component), so unpacking the
    archive reproduces ``model_dir``'s contents directly in the destination directory.

    Args:
        model_dir: The directory a subclass's ``save`` just wrote.
        archive_path: Where to write the ``.tar.gz`` (must not be inside ``model_dir``).
    """
    with tarfile.open(archive_path, "w:gz", compresslevel=_ARCHIVE_COMPRESSLEVEL) as tar:
        for item in sorted(model_dir.iterdir()):
            tar.add(item, arcname=item.name)


def _download_and_unpack_model(run_id: str, work_dir: Path, remedy: str) -> Path:
    """Download an MLflow run's model archive into ``work_dir`` and unpack it.

    Shared by ``BaseForecaster.load_from_mlflow`` and
    ``ml_core._production_helpers.fetch_model_artifacts``, the two readers of a run's saved
    model, so the archive layout is defined in exactly one place.

    The caller is responsible for setting the tracking URI (``mlflow.set_tracking_uri``)
    beforehand.

    Args:
        run_id: The MLflow run the model was saved under.
        work_dir: A scratch directory (typically a ``TemporaryDirectory``) to download and
            extract into. Needs room for the archive *and* its unpacked contents.
        remedy: What the reader should do when the run holds no archive, appended to the
            message. The two callers reach this state by different routes — a CV fold run that
            no training has written yet, versus a run id an operator chose — so neither wording
            is right for both, and the caller that knows which one it is supplies it.

    Returns:
        The directory holding the unpacked model, ready to hand to a subclass's ``load``. It
        contains only what the archive held, so a stale file from an earlier, larger model
        cannot appear in it.

    Raises:
        MlflowException: The run holds no model archive, followed by ``remedy``.
    """
    try:
        archive_path = Path(
            mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=_MLFLOW_MODEL_ARTIFACT, dst_path=str(work_dir)
            )
        )
    except MlflowException as error:
        raise MlflowException(
            f"MLflow run {run_id} has no {_MLFLOW_MODEL_ARTIFACT} artifact — {remedy}"
        ) from error
    # Unpack beside the archive rather than over it: download_artifacts has already claimed the
    # archive's own name inside work_dir.
    model_dir = work_dir / "unpacked_model"
    model_dir.mkdir()
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(model_dir, filter="data")
    return model_dir


class BaseForecasterConfig(BaseModel):
    """Universal configuration for all forecasting models.

    Subclasses add model-specific hyperparameters. Having a shared base ensures that every
    forecaster carries its own feature list and optional MLflow experiment id in one
    serialisable object, simplifying save/load and the ``conf/model/*.yaml`` config wiring.

    The tag fields (weather_source, training_strategy) are stamped onto MLflow runs for
    leaderboard grouping. They live here so that constructing the config from a model YAML's
    ``model_params`` validates them at load time — they are present in every
    ``conf/model/*.yaml`` file under ``model_params``.

    ``experiment_name`` is the per-experiment key (set to the MLflow experiment name at
    registration and stored in the saved config); it is stamped onto every ``PowerForecast``
    row, distinct from the model-family ``MODEL_NAME``. ``random_seed`` is threaded into each
    model's training so that re-training a fold reproduces the same model — keeping retries
    and the leaderboard stable.

    Model identity (name and version) lives on the ``BaseForecaster`` class itself as
    ``MODEL_NAME`` and ``MODEL_VERSION`` — those are properties of the implementation, not
    the experiment config.

    **Serialisation must be canonical.** A config is compared and stored as its serialised form:
    ``register_experiment`` stamps ``model_dump_json()`` onto the MLflow experiment as the
    ``config`` tag and compares a re-registration against it, and logs ``flatten_config(...)`` as
    write-once MLflow params. Python's per-process string hash randomisation means a ``set``
    iterates in a different order in every process, so a set dumped straight to a list would make
    two dumps of the *same* config differ — a re-registration would then look like a config change
    and its param write would be rejected. ``selected_features`` is therefore serialised sorted. A
    subclass that adds a set-valued (or otherwise unordered) field must do the same.

    **Unknown keys are rejected, not ignored** (``extra="forbid"``). A key no field declares raises
    ``ValidationError``, so a misspelled hyperparameter in a run's ``config_overrides`` fails at
    registration, and a *stored* config carrying a key the current code no longer declares is
    refused rather than silently losing it — the recovery is to re-train, never to hand-edit.
    Why this is worth failing over:
    <https://openclimatefix.github.io/nged-substation-forecast/ml_experimentation/model-configuration/#tweaking-a-config-for-an-experiment>.
    """

    model_config = ConfigDict(extra="forbid")

    selected_features: set[str]
    ml_flow_experiment_id: int | None = None
    experiment_name: str = ""
    weather_source: str = ""
    training_strategy: str = ""
    random_seed: int = 0

    @field_serializer("selected_features")
    def _serialise_selected_features(self, selected_features: set[str]) -> list[str]:
        """Serialise the feature set sorted, so two dumps of the same config always match."""
        return sorted(selected_features)


class BaseForecaster(ABC):
    """Defines the universal interface for all energy forecasting ML models.

    Every forecasting model subclasses this abstract base to allow shared Dagster assets and
    evaluation code to remain completely agnostic to the underlying model implementation.

    Subclasses must define ``MODEL_NAME``, ``MODEL_VERSION`` and ``CONFIG_CLASS`` as class-level
    constants. ``MODEL_NAME`` and ``MODEL_VERSION`` are stamped onto every ``PowerForecast`` row at
    predict time and used as the MLflow experiment name. Bumping ``MODEL_VERSION`` requires a code
    change (intentional), not a config edit.

    Lazy evaluation contract: `train` and `predict` both accept a `pt.LazyFrame[AllFeatures]`.
    Callers must not collect before passing data in; doing so wastes memory and prevents Polars
    from optimising the full query plan. A subclass materialises the data at the model boundary
    (typically a single `.collect()`, streamed). Keeping that bounded is the *caller's*
    responsibility: it prunes the inputs (NWP control member, the relevant H3 cells, the window's
    `init_time` partitions) and, where the full ensemble is needed, processes one `init_time` chunk
    at a time — filtering the engineered output cannot prune the upstream join/upsample. See the NWP
    scan-pruning notes in
    <https://openclimatefix.github.io/nged-substation-forecast/architecture/overview/>.

    Persistence has two layers. Subclasses implement ``save``/``load`` for their own on-disk
    format and need know nothing about MLflow. The concrete ``save_to_mlflow``/``load_from_mlflow``
    methods, shared by all subclasses, wrap that disk format with MLflow's artifact store, so the
    same trained model can be shared across machines by round-tripping through a run.
    """

    MODEL_NAME: ClassVar[str]
    MODEL_VERSION: ClassVar[int]

    CONFIG_CLASS: ClassVar[type[BaseForecasterConfig]]
    """The config class ``load`` rebuilds from a saved ``model_params`` mapping.

    A subclass's ``load`` must build its config through this attribute rather than naming its
    config class again, so the class ``ml_core._production_helpers._check_meta_is_servable``
    validates a saved config against is always the class ``load`` uses.
    """

    feature_engineer: ClassVar[FeatureEngineer] = TabularFeatureEngineer()
    """The feature pipeline this forecaster's data is engineered through.

    Associated by composition (the forecaster *references* a feature engineer rather than
    *implementing* feature engineering), so a forecaster can swap the whole pipeline by overriding
    this with a different ``FeatureEngineer``. The default produces the tabular ``AllFeatures``
    frame that ``train``/``predict`` consume.
    """

    def __init__(self, model_params: BaseForecasterConfig) -> None:
        """Store the config that carries this model's hyper-parameters and experiment identity."""
        self.model_params = model_params

    @property
    @abstractmethod
    def trained_time_series_ids(self) -> list[int]:
        """The sorted ``time_series_id`` population this model will serve a ``predict`` for.

        This is the model's own frozen record of who it trained on — *not* a statement about how
        the model is internally structured. A model may hold one sub-model per series, a single
        model spanning many series, or anything in between; the contract is only about which
        ``time_series_id``s it will score.

        **Why this is load-bearing.** The **train==predict population invariant**: the
        population a model is scored on must equal the population it was trained on, *even if* the
        live eligibility set has drifted since training (power coverage changes, so a series may
        newly qualify or drop out). Consumers (``cv_power_forecasts``, ``live_forecasts``) filter
        their inputs to this set, so a model scores exactly the population it learned — never
        whatever eligibility says *today*. That is what keeps the leaderboard apples-to-apples and
        stops production forecasting a series the model never saw.

        Subclasses persist and reconstruct this set through their own ``save``/``load`` (e.g. in
        ``meta.json``), so it survives a round-trip through MLflow.
        """
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save the trained model state to a directory, **replacing** anything already there.

        Two requirements on every implementation:

        - The saved directory must contain a ``meta.json`` with a ``model_class`` field — the
          fully-qualified ``{module}.{qualname}`` of the concrete subclass (e.g.
          ``"xgboost_forecaster.forecaster.XGBoostForecaster"``) — so that production inference
          (``ml_core._production_helpers.load_forecaster_from_dir``) can reconstruct the correct
          class from a plain model directory with no other context (issue #221).
        - ``path`` must be cleared first, so that saving over a directory holding a *larger*
          model's files leaves none of them behind. Merging instead of replacing is how a dropped
          time series' weights survive a re-train (issue #197); ``XGBoostForecaster.save`` clears
          with ``shutil.rmtree(path, ignore_errors=True)``.

        The clearing requirement makes ``path`` the model's to own while it saves, so anything a
        caller left there is gone afterwards. (Depositing a file *after* a save is fine, and is how
        ``_production_helpers.fetch_model_artifacts`` puts ``promotion.json`` beside the model.)
        """

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> Self:
        """Reconstruct a trained instance from a previously saved directory."""

    def save_to_mlflow(self, run_id: str) -> None:
        """Upload this trained model to the given MLflow run, as one replaceable archive.

        Writes the model to a temporary directory via ``save`` (the subclass's own format),
        packs that directory into a single ``model.tar.gz`` and logs *that one file* to the
        run's artifact root. Logging one archive rather than a directory of files is what makes
        a re-upload **replace** the previous model instead of merging with it — see
        ``_MLFLOW_MODEL_ARTIFACT``. The caller is responsible for setting the tracking URI
        (``mlflow.set_tracking_uri``) beforehand.

        Args:
            run_id: The MLflow run to attach the artifact to.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "model"
            model_dir.mkdir()
            self.save(model_dir)
            archive_path = Path(tmp_dir) / _MLFLOW_MODEL_ARTIFACT
            _archive_model_dir(model_dir, archive_path)
            with mlflow.start_run(run_id=run_id):
                mlflow.log_artifact(str(archive_path))

    @classmethod
    def load_from_mlflow(cls, run_id: str) -> Self:
        """Download a trained model's archive from an MLflow run, unpack it and load it.

        Downloads into a temporary directory and loads from there. There is deliberately no
        local-disk cache: a CV fold run is **reused** across re-materialisations
        (``get_or_create_fold_run`` resolves the same run for every re-run of a fold's partition),
        so a cache keyed by ``run_id`` would not be unique for its contents — and production
        inference makes no MLflow call at all. Full rationale:
        <https://openclimatefix.github.io/nged-substation-forecast/architecture/ml-orchestration/#why-there-is-no-local-cache>.

        The caller is responsible for setting the tracking URI (``mlflow.set_tracking_uri``)
        beforehand.

        Args:
            run_id: The MLflow run the model was saved under.

        Returns:
            The reconstructed, trained forecaster.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            return cls.load(
                _download_and_unpack_model(
                    run_id=run_id,
                    work_dir=Path(tmp_dir),
                    remedy="re-materialise `trained_cv_model` for this fold.",
                )
            )

    @abstractmethod
    def train(self, data: pt.LazyFrame[AllFeatures], time_series_ids: list[int]) -> None:
        """Fit the model on ``data`` for the given ``time_series_id`` population.

        Args:
            data: The engineered features (lazy; the caller must not pre-collect).
            time_series_ids: The population the model must train on — the caller's eligible set.
                The model decides how to map it onto boosters: one booster per id
                (``XGBoostForecaster``), or one booster per group of ids (e.g. all solar sites) for
                a future model. It is the model's frozen record of who it trained on (the
                train==predict invariant), and bounds ``predict`` to that same population.
        """

    @abstractmethod
    def predict(
        self, data: pt.LazyFrame[AllFeatures], *, fold_id: str = "live"
    ) -> pt.DataFrame[PowerForecast]:
        """Return power forecasts for all rows in the given AllFeatures data.

        Args:
            data: The engineered features to forecast from.
            fold_id: The value stamped onto every row's ``fold_id`` column. The model has no
                inherent notion of which CV fold it is serving (``fold_id`` is orchestration
                context), so the caller supplies it: ``cv_power_forecasts`` passes the fold's
                label, while production inference keeps the ``"live"`` default.
        """
