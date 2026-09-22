"""IO-light helpers for production (live) inference.

The live service forecasts on a fixed schedule, and one scheduled forecast is a *slot*. The
weather input a slot forecasts from is a numerical weather prediction (NWP) run, initialised at
``nwp_init_time`` and carrying several ensemble members, of which member 0 is the unperturbed
control member. A *scan* below is a lazy Polars query over a Delta table, not a materialised
frame. Every function here is unit-testable in isolation. The two data-shaping helpers
(``select_nwp_init_time``, ``build_live_power_frame``) take ``power_fcst_init_time`` as an
explicit parameter rather than calling ``datetime.now()`` internally. A test can therefore pass
any fixed time and get a deterministic result. The two disk/MLflow helpers are
``load_forecaster_from_dir`` and ``fetch_model_artifacts``. Both do the IO. Both also check that
this code can still build a config for the saved model, and can still parse that model's
features. Of the five helpers this module exports, ``weather_lags_lack_their_control_member`` is
the only helper that executes a read against a data table rather than against a saved model. The
read is a bounded probe against the slot's NWP scan, and the scan is an argument, so a test can
pass an in-memory frame. The ``live_forecasts`` and ``promoted_model`` Dagster assets
(``src/nged_substation_forecast/defs/production_assets.py``) stay thin shells over these helpers.
"""

import json
import shutil
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

import patito as pt
import polars as pl
from contracts.common import UTC_DATETIME_DTYPE
from contracts.config_schemas import import_class
from contracts.power_schemas import PowerTimeSeries
from contracts.weather_schemas import Nwp
from pydantic import ValidationError
from weather_utils import NWP_ANALYSIS_MEMBER

from ml_core.base_forecaster import (
    TRAINED_METADATA_FILENAME,
    BaseForecaster,
    _download_and_unpack_model,
    load_trained_metadata,
)
from ml_core.features import NWP_PUBLICATION_DELAY_HOURS
from ml_core.features._parsed_features import ParsedFeatures

AvailabilityModeType = Literal["live", "replay"]
"""Which NWP-availability rule ``select_nwp_init_time`` applies.

- ``"live"``: the scheduled path. No modelled publication delay — the Delta table only
  contains runs that have genuinely been published, so the cutoff is ``power_fcst_init_time``
  itself.
- ``"replay"``: re-running a past slot over the data the table holds today. The cutoff is
``power_fcst_init_time - nwp_publication_delay_hours``, reconstructing what was actually available
at that historical ``power_fcst_init_time``. Subtracting the delay is what makes the cutoff earlier
than the live cutoff. Without the subtraction we would leak runs that only landed afterwards. """


def select_nwp_init_time(
    available_init_times: Sequence[datetime],
    *,
    power_fcst_init_time: datetime,
    availability_mode: AvailabilityModeType,
    nwp_publication_delay_hours: int = NWP_PUBLICATION_DELAY_HOURS,
) -> datetime:
    """Return the freshest NWP ``init_time`` available at ``power_fcst_init_time``.

    Which runs count as available depends on ``availability_mode``.

    Args:
        available_init_times: The ``init_time``s genuinely present in the NWP Delta table
            (e.g. from ``DeltaTable(...).partitions()``).
        power_fcst_init_time: The scheduled forecast time (the partition's window end).
        availability_mode: ``"live"`` uses cutoff ``power_fcst_init_time``; ``"replay"`` uses
            cutoff ``power_fcst_init_time - nwp_publication_delay_hours``.
        nwp_publication_delay_hours: Only used in ``"replay"`` mode.

    Returns:
        The freshest ``init_time`` that is ``<=`` the cutoff.

    Raises:
        ValueError: If no available ``init_time`` qualifies.
    """
    cutoff = (
        power_fcst_init_time
        if availability_mode == "live"
        else power_fcst_init_time - timedelta(hours=nwp_publication_delay_hours)
    )
    qualifying = [init_time for init_time in available_init_times if init_time <= cutoff]
    if not qualifying:
        raise ValueError(
            f"No NWP run available at or before cutoff {cutoff.isoformat()} "
            f"(power_fcst_init_time={power_fcst_init_time.isoformat()}, "
            f"availability_mode={availability_mode!r}). Available init times: "
            f"{sorted(available_init_times)}"
        )
    return max(qualifying)


def weather_lags_lack_their_control_member(
    nwp: pt.LazyFrame[Nwp], *, selected_features: set[str]
) -> bool:
    """Whether this slot asks for weather lags that its NWP run cannot fully supply.

    A weather model's *analysis* is its best estimate of the weather that actually happened. This
    pipeline has no analysis field, so it approximates an analysis with the control member of the
    freshest run — the analysis proxy. A weather lag reaching back before ``power_fcst_init_time``
    is answered by that analysis proxy, which reads the control member (``ensemble_member == 0``)
    alone. A run may carry no control-member rows at all, which happens when an ECMWF ENS download
    is partial or malformed. Such a run therefore nulls each weather lag over the first
    ``lag_hours`` of the horizon, where the lag still points into the past. The rest of the horizon
    is answered by the same-run join, which reads whichever ensemble members the run does carry.
    ``_engineer_features`` already degrades rather than failing there and logs a warning naming the
    run. This function exists so ``live_forecasts`` can *also* report the degradation on the Sentry
    channel. [Rule
    4](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules)
    requires that Sentry report alongside the log, not as a substitute for the log. Reading the logs
    is not a monitoring strategy: the operator reads the alert.

    Returns ``False`` when the model selects no weather lag. A model with no weather lags never
    reads the control member, because the same-run weather join reads whichever ensemble members the
    run does carry.

    H3 is a hexagonal grid over the globe, and NWP is stored one row per H3 cell, so each time
    series is matched to the cell it sits in. ``live_forecasts`` probes the NWP scan *before* that
    H3 spatial join, while ``_engineer_features`` probes the scan after. The earlier probe point
    makes the frame here a superset of the frame the pipeline sees. The alert can therefore in
    principle miss a degradation the pipeline hits, but can never fire on a degradation the pipeline
    does not hit. In practice ``load_engineering_inputs`` has already pruned the scan to the model's
    own frozen H3 cells — frozen meaning the copy the model saved at training time, in
    ``TRAINED_METADATA_FILENAME`` — so the two frames hold the same cells today.

    The probe reads at most one row, so a healthy NWP run answers the probe from a single row group.
    Proving an NWP run has *no* control member requires scanning the whole run, because absence
    cannot be shown early.

    Args:
        nwp: The slot's NWP rows, already narrowed to the selected run.
        selected_features: The promoted model's feature names.

    Returns:
        ``True`` when a weather lag is selected and the NWP run has no control-member rows.
    """
    weather_lags = [
        lag
        for lag in ParsedFeatures.from_strings(selected_features).lags
        if lag.base_col != "power"
    ]
    if not weather_lags:
        return False
    control_member = nwp.filter(pl.col("ensemble_member") == NWP_ANALYSIS_MEMBER)
    return control_member.limit(1).collect().is_empty()


def build_live_power_frame(
    observed_power: pt.LazyFrame[PowerTimeSeries],
    time_series_ids: list[int],
    *,
    power_fcst_init_time: datetime,
    history: timedelta,
    horizon: timedelta,
) -> pt.LazyFrame[PowerTimeSeries]:
    """Build a dense half-hourly ``(time_series_id, time)`` spine for live inference.

    Needed because ``ml_core.features._nwp._join_nwp_single_run`` is power-centric — with no future
    power rows a live run would emit zero forecast rows. Left-joins observed power onto a spine
    covering ``(power_fcst_init_time - history, power_fcst_init_time + horizon]`` for every
    requested ``time_series_id``. Rows beyond the last observation are therefore present with
    ``power = null``. The spine is also harmless for replay. Future observations already exist in
    replay, and ``_nullify_leaky_lags`` prevents lag leakage regardless.

    Args:
        observed_power: Lazy observed power, one row per ``(time_series_id, time)``.
        time_series_ids: The series to build a spine for (typically
            ``forecaster.trained_time_series_ids``).
        power_fcst_init_time: The forecast init time. The spine's window is anchored on this.
        history: How far before ``power_fcst_init_time`` the spine extends (exclusive) — must
            cover the longest power lag feature the model uses.
        horizon: How far after ``power_fcst_init_time`` the spine extends (inclusive) — the
            forecast horizon.

    Returns:
        A lazy ``PowerTimeSeries`` frame with one row per ``(time_series_id, time)`` on the
        half-hourly grid, observed values joined in, future/missing rows null.
    """
    grid_start = power_fcst_init_time - history + timedelta(minutes=30)
    grid_end = power_fcst_init_time + horizon
    grid_times = pl.datetime_range(
        grid_start, grid_end, interval="30m", time_zone="UTC", eager=True
    )

    ids_lf = pl.LazyFrame({"time_series_id": time_series_ids}, schema={"time_series_id": pl.Int32})
    times_lf = pl.LazyFrame({"time": grid_times}, schema={"time": UTC_DATETIME_DTYPE})
    spine = ids_lf.join(times_lf, how="cross")

    # Strip the Patito subclass before joining (see the `polars-patito-gotchas` skill).
    power_plain = pl.LazyFrame._from_pyldf(observed_power._ldf)
    dense = spine.join(power_plain, on=["time_series_id", "time"], how="left").sort(
        ["time_series_id", "time"]
    )
    return pt.LazyFrame.from_existing(dense).set_model(PowerTimeSeries)


def _check_meta_is_servable(meta: dict[str, Any], source: str) -> type[BaseForecaster]:
    """Raise if this code cannot serve the model that ``meta.json`` describes; return its class.

    A saved model names its class, its hyper-parameters, and its features as strings. Every model
    saved before the change therefore becomes unservable if any of those names is renamed or removed
    in code. The whole of ``model_params`` is validated against the concrete ``CONFIG_CLASS``
    reached from ``model_class``. That ``CONFIG_CLASS`` is the same object the subclass's ``load``
    builds its config from. A model that passes here is therefore a model ``load`` will accept. See
    <https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules>
    for why this raises rather than degrading.

    Args:
        meta: The parsed contents of a model's ``meta.json``.
        source: What is being validated — a directory or a run id — quoted back in the message so
            an operator knows which model to re-train.

    Returns:
        The concrete ``BaseForecaster`` subclass named by ``meta["model_class"]``.

    Raises:
        ValueError: ``meta`` names no importable ``model_class``. Or ``meta``'s ``model_params`` do
            not build that class's ``CONFIG_CLASS``, because a key the model declared has since been
            removed or renamed. Or one of ``meta``'s features is a name this code cannot parse.
    """
    # scripts/deploy/build_and_verify_image.sh proves the runtime never uses MLflow by grepping
    # the container log case-insensitively for "mlflow". Every message here can reach that log, so
    # a message carrying the word would fail that gate. No message here may contain it.
    remedy = (
        "Re-train against the current code and promote that run. Never hand-edit meta.json: that "
        "changes what the model claims, not what it was trained with."
    )
    model_class = meta.get("model_class")
    if model_class is None:
        raise ValueError(
            f"The model at {source} has no 'model_class' field, so the concrete forecaster class "
            f"cannot be reconstructed (see BaseForecaster.save). {remedy}"
        )
    forecaster_cls = cast(type[BaseForecaster], import_class(model_class))

    config_cls = forecaster_cls.CONFIG_CLASS
    try:
        config = config_cls.model_validate(meta.get("model_params"))
    except ValidationError as error:
        raise ValueError(
            f"The model at {source} has model_params that {config_cls.__name__} cannot build, so "
            f"{forecaster_cls.__name__} cannot load it. {remedy}"
        ) from error

    # Parse the features one at a time and in sorted order, so the feature named in the error is the
    # same in every process. The features live in a set, and a set's iteration order is not the same
    # in every process.
    for feature in sorted(config.selected_features):
        try:
            ParsedFeatures.from_strings({feature})
        except ValueError as error:
            raise ValueError(
                f"The model at {source} requests a feature this code cannot parse: {feature}. "
                f"{remedy}"
            ) from error
    return forecaster_cls


def _check_trained_metadata_is_readable(model_dir: Path, run_id: str) -> None:
    """Raise if a staged model carries no readable frozen metadata to locate its series by.

    Training writes a frozen snapshot of the metadata rows for a model's own series into
    ``TRAINED_METADATA_FILENAME`` inside the saved model directory, so that serving uses the rows
    training used. Production inference reads each series' H3 cell from that file, so a model
    without a usable copy of the file would forecast nothing at its next 6-hourly slot. Checking
    here refuses the promotion instead, before the swap, leaving the outgoing champion serving.

    Whether the file *covers* the trained population is not checked. ``write_trained_metadata`` is
    the one function in this repo that writes the file, and ``save_to_mlflow`` is that function's
    only production caller. ``save_to_mlflow``'s caller has already passed
    ``_require_metadata_coverage`` over a wider population. The trained population is a subset of
    that wider population.

    Args:
        model_dir: The staged, unpacked model directory (not yet moved into place).
        run_id: The run being promoted, quoted back in the message.

    Raises:
        ValueError: The file is absent, or cannot be read.
    """
    try:
        load_trained_metadata(model_dir)
    except Exception as error:
        raise ValueError(
            f"The model saved under run {run_id} has no readable {TRAINED_METADATA_FILENAME}, so "
            "live inference could not locate its time series. Re-train against the current code "
            "and promote that run."
        ) from error


def load_forecaster_from_dir(path: Path) -> BaseForecaster:
    """Load the production model from a plain disk directory (no MLflow at inference time).

    Reads ``meta.json`` and resolves ``model_class`` via ``contracts.config_schemas.import_class``
    (the same mechanism ``ml_core.mlflow_runs.load_experiment_forecaster`` uses), then calls the
    concrete subclass's ``load(path)``.

    The forecaster returned is a model this code can actually serve, not merely a model this code
    could deserialise. A config this code cannot rebuild is rejected here, as is a feature
    vocabulary this code cannot parse. Rejecting here beats failing partway through a live tick's
    feature engineering.

    Args:
        path: Directory previously populated by ``fetch_model_artifacts`` (the
            ``promoted_model`` asset's output).

    Returns:
        The reconstructed, trained forecaster.

    Raises:
        FileNotFoundError: ``path`` or its ``meta.json`` does not exist — materialise the
            ``promoted_model`` asset first.
        ValueError: This code cannot serve the saved model — see ``_check_meta_is_servable``.
            Promotion applies the same check, so this fires only when the code changed after the
            champion was promoted.
    """
    meta_path = path / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No production model found at {path} (missing meta.json). Materialise the "
            "`promoted_model` asset first."
        )
    meta = json.loads(meta_path.read_text())
    # Check before `load`, not after. `load` reads every serialised sub-model off disk. That is a
    # lot of IO to do on the way to rejecting the directory over fields already parsed here.
    forecaster_cls = _check_meta_is_servable(meta=meta, source=str(path))
    return forecaster_cls.load(path)


def fetch_model_artifacts(run_id: str, dest: Path) -> None:
    """Download and unpack an MLflow run's saved model into ``dest``, replacing it atomically.

    Downloads and unpacks into a temporary directory first, so a failed or interrupted download
    never touches ``dest``. Only a fully-downloaded model is moved into place, via ``rmtree`` +
    ``move``. ``dest`` is local disk by convention — ``Settings.production_model_path`` derives from
    ``local_artifacts_path``, though nothing enforces that. Unlike a Delta table, ``dest`` is a
    directory of many files with no commit protocol over it, and a part-written directory would be
    served. The run holds the model as a single archive artifact
    (``ml_core.base_forecaster._MLFLOW_MODEL_ARTIFACT``). ``dest`` therefore gets exactly the files
    the last ``save_to_mlflow`` wrote, and can never inherit a stale file from an earlier, larger
    model.

    The downloaded model's saved config is checked against the running code *before* the swap,
    reading the staged ``meta.json`` rather than loading the model. A model this code cannot serve
    is therefore refused while the previous champion stays in ``dest`` and keeps serving. Reading
    the JSON is deliberate: it applies the same validation the subclass's ``load`` would apply,
    without pulling every booster into memory first.

    Also writes a ``promotion.json`` (``{"mlflow_run_id", "promoted_at"}``) into ``dest`` for
    provenance. That extra file is harmless, because a ``BaseForecaster.load`` implementation reads
    its own population from its saved record, never from a directory listing. ``XGBoostForecaster``,
    for example, reads the population from ``meta.json``'s ``trained_time_series_ids``.

    The caller is responsible for setting the tracking URI (``mlflow.set_tracking_uri``)
    beforehand.

    Args:
        run_id: The MLflow run the model was saved under (via ``BaseForecaster.save_to_mlflow``).
        dest: Directory to populate — typically ``Settings.production_model_path``.

    Raises:
        MlflowException: ``run_id`` names a run holding no model archive — most often a mistyped
            or stale run id, since a run that trained a model has one. Raised by
            ``ml_core.base_forecaster._download_and_unpack_model``, before ``dest`` is touched.
        ValueError: The run holds no ``meta.json``. Or this code cannot serve the model that
            ``meta.json`` describes; see ``_check_meta_is_servable``. Or the run carries no readable
            frozen metadata; see ``_check_trained_metadata_is_readable``. Re-train against the
            current code and promote that run instead.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        downloaded_dir = _download_and_unpack_model(
            run_id=run_id,
            work_dir=Path(tmp_dir),
            remedy="check the run id, and pick one whose training completed.",
        )
        meta_path = downloaded_dir / "meta.json"
        if not meta_path.exists():
            raise ValueError(
                f"The model saved under run {run_id} has no meta.json, so no forecaster here can "
                "load it. Re-train against the current code and promote that run (see "
                "BaseForecaster.save)."
            )
        meta = json.loads(meta_path.read_text())
        _check_meta_is_servable(meta=meta, source=f"run {run_id}")
        _check_trained_metadata_is_readable(model_dir=downloaded_dir, run_id=run_id)

        promotion = {
            "mlflow_run_id": run_id,
            "promoted_at": datetime.now(UTC).isoformat(),
        }
        (downloaded_dir / "promotion.json").write_text(json.dumps(promotion))

        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(downloaded_dir), str(dest))
