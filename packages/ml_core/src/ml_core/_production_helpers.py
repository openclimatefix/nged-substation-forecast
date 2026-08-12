"""Pure, IO-light helpers for production (live) inference.

Every function here is unit-testable in isolation: the two data-shaping helpers
(``select_nwp_init_time``, ``build_live_power_frame``) take ``power_fcst_init_time`` as an
explicit parameter rather than calling ``datetime.now()`` internally, so a test can pass any
fixed time and get a deterministic result; the two disk/MLflow helpers
(``load_forecaster_from_dir``, ``fetch_model_artifacts``) do the IO and check that the saved
model's feature names still parse. The ``live_forecasts`` and ``promoted_model`` Dagster assets
(``src/nged_substation_forecast/defs/production_assets.py``) stay thin shells over these.
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
from pydantic import ValidationError

from ml_core.base_forecaster import (
    BaseForecaster,
    BaseForecasterConfig,
    _download_and_unpack_model,
)
from ml_core.features._nwp import NWP_PUBLICATION_DELAY_HOURS
from ml_core.features._parsed_features import ParsedFeatures

AvailabilityModeType = Literal["live", "replay"]
"""Which NWP-availability rule ``select_nwp_init_time`` applies.

- ``"live"``: the scheduled path. No modelled publication delay — the Delta table only
  contains runs that have genuinely been published, so the cutoff is ``power_fcst_init_time``
  itself.
- ``"replay"``: re-running a past slot. The cutoff is
  ``power_fcst_init_time - nwp_publication_delay_hours``, reconstructing what was actually
  available at that historical ``power_fcst_init_time`` (without the delay we would leak runs
  that only landed afterwards).
"""


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


def build_live_power_frame(
    observed_power: pt.LazyFrame[PowerTimeSeries],
    time_series_ids: list[int],
    *,
    power_fcst_init_time: datetime,
    history: timedelta,
    horizon: timedelta,
) -> pt.LazyFrame[PowerTimeSeries]:
    """Build a dense half-hourly ``(time_series_id, time)`` spine for live inference.

    Needed because ``ml_core.features._nwp._join_nwp_single_run`` is power-centric — with no
    future power rows a live run would emit zero forecast rows. Left-joins observed power onto
    a spine covering ``(power_fcst_init_time - history, power_fcst_init_time + horizon]`` for
    every requested ``time_series_id``, so rows beyond the last observation are present with
    ``power = null``. Also harmless for replay (future observations already exist there;
    ``_nullify_leaky_lags`` prevents lag leakage regardless).

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


def _check_meta_features_are_parseable(meta: dict[str, Any], source: str) -> None:
    """Raise if this code cannot serve the model that ``meta.json`` describes.

    A saved model names its features as strings, so renaming or removing a feature in code leaves
    every model trained before the change unservable. See
    <https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules>
    for why this raises rather than degrading.

    Args:
        meta: The parsed contents of a model's ``meta.json``.
        source: What is being validated — a directory or a run id — quoted back in the message so
            an operator knows which model to re-train.

    Raises:
        ValueError: ``model_params`` carries no usable ``selected_features``, or one of those
            features is a name this code cannot parse.
    """
    # Both messages can reach the container log that scripts/build_and_verify_image.sh greps
    # case-insensitively for "mlflow" to prove the runtime is hermetic, so neither may contain that
    # word.
    remedy = (
        "Re-train against the current feature vocabulary and promote that run; never hand-edit "
        "meta.json, which leaves the trained model carrying the old names."
    )
    # Validate only the field this function reads. A saved config is always a *subclass* instance,
    # so handing the whole mapping to the base class would reject every real model on the subclass's
    # own hyper-parameters — ``BaseForecasterConfig`` forbids unknown keys. Whether the full config
    # still builds is the concrete subclass's ``load`` to decide, and it does.
    model_params = meta.get("model_params")
    try:
        config = BaseForecasterConfig.model_validate(
            {"selected_features": model_params.get("selected_features")}
            if isinstance(model_params, dict)
            else model_params
        )
    except (ValidationError, AttributeError) as error:
        raise ValueError(
            f"The model at {source} has model_params with no usable selected_features, so no "
            f"forecaster here can load it. {remedy}"
        ) from error

    # One at a time and sorted, so the feature named is the same in every process: they live in a
    # set, whose iteration order is not.
    for feature in sorted(config.selected_features):
        try:
            ParsedFeatures.from_strings({feature})
        except ValueError as error:
            raise ValueError(
                f"The model at {source} requests a feature this code cannot parse: {feature}. "
                f"{remedy}"
            ) from error


def load_forecaster_from_dir(path: Path) -> BaseForecaster:
    """Load the production model from a plain disk directory (no MLflow at inference time).

    Reads ``meta.json`` and resolves ``model_class`` via ``contracts.config_schemas.import_class``
    (the same mechanism ``ml_core._mlflow_runs.load_experiment_forecaster`` uses), then calls the
    concrete subclass's ``load(path)``.

    The forecaster returned is one this code can engineer features for, not merely one it could
    deserialise: an unparseable feature vocabulary is rejected here, so it surfaces at model load
    rather than partway through a live tick's feature engineering.

    Args:
        path: Directory previously populated by ``fetch_model_artifacts`` (the
            ``promoted_model`` asset's output).

    Returns:
        The reconstructed, trained forecaster.

    Raises:
        FileNotFoundError: ``path`` or its ``meta.json`` does not exist — materialise the
            ``promoted_model`` asset first.
        ValueError: ``meta.json`` has no ``model_class`` field — it was saved by a code version
            predating this contract; re-promote with a version that stamps ``model_class``. Or
            the model's ``selected_features`` name a feature this code cannot parse, in which case
            re-train and promote the new run.
    """
    meta_path = path / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No production model found at {path} (missing meta.json). Materialise the "
            "`promoted_model` asset first."
        )
    meta = json.loads(meta_path.read_text())
    model_class = meta.get("model_class")
    if model_class is None:
        raise ValueError(
            f"{meta_path} has no 'model_class' field, so the concrete forecaster class cannot "
            "be reconstructed. Re-promote the model with a code version that stamps "
            "model_class (see BaseForecaster.save)."
        )
    forecaster_cls = cast(type[BaseForecaster], import_class(model_class))
    # Before `load`, not after: `load` reads every serialised sub-model off disk, which is a lot of
    # IO to do on the way to rejecting the directory over one field already parsed above.
    _check_meta_features_are_parseable(meta=meta, source=str(path))
    return forecaster_cls.load(path)


def fetch_model_artifacts(run_id: str, dest: Path) -> None:
    """Download and unpack an MLflow run's saved model into ``dest``, replacing it atomically.

    Downloads and unpacks into a temporary directory first, so a failed or interrupted download
    never touches ``dest`` — only a fully-downloaded model is moved into place (via ``rmtree`` +
    ``move``). The run holds the model as a single archive artifact
    (``ml_core.base_forecaster._MLFLOW_MODEL_ARTIFACT``), so ``dest`` gets exactly the files the
    last ``save_to_mlflow`` wrote and can never inherit a stale file from an earlier, larger model.

    The downloaded model's ``selected_features`` are checked against the running code *before* the
    swap, reading the staged ``meta.json`` rather than loading the model, so a model this code
    cannot serve is refused while the previous champion stays in ``dest`` and keeps serving.
    Reading the one field is deliberate: loading would additionally pull every booster into memory
    to answer a question one JSON field already answers.

    Also writes a ``promotion.json`` (``{"mlflow_run_id", "promoted_at"}``) into ``dest`` for
    provenance; a ``BaseForecaster.load`` implementation reads its own population from its saved
    record (e.g. ``XGBoostForecaster`` from ``meta.json``'s ``trained_time_series_ids``), never
    from a directory listing, so this extra file is harmless.

    The caller is responsible for setting the tracking URI (``mlflow.set_tracking_uri``)
    beforehand.

    Args:
        run_id: The MLflow run the model was saved under (via ``BaseForecaster.save_to_mlflow``).
        dest: Directory to populate — typically ``Settings.production_model_path``.

    Raises:
        ValueError: The run holds no ``meta.json``, or the model's ``selected_features`` are
            absent, malformed, or name a feature this code cannot parse — re-train and promote a
            run this code version can read instead.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        downloaded_dir = _download_and_unpack_model(run_id, Path(tmp_dir))
        meta_path = downloaded_dir / "meta.json"
        if not meta_path.exists():
            raise ValueError(
                f"The model saved under run {run_id} has no meta.json, so no forecaster here can "
                "load it. Re-train and promote a run saved by a code version that writes "
                "meta.json (see BaseForecaster.save)."
            )
        meta = json.loads(meta_path.read_text())
        _check_meta_features_are_parseable(meta=meta, source=f"run {run_id}")

        promotion = {
            "mlflow_run_id": run_id,
            "promoted_at": datetime.now(UTC).isoformat(),
        }
        (downloaded_dir / "promotion.json").write_text(json.dumps(promotion))

        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(downloaded_dir), str(dest))
