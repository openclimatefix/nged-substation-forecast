"""Pure, IO-light helpers for production (live) inference.

Every function here is unit-testable in isolation: the two data-shaping helpers
(``select_nwp_init_time``, ``build_live_power_frame``) take ``power_fcst_init_time`` as an
explicit parameter rather than calling ``datetime.now()`` internally, so a test can pass any
fixed time and get a deterministic result; the two disk/MLflow helpers
(``load_forecaster_from_dir``, ``fetch_model_artifacts``) are thin, single-purpose IO wrappers.
The ``live_forecasts`` and ``promoted_model`` Dagster assets
(``src/nged_substation_forecast/defs/production_assets.py``) stay thin shells over these.
"""

import json
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Sequence, cast

import hydra
import mlflow
import patito as pt
import polars as pl
from contracts.common import UTC_DATETIME_DTYPE
from contracts.power_schemas import PowerTimeSeries

from ml_core.base_forecaster import _MLFLOW_ARTIFACT_PATH, BaseForecaster
from ml_core.features._nwp import NWP_PUBLICATION_DELAY_HOURS

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
    """Return the freshest NWP ``init_time`` available at ``power_fcst_init_time`` under the given mode.

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

    # Strip the Patito subclass before joining (see CLAUDE.md's cross-model-join gotcha).
    power_plain = pl.LazyFrame._from_pyldf(observed_power._ldf)
    dense = spine.join(power_plain, on=["time_series_id", "time"], how="left").sort(
        ["time_series_id", "time"]
    )
    return pt.LazyFrame.from_existing(dense).set_model(PowerTimeSeries)


def load_forecaster_from_dir(path: Path) -> BaseForecaster:
    """Load the production model from a plain disk directory (no MLflow at inference time).

    Reads ``meta.json`` and resolves ``model_class`` via ``hydra.utils.get_class`` (the same
    mechanism ``ml_core._mlflow_runs.load_experiment_forecaster`` uses), then calls the
    concrete subclass's ``load(path)``.

    Args:
        path: Directory previously populated by ``fetch_model_artifacts`` (the
            ``promoted_model`` asset's output).

    Returns:
        The reconstructed, trained forecaster.

    Raises:
        FileNotFoundError: ``path`` or its ``meta.json`` does not exist — materialise the
            ``promoted_model`` asset first.
        ValueError: ``meta.json`` has no ``model_class`` field — it was saved by a code version
            predating this contract; re-promote with a version that stamps ``model_class``.
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
    forecaster_cls = cast(type[BaseForecaster], hydra.utils.get_class(model_class))
    return forecaster_cls.load(path)


def fetch_model_artifacts(run_id: str, dest: Path) -> None:
    """Download an MLflow run's saved model artifacts to ``dest``, replacing it atomically.

    Downloads into a temporary directory first, so a failed or interrupted download never
    touches ``dest`` — only a fully-downloaded model is moved into place (via ``rmtree`` +
    ``move``). Also writes a ``promotion.json`` (``{"mlflow_run_id", "promoted_at"}``) into
    ``dest`` for provenance; ``BaseForecaster.load`` implementations glob for their own model
    files (e.g. ``*.ubj``), so this extra file is harmless.

    The caller is responsible for setting the tracking URI (``mlflow.set_tracking_uri``)
    beforehand.

    Args:
        run_id: The MLflow run the model was saved under (via ``BaseForecaster.save_to_mlflow``).
        dest: Directory to populate — typically ``Settings.production_model_path``.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=_MLFLOW_ARTIFACT_PATH, dst_path=tmp_dir
        )
        downloaded_dir = Path(tmp_dir) / _MLFLOW_ARTIFACT_PATH
        promotion = {
            "mlflow_run_id": run_id,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }
        (downloaded_dir / "promotion.json").write_text(json.dumps(promotion))

        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(downloaded_dir), str(dest))
