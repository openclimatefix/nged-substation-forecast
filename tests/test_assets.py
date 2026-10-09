"""Materialisation tests for the three ingest Dagster assets, plus a definitions-load smoke test.

Fires up Dagster for each ingest asset — ``power_time_series_and_metadata``, ``h3_grid_weights``,
``ecmwf_ens`` — against temp Delta/parquet tables, and asserts the whole asset graph (assets +
jobs + schedules) resolves. The three leaf data pipelines (NGED JSON parsing, H3 weighting, ECMWF
download/convert) are unit-tested in their own packages; here we exercise only the asset *bodies*
— the wiring, branching, and metadata each asset owns — stubbing the S3/network boundary and the
~30-second GB-boundary buffer so the tests stay fast and offline.
"""

import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import patito as pt
import polars as pl
import pytest
import shapely
import xarray as xr
from contracts.geo_schemas import H3GridWeights
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from contracts.settings import Settings
from contracts.weather_schemas import Nwp, NwpQualityReport, NwpVariableWhollyMissing
from dagster import (
    AssetCheckEvaluation,
    AssetCheckResult,
    AssetCheckSeverity,
    DagsterExecutionInterruptedError,
    DagsterInstance,
    ExecuteInProcessResult,
    RetryRequested,
    TableMetadataValue,
    build_asset_context,
    materialize,
)
from deltalake import DeltaTable
from dynamical_data.ecmwf_ens.download import _ECMWF_ENS_VARS_TO_DOWNLOAD, NwpRunNotYetAvailable
from dynamical_data.ecmwf_ens.upstream_nulls import UpstreamNullRate
from nged_data.storage import DownloadedFilesError, _ProcessedFileListing

from nged_substation_forecast.defs import assets
from nged_substation_forecast.defs.assets import (
    _ECMWF_ENS_MAX_RETRIES,
    _ECMWF_ENS_RETRY_DELAY_SECONDS,
    _BaseSummary,
    _FileListingSummary,
    _PowerTimeSeriesSummary,
    ecmwf_ens,
    h3_grid_weights,
    power_time_series_and_metadata,
)

pytestmark = pytest.mark.integration

_NGED_JSON_DIR = Path(__file__).resolve().parents[1] / "packages" / "nged_data" / "tests" / "data"
"""Reuse the real (tiny) NGED JSON fixtures rather than duplicating them into this directory."""

_NGED_FILES: dict[str, bytes] = {
    "timeseries/1774512000000_1774533600000/TimeSeries_10_20260326T080000Z_20260326T140000Z.json": (
        _NGED_JSON_DIR / "TimeSeries_10.json"
    ).read_bytes(),
    "timeseries/1774512000000_1774533600000/TimeSeries_11_20260326T080000Z_20260326T140000Z.json": (
        _NGED_JSON_DIR / "TimeSeries_11.json"
    ).read_bytes(),
}
"""Paths of the form NGED publishes (``…/<start_ms>_<end_ms>/TimeSeries_<id>_…json``) so the real
path-parsing regex in ``list_timeseries_json_files`` extracts a valid listing."""


# Aliases used in the fake-store annotations below: the ``.bytes()`` and ``.list()`` methods
# (named to match obstore's API) shadow the ``bytes``/``list`` builtins inside their own class
# scope, so the annotations reference these module-level names instead.
_JsonBytes = bytes
_StoreListing = list[list[dict[str, object]]]


class _FakeGetResult:
    def __init__(self, data: _JsonBytes) -> None:
        self._data = data

    async def bytes_async(self) -> _JsonBytes:
        return self._data


_FIRST_LAST_MODIFIED = datetime(2026, 3, 26, 15, 0, tzinfo=UTC)
"""The `LastModified` a file gets in the fake bucket unless a test gives it another."""


class _FakeS3Store:
    """Minimal ``obstore`` store stand-in serving a set of NGED JSON files that can grow.

    ``list_timeseries_json_files`` and ``download_and_parse_files`` only call ``.list()`` and
    ``.get_async()``, so duck-typing those two methods lets the real asset body run offline.
    ``put`` adds or rewrites a file between runs, and ``requested_paths`` records every download.
    """

    def __init__(self, files: dict[str, _JsonBytes]) -> None:
        self._files = {path: (data, _FIRST_LAST_MODIFIED) for path, data in files.items()}
        self.requested_paths: list[str] = []
        self.n_list_calls = 0

    def put(
        self, path: str, data: _JsonBytes, last_modified: datetime = _FIRST_LAST_MODIFIED
    ) -> None:
        self._files[path] = (data, last_modified)

    def list(self, prefix: str) -> _StoreListing:
        self.n_list_calls += 1
        return [
            [
                {"path": path, "size": len(data), "last_modified": last_modified}
                for path, (data, last_modified) in self._files.items()
            ]
        ]

    async def get_async(self, path: str) -> _FakeGetResult:
        self.requested_paths.append(path)
        return _FakeGetResult(self._files[path][0])


_CONTINUOUS_NWP_VALUES: dict[str, float] = {
    "temperature_2m": 15.7031,
    "dew_point_temperature_2m": 9.1234,
    "wind_speed_10m": 5.6789,
    "wind_direction_10m": 123.456,
    "wind_speed_100m": 8.9101,
    "wind_direction_100m": 234.567,
    "pressure_surface": 101_234.5,
    "pressure_reduced_to_mean_sea_level": 101_567.8,
    "geopotential_height_500hpa": 5_432.1,
    "downward_long_wave_radiation_flux_surface": 312.34,
    "downward_short_wave_radiation_flux_surface": 456.78,
    "precipitation_surface": 0.00123,
}


def _make_nwp(init_time: datetime, n: int = 4) -> pl.DataFrame:
    """A tiny valid ``Nwp`` frame for one run — stands in for ``convert_…``'s output."""
    rows = {
        "nwp_model_id": ["ECMWF_ENS_0_25_degree"] * n,
        "init_time": [init_time] * n,
        "valid_time": [init_time + timedelta(hours=i + 1) for i in range(n)],
        "ensemble_member": list(range(n)),
        "h3_index": [100 + i for i in range(n)],
        "categorical_precipitation_type_surface": [1] * n,
        **{var: [value] * n for var, value in _CONTINUOUS_NWP_VALUES.items()},
    }
    return Nwp.DataFrame(rows).cast().validate()


def _make_downloaded_ds(
    n_null_grid_points: int = 0, n_null_instantaneous_grid_points: int = 1
) -> xr.Dataset:
    """A tiny stand-in for ``download_ecmwf_ens_data``'s output.

    A ``(lead_time, ensemble_member, latitude, longitude)`` grid whose first step is lead-0, so
    nulls placed beyond it are counted rather than filtered out as the by-design lead-0 ones.

    Carries all thirteen downloaded variables, under their *download* names, because the two null
    populations are counted over sets drawn from that list rather than from whatever ``ds``
    happens to hold — a fixture holding only the variables a test cares about would let a counter
    reading the wrong namespace pass. ``temperature_2m`` is the corrupt instantaneous one and
    ``precipitation_surface`` the corrupt de-accumulated one, so a count that pooled the two, or
    attributed one to the other, disagrees with these tests.
    """
    shape = (2, 1, 1, 3)  # 2 steps (one is lead-0), 1 member, 3 grid points
    dims = ("lead_time", "ensemble_member", "latitude", "longitude")
    corrupt = np.full(shape, 0.001, dtype=np.float32)
    corrupt[1, 0, 0, :n_null_grid_points] = np.nan
    instantaneous = np.full(shape, 15.0, dtype=np.float32)
    instantaneous[1, 0, 0, :n_null_instantaneous_grid_points] = np.nan
    return xr.Dataset(
        data_vars={
            "temperature_2m": (dims, instantaneous),
            **{
                name: (
                    dims,
                    corrupt
                    if name == "precipitation_surface"
                    else np.full(shape, 0.001, dtype=np.float32),
                )
                for name in _ECMWF_ENS_VARS_TO_DOWNLOAD
                if name != "temperature_2m"
            },
        },
        coords={
            "lead_time": np.asarray([0, 6], dtype="timedelta64[h]").astype("timedelta64[ns]"),
            "ensemble_member": np.asarray([0]),
            "latitude": np.asarray([52.5], dtype=np.float32),
            "longitude": np.asarray([-1.0, -0.75, -0.5], dtype=np.float32),
        },
    )


def _write_h3_grid_weights(path: str) -> None:
    """A minimal valid ``H3GridWeights`` parquet — ``ecmwf_ens`` reads it before downloading."""
    H3GridWeights.DataFrame(
        {"h3_index": [100], "nwp_lat": [52.5], "nwp_lon": [-1.0], "proportion": [1.0]}
    ).cast().validate().write_parquet(path)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every managed data-path root at a temp dir, fully isolating the assets from the
    developer's real configuration."""
    monkeypatch.setenv("DATA_PATH_INTERNAL", str(tmp_path))
    monkeypatch.setenv("DATA_PATH_DELIVERY", str(tmp_path))
    monkeypatch.setenv("LOCAL_ARTIFACTS_PATH", str(tmp_path))
    return tmp_path


# --- power_time_series_and_metadata --------------------------------------------------------------


def test_power_time_series_and_metadata_ingests_and_writes(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """Happy path: a fake S3 store serving two real NGED JSON files → metadata parquet + power
    Delta table both written, and the asset materialises successfully."""
    monkeypatch.setattr(
        target=assets.Settings,
        name="get_nged_s3_store",
        value=lambda self: _FakeS3Store(_NGED_FILES),
    )

    result = materialize([power_time_series_and_metadata], instance=dagster_instance)
    assert result.success

    metadata = pl.read_parquet(env / "NGED" / "metadata.parquet")
    TimeSeriesMetadata.validate(metadata)
    assert set(metadata["time_series_id"].to_list()) == {10, 11}

    # Reading a time_series_id-partitioned Delta table doesn't guarantee global sort order, so
    # sort before validating against the (sortedness-checking) PowerTimeSeries contract.
    power = pl.read_delta(str(env / "NGED" / "power_time_series.delta")).sort(
        PowerTimeSeries.columns_to_sort_by
    )
    PowerTimeSeries.validate(power)
    assert set(power["time_series_id"].unique().to_list()) == {10, 11}

    # The asset wires both summary tables into its Dagster output metadata (the summary classes'
    # own logic is unit-tested below; this covers the asset → add_output_metadata glue).
    materialisations = result.asset_materializations_for_node("power_time_series_and_metadata")
    metadata_keys = set().union(*(mat.metadata.keys() for mat in materialisations))
    assert {"nged_s3_paths", "PowerTimeSeries"} <= metadata_keys


@pytest.mark.parametrize("raised", [RuntimeError, BaseException], ids=["exception", "rust_panic"])
def test_power_time_series_and_metadata_writes_power_when_the_metadata_upsert_fails(
    raised: type[BaseException],
    env: Path,
    monkeypatch: pytest.MonkeyPatch,
    dagster_instance: DagsterInstance,
) -> None:
    """The headline property of #508: the metadata table is derived data, so a fault in it must
    not stall the power stream until an operator intervenes. The downloaded-files list is still
    written, so the next run does not download the same files again.

    Also asserts the degradation is *reported*, since a step that no longer fails no longer fires
    ``sentry_capture_failure``. The ``rust_panic`` case is why the guard catches
    ``BaseException``: a pyo3 ``PanicException`` from Polars or obstore is not an ``Exception``,
    and the cancellation test below cannot catch a narrowed guard, because
    ``DagsterExecutionInterruptedError`` escapes one on its own.
    """
    monkeypatch.setattr(
        target=assets.Settings,
        name="get_nged_s3_store",
        value=lambda self: _FakeS3Store(_NGED_FILES),
    )

    def boom(*_: object, **__: object) -> None:
        raise raised("metadata table upsert exploded")

    monkeypatch.setattr(target=assets, name="upsert_metadata", value=boom)
    reported: list[tuple[str, BaseException]] = []
    monkeypatch.setattr(
        target=assets,
        name="report_asset_degradation",
        value=lambda asset_name, exc: reported.append((asset_name, exc)),
    )

    result = materialize([power_time_series_and_metadata], instance=dagster_instance)
    assert result.success

    power = pl.read_delta(str(env / "NGED" / "power_time_series.delta")).sort(
        PowerTimeSeries.columns_to_sort_by
    )
    PowerTimeSeries.validate(power)
    assert set(power["time_series_id"].unique().to_list()) == {10, 11}

    materialisations = result.asset_materializations_for_node("power_time_series_and_metadata")
    metadata_keys = set().union(*(mat.metadata.keys() for mat in materialisations))
    assert "metadata_upsert_failed" in metadata_keys

    assert [name for name, _ in reported] == ["power_time_series_and_metadata"]
    # `type(...) is`, not `isinstance`: under `isinstance` the `rust_panic` case would pass on a
    # `RuntimeError`, so a guard narrowed to `except Exception` would still look correct.
    assert type(reported[0][1]) is raised
    assert pl.read_parquet(env / "NGED" / "downloaded_files.parquet").height == len(_NGED_FILES)


def test_power_time_series_and_metadata_re_raises_a_cancelled_run(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """The one thing the guard must *not* swallow. Cancellation lands in the same
    ``BaseException`` net as a panic, so the handler re-raises it explicitly: a run the
    operator cancelled has to stop, not finish green having quietly skipped the metadata table."""
    monkeypatch.setattr(
        target=assets.Settings,
        name="get_nged_s3_store",
        value=lambda self: _FakeS3Store(_NGED_FILES),
    )

    def _cancel(*_: object, **__: object) -> None:
        raise DagsterExecutionInterruptedError

    monkeypatch.setattr(target=assets, name="upsert_metadata", value=_cancel)
    reported: list[str] = []
    monkeypatch.setattr(
        target=assets,
        name="report_asset_degradation",
        value=lambda asset_name, exc: reported.append(asset_name),
    )

    result = materialize(
        [power_time_series_and_metadata], instance=dagster_instance, raise_on_error=False
    )
    assert not result.success
    # Not merely "it failed": dropping the re-raise also fails the run, via the degradation path.
    assert reported == []


def test_power_time_series_and_metadata_drops_and_reports_malformed_rows(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """A row with a malformed `time` is dropped and counted, not allowed to abort ingestion of
    every other well-formed row in the batch."""
    fixture = json.loads((_NGED_JSON_DIR / "TimeSeries_10.json").read_text())
    fixture["data"] = [
        {
            "value": 1.0,
            "startTime": "2026-03-05 12:00:00+0000",
            "endTime": "2026-03-05 12:30:00+0000",
        },
        # Malformed: outside the plausible datetime range.
        {
            "value": 2.0,
            "startTime": "1840-06-01 00:00:00+0000",
            "endTime": "1840-06-01 00:30:00+0000",
        },
    ]
    object_key = (
        "timeseries/1774512000000_1774533600000"
        "/TimeSeries_10_20260326T080000Z_20260326T140000Z.json"
    )
    files = {object_key: json.dumps(fixture).encode()}
    monkeypatch.setattr(
        target=assets.Settings, name="get_nged_s3_store", value=lambda self: _FakeS3Store(files)
    )

    result = materialize([power_time_series_and_metadata], instance=dagster_instance)
    assert result.success

    power = pl.read_delta(str(env / "NGED" / "power_time_series.delta"))
    assert power.height == 1
    # Expect 12:00, not the 12:30 NGED stamped: this reading predates
    # `POWER_TIMESTAMPS_CORRECTED_BEFORE`, so the ingest moves the reading back 30 minutes
    # (`PowerTimeSeries.correct_late_timestamps`).
    assert power["time"][0] == datetime(year=2026, month=3, day=5, hour=12, minute=0, tzinfo=UTC)

    materialisations = result.asset_materializations_for_node("power_time_series_and_metadata")
    metadata = {k: v for mat in materialisations for k, v in mat.metadata.items()}
    assert metadata["n_implausible_power_rows_dropped"].value == 1


def test_the_newest_data_less_file_of_a_series_sets_its_metadata(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """Series 10 has an older file with readings and a newer data-less file that carries NGED's
    note. The data-less file adds no rows, but its metadata must win."""
    small_file = json.loads((_NGED_JSON_DIR / "TimeSeries_33_no_data.json").read_text())
    small_file.update(TimeSeriesID=10, Information="Invented fault note.")
    small_file_key = (
        "timeseries/1774555200000_1774576800000"
        "/TimeSeries_10_20260326T140000Z_20260326T200000Z.json"
    )
    files = {**_NGED_FILES, small_file_key: json.dumps(small_file).encode()}
    monkeypatch.setattr(
        target=assets.Settings, name="get_nged_s3_store", value=lambda self: _FakeS3Store(files)
    )

    result = materialize([power_time_series_and_metadata], instance=dagster_instance)

    assert result.success
    metadata = pl.read_parquet(env / "NGED" / "metadata.parquet")
    notes = dict(metadata.select("time_series_id", "information").rows())
    assert notes[10] == "Invented fault note."
    assert notes[11] is None
    assert pl.read_delta(str(env / "NGED" / "power_time_series.delta")).height > 0


def test_power_time_series_and_metadata_retries_a_transient_upstream_failure(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """A blip reading NGED's bucket costs a short wait, not a failed run: no data is lost either
    way, so failing would report a fault that has already fixed itself.

    The assertions on the written data are the point of doing this through ``materialize`` rather
    than counting calls alone — a second attempt re-lists and re-downloads everything, so this is
    also what pins down that re-running the ingest cannot duplicate rows.
    """
    monkeypatch.setattr(
        target=assets.Settings,
        name="get_nged_s3_store",
        value=lambda self: _FakeS3Store(_NGED_FILES),
    )
    # Nothing here asserts on timing, so skip the wait rather than sleeping through it.
    monkeypatch.setattr(target=assets, name="_POWER_INGEST_RETRY_DELAY_SECONDS", value=0)
    real_download = assets.download_and_parse_files
    calls = 0

    def _fail_once(store: Any, paths_df: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("transient object-store error")
        return real_download(store, paths_df)

    monkeypatch.setattr(target=assets, name="download_and_parse_files", value=_fail_once)

    result = materialize([power_time_series_and_metadata], instance=dagster_instance)
    assert result.success
    assert calls == 2

    metadata = pl.read_parquet(env / "NGED" / "metadata.parquet")
    assert set(metadata["time_series_id"].to_list()) == {10, 11}
    power = pl.read_delta(str(env / "NGED" / "power_time_series.delta")).sort(
        PowerTimeSeries.columns_to_sort_by
    )
    PowerTimeSeries.validate(power)
    assert set(power["time_series_id"].unique().to_list()) == {10, 11}


def test_power_time_series_and_metadata_gives_up_after_its_retry_budget(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """A persistent outage still fails, and still reports — the retry only buys the budget."""
    monkeypatch.setattr(
        target=assets.Settings,
        name="get_nged_s3_store",
        value=lambda self: _FakeS3Store(_NGED_FILES),
    )
    monkeypatch.setattr(target=assets, name="_POWER_INGEST_RETRY_DELAY_SECONDS", value=0)
    calls = 0
    retries_raised: list[RetryRequested] = []

    class _RecordingRetryRequested(RetryRequested):
        """Keeps each request the guard raises, so the test can inspect its ``__cause__``."""

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            retries_raised.append(self)

    monkeypatch.setattr(target=assets, name="RetryRequested", value=_RecordingRetryRequested)

    def _always_fail(store: object, paths_df: object) -> None:
        nonlocal calls
        calls += 1
        raise OSError("object store is down")

    monkeypatch.setattr(target=assets, name="download_and_parse_files", value=_always_fail)

    result = materialize(
        [power_time_series_and_metadata], instance=dagster_instance, raise_on_error=False
    )
    assert not result.success
    # A literal, not `_POWER_INGEST_MAX_RETRIES + 1`: computing the expected count from the constant
    # pins only "the budget is honoured", so raising the budget to 99 would keep this green while
    # every failing hour re-listed and re-downloaded NGED's bucket a hundred times.
    assert calls == 3

    # The failure hook reports `__cause__`, so dropping the `from exc` on the guard's raise would
    # silently put us back to Sentry issues titled `RetryRequested`. This is the raising half of
    # that contract; `test_sentry.py` covers the unwrapping half.
    assert [type(request.__cause__) for request in retries_raised] == [OSError] * 3


@pytest.mark.parametrize(
    "interrupt",
    [DagsterExecutionInterruptedError, KeyboardInterrupt, SystemExit],
    ids=["interrupted", "keyboard_interrupt", "system_exit"],
)
def test_power_time_series_and_metadata_does_not_retry_a_cancelled_run(
    interrupt: type[BaseException],
    env: Path,
    monkeypatch: pytest.MonkeyPatch,
    dagster_instance: DagsterInstance,
) -> None:
    """The retry guard wraps everything that reads NGED's bucket, so it is also the place a
    cancellation would be swallowed. It re-raises instead: a run the operator cancelled has to
    stop at once, not read the bucket twice more first.

    ``DagsterExecutionInterruptedError`` is what a production termination actually delivers; the
    other two cover a Ctrl-C at a local ``dg dev``."""
    monkeypatch.setattr(
        target=assets.Settings,
        name="get_nged_s3_store",
        value=lambda self: _FakeS3Store(_NGED_FILES),
    )
    calls = 0

    def _cancel(store: object, paths_df: object) -> None:
        nonlocal calls
        calls += 1
        raise interrupt

    monkeypatch.setattr(target=assets, name="download_and_parse_files", value=_cancel)

    result = materialize(
        [power_time_series_and_metadata], instance=dagster_instance, raise_on_error=False
    )
    assert not result.success
    assert calls == 1


def test_power_time_series_and_metadata_does_not_retry_a_failure_after_the_write(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """The guard stops before the writes deliberately, and this is what that buys.

    Were it extended over them, a bug after the Delta append would be retried; the second
    attempt's ``select_new_rows`` would dedupe the already-written rows to nothing, the body
    would run to the end, and a real failure would land as a green run — every hour, with nothing
    sent to Sentry.
    """
    monkeypatch.setattr(
        target=assets.Settings,
        name="get_nged_s3_store",
        value=lambda self: _FakeS3Store(_NGED_FILES),
    )
    monkeypatch.setattr(target=assets, name="_POWER_INGEST_RETRY_DELAY_SECONDS", value=0)
    calls = 0

    def _boom(*_: object, **__: object) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("a bug in our own code, after the rows have landed")

    # `_PowerTimeSeriesSummary.make_table` is the first statement after the Delta append; patching
    # it on the subclass leaves `_FileListingSummary`'s inherited copy (used inside the guard)
    # alone.
    monkeypatch.setattr(target=assets._PowerTimeSeriesSummary, name="make_table", value=_boom)

    result = materialize(
        [power_time_series_and_metadata], instance=dagster_instance, raise_on_error=False
    )
    assert not result.success
    assert calls == 1


# --- the downloaded-files list -------------------------------------------------------------------

_DELTA = "NGED/power_time_series.delta"
_METADATA = "NGED/metadata.parquet"
_DOWNLOADED_FILES = "NGED/downloaded_files.parquet"


def _key(time_series_id: int, end_hours: int) -> str:
    """The key of a six-hour window ending `end_hours` after 2026-01-01 00:00 UTC."""
    end_ms = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp() * 1000) + end_hours * 3_600_000
    return (
        f"timeseries/{end_ms - 21_600_000}_{end_ms}/TimeSeries_{time_series_id}_"
        "20260101T000000Z_20260101T060000Z.json"
    )


def _window_file(
    time_series_id: int, end_hours: int, values: list[float], information: str | None = None
) -> bytes:
    """A real NGED file's metadata fields, with half-hourly readings ending at the window's end."""
    file_contents = json.loads((_NGED_JSON_DIR / "TimeSeries_11.json").read_text())
    end = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=end_hours)
    fmt = "%Y-%m-%d %H:%M:%S%z"
    file_contents.update(
        TimeSeriesID=time_series_id,
        Information=information,
        data=[
            {
                "value": value,
                "startTime": (end - timedelta(minutes=30 * (len(values) - i))).strftime(fmt),
                "endTime": (end - timedelta(minutes=30 * (len(values) - i - 1))).strftime(fmt),
            }
            for i, value in enumerate(values)
        ],
    )
    return json.dumps(file_contents).encode()


def _data_less_file(time_series_id: int, information: str | None) -> bytes:
    file_contents = json.loads((_NGED_JSON_DIR / "TimeSeries_33_no_data.json").read_text())
    file_contents.update(TimeSeriesID=time_series_id, Information=information)
    return json.dumps(file_contents).encode()


def _use_store(monkeypatch: pytest.MonkeyPatch, store: _FakeS3Store) -> None:
    monkeypatch.setattr(target=assets.Settings, name="get_nged_s3_store", value=lambda self: store)


def _run(instance: DagsterInstance, *, succeeds: bool = True) -> ExecuteInProcessResult:
    result = materialize(
        [power_time_series_and_metadata], instance=instance, raise_on_error=succeeds
    )
    assert result.success == succeeds
    return result


def _power(root: Path) -> pl.DataFrame:
    return pl.read_delta(str(root / _DELTA)).sort(PowerTimeSeries.columns_to_sort_by)


def _stored_note(root: Path, series_id: int) -> str | None:
    metadata = pl.read_parquet(root / _METADATA)
    return metadata.filter(pl.col("time_series_id") == series_id)["information"][0]


def _stored_power(root: Path, series_id: int) -> dict[datetime, float]:
    power = _power(root).filter(pl.col("time_series_id") == series_id)
    return dict(power.select("time", "power").rows())


def _delta_version(root: Path) -> int:
    return DeltaTable(str(root / _DELTA)).version()


class _ReplayArms:
    """Two ingests over identical buckets: the real one, and a reference that downloads everything.

    The reference arm reads an empty downloaded-files list every run, so it downloads the whole
    bucket each time.
    """

    def __init__(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, instance: DagsterInstance
    ) -> None:
        self.roots = {"once": tmp_path / "once", "reference": tmp_path / "reference"}
        self.stores = {arm: _FakeS3Store({}) for arm in self.roots}
        self._monkeypatch = monkeypatch
        self._instance = instance
        self._current_arm = ""
        real_read = assets.read_downloaded_files

        def read(**kwargs: Any) -> Any:
            downloaded_files = real_read(**kwargs)
            return (
                downloaded_files.clear() if self._current_arm == "reference" else downloaded_files
            )

        monkeypatch.setattr(
            target=assets.Settings,
            name="get_nged_s3_store",
            value=lambda settings: self.stores[self._current_arm],
        )
        monkeypatch.setattr(target=assets, name="read_downloaded_files", value=read)

    def put(self, path: str, data: bytes, last_modified: datetime = _FIRST_LAST_MODIFIED) -> None:
        for store in self.stores.values():
            store.put(path, data, last_modified)

    def run_and_compare(self, *, n_new_files: int) -> None:
        """Run both arms, then check the downloads made and that the stored tables are equal."""
        n_files_in_bucket = len(self.stores["once"]._files)
        n_downloads = {}
        for arm, root in self.roots.items():
            self._current_arm = arm
            self._monkeypatch.setenv("DATA_PATH_INTERNAL", str(root))
            self._monkeypatch.setenv("DATA_PATH_DELIVERY", str(root))
            n_before = len(self.stores[arm].requested_paths)
            _run(self._instance)
            n_downloads[arm] = len(self.stores[arm].requested_paths) - n_before
        assert n_downloads == {"once": n_new_files, "reference": n_files_in_bucket}
        assert _power(self.roots["once"]).equals(_power(self.roots["reference"]))
        once_metadata, reference_metadata = (
            pl.read_parquet(self.roots[arm] / _METADATA).sort("time_series_id")
            for arm in ("once", "reference")
        )
        assert once_metadata.equals(reference_metadata)


def test_the_downloaded_files_list_matches_a_download_everything_reference_on_every_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """The ingest that downloads each file once leaves the same power table and metadata parquet,
    after every run, as an ingest that downloads every file in the bucket every run.

    The two arms share the parsing code, so equality alone cannot catch a parsing bug. The
    assertions on stored values below pin the outcomes the replay is designed around.
    """
    monkeypatch.setenv("LOCAL_ARTIFACTS_PATH", str(tmp_path))
    arms = _ReplayArms(tmp_path, monkeypatch, dagster_instance)
    put, run_and_compare = arms.put, arms.run_and_compare
    once_root = arms.roots["once"]

    # Run 1: the first run downloads the whole bucket, in both arms.
    put(_key(10, 6), _window_file(10, 6, [1.0, 2.0]))
    put(_key(11, 6), _window_file(11, 6, [3.0, 4.0]))
    put(_key(12, 6), _window_file(12, 6, [5.0, 6.0], information="series 12 reporting"))
    run_and_compare(n_new_files=3)

    # Run 2: new windows for two series. Series 12 has gone quiet.
    put(_key(10, 12), _window_file(10, 12, [7.0], information="series 10 note"))
    put(_key(11, 12), _window_file(11, 12, [8.0]))
    run_and_compare(n_new_files=2)

    # Run 3: a back-fill two months before the data, with a different note, and a late file more
    # than 3 days before the newest reading of its series.
    put(_key(10, -24 * 60), _window_file(10, -24 * 60, [9.0], information="old back-fill note"))
    put(_key(11, 12 - 24 * 4), _window_file(11, 12 - 24 * 4, [10.0]))
    put(_key(10, 24), _window_file(10, 24, [11.0], information="series 10 newest note"))
    run_and_compare(n_new_files=3)
    assert _stored_note(once_root, 10) == "series 10 newest note"
    old_time = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=12 - 24 * 4)
    assert _stored_power(once_root, 11)[old_time - timedelta(minutes=30)] == 10.0

    # Run 4: a file for series 10 that falls between two windows already downloaded, and a
    # data-less file for the quiet series 12 carrying a new note.
    put(_key(10, 18), _window_file(10, 18, [12.0], information="late-visible note"))
    put(_key(12, 30), _data_less_file(12, "series 12 stopped reporting"))
    run_and_compare(n_new_files=2)
    assert _stored_note(once_root, 10) == "series 10 newest note"
    assert _stored_note(once_root, 12) == "series 12 stopped reporting"

    # Run 5: NGED rewrites series 11's newest key with one earlier reading and a new note.
    n_readings_of_11 = len(_stored_power(once_root, 11))
    put(
        _key(11, 12),
        _window_file(11, 12, [13.0, 8.0], information="rewritten note"),
        last_modified=_FIRST_LAST_MODIFIED + timedelta(hours=1),
    )
    run_and_compare(n_new_files=1)
    assert _stored_note(once_root, 11) == "rewritten note"
    assert len(_stored_power(once_root, 11)) == n_readings_of_11 + 1

    # Run 6: nothing new, so the arm with the list makes no request at all.
    run_and_compare(n_new_files=0)


def test_a_run_with_nothing_new_downloads_nothing_and_leaves_the_files_unchanged(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    store = _FakeS3Store(_NGED_FILES)
    _use_store(monkeypatch, store)
    _run(dagster_instance)
    list_bytes = (env / _DOWNLOADED_FILES).read_bytes()
    version = _delta_version(env)
    n_requests = len(store.requested_paths)

    result = _run(dagster_instance)

    assert len(store.requested_paths) == n_requests
    assert (env / _DOWNLOADED_FILES).read_bytes() == list_bytes
    assert _delta_version(env) == version
    metadata = {
        k: v
        for mat in result.asset_materializations_for_node("power_time_series_and_metadata")
        for k, v in mat.metadata.items()
    }
    assert metadata["metadata_n_new_TimeSeriesIDs"].value == 0


def test_a_malformed_file_fails_the_run_without_a_retry_and_records_nothing(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    store = _FakeS3Store({**_NGED_FILES, _key(10, 100_000): b"{not json"})
    _use_store(monkeypatch, store)

    result = _run(dagster_instance, succeeds=False)

    assert not (env / _DELTA).exists()
    assert not (env / _METADATA).exists()
    assert not (env / _DOWNLOADED_FILES).exists()
    assert len(store.requested_paths) == len(store._files)
    failures = [event for event in result.all_events if event.is_step_failure]
    assert "NgedFileParseError" in str(failures[0].step_failure_data.error)


def test_the_downloaded_files_list_is_never_ahead_of_the_rows(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    store = _FakeS3Store(_NGED_FILES)
    _use_store(monkeypatch, store)
    monkeypatch.setattr(target=assets, name="_POWER_INGEST_RETRY_DELAY_SECONDS", value=0)
    real_write = assets.write_power_time_series
    monkeypatch.setattr(
        target=assets,
        name="write_power_time_series",
        value=lambda **_: (_ for _ in ()).throw(RuntimeError("the append crashed")),
    )

    _run(dagster_instance, succeeds=False)
    assert not (env / _DOWNLOADED_FILES).exists()

    monkeypatch.setattr(target=assets, name="write_power_time_series", value=real_write)
    _run(dagster_instance)
    assert set(_power(env)["time_series_id"].unique().to_list()) == {10, 11}


def test_a_failed_downloaded_files_write_is_reported_and_the_next_run_downloads_again(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    store = _FakeS3Store(_NGED_FILES)
    _use_store(monkeypatch, store)
    real_write = assets.write_downloaded_files

    def boom(**_: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(target=assets, name="write_downloaded_files", value=boom)
    reported: list[tuple[BaseException, list[str] | None]] = []
    monkeypatch.setattr(
        target=assets,
        name="report_asset_degradation",
        value=lambda asset_name, exc, fingerprint=None: reported.append((exc, fingerprint)),
    )

    _run(dagster_instance)

    n_rows = _power(env).height
    assert n_rows > 0
    assert not (env / _DOWNLOADED_FILES).exists()
    [(exc, fingerprint)] = reported
    assert str(env / _DOWNLOADED_FILES) in str(exc)
    assert fingerprint == ["downloaded_files_write_failed"]

    monkeypatch.setattr(target=assets, name="write_downloaded_files", value=real_write)
    _run(dagster_instance)

    assert len(store.requested_paths) == 2 * len(_NGED_FILES)
    assert _power(env).height == n_rows


@pytest.mark.parametrize("rebuilt", [_DELTA, _METADATA], ids=["power_table", "metadata_table"])
def test_a_rebuilt_table_downloads_the_whole_bucket_again(
    rebuilt: str,
    env: Path,
    monkeypatch: pytest.MonkeyPatch,
    dagster_instance: DagsterInstance,
) -> None:
    store = _FakeS3Store(_NGED_FILES)
    _use_store(monkeypatch, store)
    _run(dagster_instance)
    expected_power = _power(env)
    if rebuilt == _DELTA:
        shutil.rmtree(env / rebuilt)
    else:
        (env / rebuilt).unlink()

    _run(dagster_instance)

    assert len(store.requested_paths) == 2 * len(_NGED_FILES)
    assert _power(env).equals(expected_power)
    assert set(pl.read_parquet(env / _METADATA)["time_series_id"].to_list()) == {10, 11}


def test_a_corrupt_downloaded_files_list_stops_the_run_without_a_retry_or_a_listing(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    store = _FakeS3Store(_NGED_FILES)
    _use_store(monkeypatch, store)
    _run(dagster_instance)
    (env / _DOWNLOADED_FILES).write_bytes(b"not a parquet file")
    n_lists = store.n_list_calls
    reads = 0
    real_read = assets.read_downloaded_files

    def counting_read(**kwargs: Any) -> Any:
        nonlocal reads
        reads += 1
        return real_read(**kwargs)

    monkeypatch.setattr(target=assets, name="read_downloaded_files", value=counting_read)

    result = _run(dagster_instance, succeeds=False)

    assert reads == 1
    assert store.n_list_calls == n_lists
    failures = [event for event in result.all_events if event.is_step_failure]
    assert DownloadedFilesError.__name__ in str(failures[0].step_failure_data.error)


def test_the_ingest_asset_is_limited_to_one_run_at_a_time() -> None:
    assert power_time_series_and_metadata.op.pool == "NGED_INGEST"


# --- h3_grid_weights -----------------------------------------------------------------------------


def test_h3_grid_weights_materialises_and_writes_parquet(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """Materialise ``h3_grid_weights`` against a small stand-in boundary, and assert a valid parquet
    lands on disk.

    The real GB boundary buffers for ~30 s, and is exercised in ``packages/geo`` instead.
    """
    # A 1×1-degree box over central GB — enough to yield several H3 cells, milliseconds to compute.
    monkeypatch.setattr(
        target=assets, name="load_gb_boundary", value=lambda: shapely.box(-2.0, 52.0, -1.0, 53.0)
    )

    result = materialize([h3_grid_weights], instance=dagster_instance)
    assert result.success

    weights = pl.read_parquet(env / "h3_grid_weights.parquet")
    H3GridWeights.validate(weights)
    assert weights.height > 0


# --- ecmwf_ens -----------------------------------------------------------------------------------


def _check_evaluations(result: ExecuteInProcessResult) -> dict[str, AssetCheckEvaluation]:
    """The run's asset-check evaluations, keyed by check name.

    ``ecmwf_ens`` emits three independent checks, so tests look theirs up by name rather than
    relying on the order Dagster happens to report them in.
    """
    return {
        evaluation.check_name: evaluation for evaluation in result.get_asset_check_evaluations()
    }


def test_ecmwf_ens_materialises_and_writes_nwp(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """Happy path with the download/convert pipeline stubbed: the partition key parses into
    ``nwp_init_time`` (passed to ``open_ecmwf_ens_run``) and the converted frame is written to
    the NWP Delta table via ``write_nwp``."""
    _write_h3_grid_weights(Settings().h3_grid_weights_path)
    # After 2024-11-12, when categorical_precipitation_type_surface became a non-null Nwp variable.
    init_time = datetime(year=2024, month=12, day=1, tzinfo=UTC)
    captured: dict[str, datetime] = {}

    def _open(*, nwp_init_time: datetime, h3_grid: object) -> object:
        captured["nwp_init_time"] = nwp_init_time
        return object()

    monkeypatch.setattr(target=assets, name="open_ecmwf_ens_run", value=_open)
    monkeypatch.setattr(
        target=assets, name="download_ecmwf_ens_data", value=lambda ds: _make_downloaded_ds()
    )
    monkeypatch.setattr(
        target=assets,
        name="convert_nwp_xarray_dataset_to_polars_dataframe",
        value=lambda ds, h3_grid: _make_nwp(init_time),
    )

    result = materialize([ecmwf_ens], partition_key="2024-12-01", instance=dagster_instance)
    assert result.success
    # The partition key is parsed into nwp_init_time and handed to open_ecmwf_ens_run...
    assert captured["nwp_init_time"] == init_time
    # ...and the converted frame is actually persisted via write_nwp (all 4 rows round-trip).
    written = pl.read_delta(Settings().nwp_data_path)
    assert written.height == 4
    # The clean run emits a passing data-quality check.
    assert _check_evaluations(result)["nwp_has_no_unexpected_nulls"].passed
    # The run's observed shape is published on the materialisation itself, not only on the
    # completeness check, so drift stays visible in the Dagster UI timeline on a passing run too.
    # (The tiny stub frame is not a full ECMWF ENS run, so nwp_run_is_complete does WARN here —
    # that path is asserted in test_ecmwf_ens_warns_on_incomplete_run_but_still_materialises.)
    (materialisation,) = result.asset_materializations_for_node("ecmwf_ens")
    assert {
        "n_rows",
        "n_ensemble_members",
        "n_valid_times",
        "n_h3_cells",
        "valid_time_min",
        "valid_time_max",
    } <= set(materialisation.metadata)
    assert materialisation.metadata["n_ensemble_members"].value == 4
    assert materialisation.metadata["n_valid_times"].value == 4
    assert materialisation.metadata["n_h3_cells"].value == 4

    # A run with nothing wrong at either level must not be described as tolerated corruption.
    description = str(_check_evaluations(result)["nwp_has_no_unexpected_nulls"].description)
    assert "Known upstream ECMWF ENS corruption" not in description


def test_ecmwf_ens_warns_on_scattered_nulls_but_still_materialises(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """Scattered per-pixel nulls in a de-accumulated variable (the known upstream ECMWF ENS
    corruption) are tolerated: the run still materialises, and the data-quality check WARNs."""
    _write_h3_grid_weights(Settings().h3_grid_weights_path)
    init_time = datetime(year=2024, month=12, day=1, tzinfo=UTC)

    # One (member, valid_time) slice across three h3 cells, one cell's precipitation nulled.
    scattered = _make_nwp(init_time, n=3).with_columns(
        init_time=pl.lit(init_time),
        valid_time=pl.lit(init_time + timedelta(hours=3)),
        ensemble_member=pl.lit(0, dtype=pl.Int8),
        precipitation_surface=pl.Series([0.001, None, 0.001], dtype=pl.Float32),
    )
    # `object` cannot be inlined in place of this stub: the real function is called with
    # keyword arguments, which `object()` rejects.
    monkeypatch.setattr(
        target=assets,
        name="open_ecmwf_ens_run",
        value=lambda *, nwp_init_time, h3_grid: object(),  # noqa: PLW0108
    )
    monkeypatch.setattr(
        target=assets, name="download_ecmwf_ens_data", value=lambda ds: _make_downloaded_ds()
    )
    monkeypatch.setattr(
        target=assets,
        name="convert_nwp_xarray_dataset_to_polars_dataframe",
        value=lambda ds, h3_grid: scattered,
    )

    result = materialize([ecmwf_ens], partition_key="2024-12-01", instance=dagster_instance)
    assert result.success  # tolerated — the run is NOT failed
    assert pl.read_delta(Settings().nwp_data_path).height == 3  # data was persisted
    evaluation = _check_evaluations(result)["nwp_has_no_unexpected_nulls"]
    assert not evaluation.passed  # WARN: the scatter is surfaced
    assert evaluation.metadata["n_null_h3_cells"].value == 1
    assert evaluation.metadata["n_whole_null_h3_slices"].value == 0
    # Both halves of the split are emitted, not just the whole-null one: the operations runbook
    # names `n_scattered_h3_slices` as a number to read off this check.
    assert evaluation.metadata["n_scattered_h3_slices"].value == 1
    assert (
        "Stored H3 cells: 1 null cell(s) in precipitation_surface, across 1 partly-null and 0 "
        "wholly-null (member, valid_time) slice(s)" in str(evaluation.description)
    )


def test_ecmwf_ens_reports_whole_null_slices_in_its_quality_check(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """A wholly-null (member, valid_time) slice — the 2026-08-09 class — reaches the operator as a
    WARN that counts it, rather than passing silently.

    Scoped to the *reporting* half deliberately: the converter is stubbed here, so `Nwp.validate`
    never sees this frame, and that it no longer rejects such a slice is pinned by
    ``test_whole_slice_deaccumulated_null_beyond_lead0_is_tolerated`` in the contracts package.
    What fails on ``main`` is the count: ``assess_nwp_quality`` filtered wholly-null slices out
    of its report entirely, so the check passed and the missing field was surfaced nowhere.
    """
    _write_h3_grid_weights(Settings().h3_grid_weights_path)
    init_time = datetime(year=2024, month=12, day=1, tzinfo=UTC)

    # `_make_nwp` gives each row its own (member, valid_time), so nulling one row's precipitation
    # empties one whole slice of three while the other two stay intact.
    one_slice_missing = _make_nwp(init_time, n=3).with_columns(
        precipitation_surface=pl.Series([None, 0.001, 0.001], dtype=pl.Float32)
    )
    # `object` cannot be inlined in place of this stub: the real function is called with
    # keyword arguments, which `object()` rejects.
    monkeypatch.setattr(
        target=assets,
        name="open_ecmwf_ens_run",
        value=lambda *, nwp_init_time, h3_grid: object(),  # noqa: PLW0108
    )
    monkeypatch.setattr(
        target=assets, name="download_ecmwf_ens_data", value=lambda ds: _make_downloaded_ds()
    )
    monkeypatch.setattr(
        target=assets,
        name="convert_nwp_xarray_dataset_to_polars_dataframe",
        value=lambda ds, h3_grid: one_slice_missing,
    )

    result = materialize([ecmwf_ens], partition_key="2024-12-01", instance=dagster_instance)
    assert result.success  # tolerated — the run is NOT failed
    assert pl.read_delta(Settings().nwp_data_path).height == 3  # data was persisted
    evaluation = _check_evaluations(result)["nwp_has_no_unexpected_nulls"]
    assert not evaluation.passed  # WARN: the missing slice is surfaced
    assert evaluation.metadata["n_whole_null_h3_slices"].value == 1
    assert evaluation.metadata["n_null_h3_cells"].value == 1
    # The mirror of the scattered test above: the same slice must be counted once, on one side of
    # the split, so the two metadata fields cannot both claim it.
    assert evaluation.metadata["n_scattered_h3_slices"].value == 0


def test_ecmwf_ens_retries_when_a_variable_is_wholly_missing(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """``NwpVariableWhollyMissing`` → ``RetryRequested``, not a failed partition.

    An all-null weather column is one way a half-published upstream run reads, and Dynamical.org
    republishes a defective one — the 2026-08-09 repair landed 3h25m later, inside this budget.

    The stub calls the *real* ``Nwp.validate``, so this pins the whole chain the widened ``try``
    exists for: an empty column raises from validation, which the converter calls, which sits
    past where ``main``'s ``try`` block ended. It fails on ``main``, where the exception escaped
    as a hard failure.
    """
    _write_h3_grid_weights(Settings().h3_grid_weights_path)
    init_time = datetime(year=2024, month=12, day=1, tzinfo=UTC)
    # A run whose radiation column carries no weather at all, exactly as the converter would hand
    # it over: `_make_nwp` gives each row its own (member, valid_time), so nulling every row empties
    # the column across every slice beyond lead-0.
    wholly_missing = _make_nwp(init_time, n=3).with_columns(
        downward_short_wave_radiation_flux_surface=pl.Series([None] * 3, dtype=pl.Float32)
    )

    def _convert_via_real_validation(ds: object, h3_grid: object) -> pt.DataFrame[Nwp]:
        return Nwp.validate(wholly_missing)

    _stub_ecmwf_download(monkeypatch=monkeypatch, init_time=init_time)
    monkeypatch.setattr(
        target=assets,
        name="convert_nwp_xarray_dataset_to_polars_dataframe",
        value=_convert_via_real_validation,
    )

    requests = _materialize_expecting_retries(
        monkeypatch=monkeypatch, instance=dagster_instance, partition_key=_today_key()
    )

    assert [type(request.__cause__) for request in requests] == [NwpVariableWhollyMissing] * (
        _ECMWF_ENS_MAX_RETRIES + 1
    )
    # Validation runs before the Delta write, so a retry leaves no partial partition behind.
    assert not Path(Settings().nwp_data_path).exists()


def test_ecmwf_ens_warns_on_incomplete_run_but_still_materialises(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """A short run is landed anyway and surfaced as a WARN — an incomplete upstream run is absent
    input, so we keep the rows that arrived rather than discarding the whole partition.

    ``_make_nwp`` builds 4 rows carrying 4 distinct members, valid_times and cells (a diagonal,
    not a cross-product), which is nothing like a complete 51 x 85 x 1 ECMWF ENS run.
    """
    _write_h3_grid_weights(Settings().h3_grid_weights_path)
    init_time = datetime(year=2024, month=12, day=1, tzinfo=UTC)
    _stub_ecmwf_download(monkeypatch=monkeypatch, init_time=init_time)

    result = materialize([ecmwf_ens], partition_key="2024-12-01", instance=dagster_instance)
    assert result.success  # WARN, not a failure: the partial run is NOT thrown away
    assert pl.read_delta(Settings().nwp_data_path).height == 4  # data was persisted

    evaluation = _check_evaluations(result)["nwp_run_is_complete"]
    assert not evaluation.passed
    assert evaluation.severity == AssetCheckSeverity.WARN
    # The single-cell H3 grid weights fixture is where the cell expectation comes from.
    assert evaluation.metadata["expected_n_h3_cells"].value == 1
    assert evaluation.metadata["n_h3_cells"].value == 4
    # The stub frame carries members 0-3, so 4-50 of the 51 ECMWF ENS members are named as absent.
    assert evaluation.metadata["missing_ensemble_members"].value == list(range(4, 51))


class _FakePanic(BaseException):
    """Stands in for pyo3's ``PanicException``, which also derives from ``BaseException``.

    The real class cannot be imported: each compiled extension defines its own.
    """


def _never_called(check_name: str, exc: BaseException) -> None:
    """Stand in for ``report_check_degradation`` on a path that must not report to Sentry.

    Parameter names match the real function's, because the caller passes them by keyword.
    """
    raise AssertionError(
        f"report_check_degradation({check_name!r}, {exc!r}) should not have been called"
    )


def _stub_ecmwf_download(monkeypatch: pytest.MonkeyPatch, init_time: datetime) -> None:
    """Stub out the download/convert pipeline so it yields ``_make_nwp(init_time)``.

    ``open_ecmwf_ens_run`` is called with keyword arguments, which a bare ``object()`` rejects.
    """
    monkeypatch.setattr(
        target=assets,
        name="open_ecmwf_ens_run",
        value=lambda *, nwp_init_time, h3_grid: object(),  # noqa: PLW0108
    )
    monkeypatch.setattr(
        target=assets, name="download_ecmwf_ens_data", value=lambda ds: _make_downloaded_ds()
    )
    monkeypatch.setattr(
        target=assets,
        name="convert_nwp_xarray_dataset_to_polars_dataframe",
        value=lambda ds, h3_grid: _make_nwp(init_time),
    )


def test_ecmwf_ens_assesses_before_writing(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """The check *results* are built before the Delta write, not just the assessments.

    ``_nwp_quality_check_result`` reaches ``_nwp_null_slices_metadata``, which sorts the affected
    frame and builds a ``TableRecord`` per row — the most raise-prone code in the block, and the
    half a partial fix would leave below the write.
    """
    _write_h3_grid_weights(Settings().h3_grid_weights_path)
    init_time = datetime(year=2024, month=12, day=1, tzinfo=UTC)
    _stub_ecmwf_download(monkeypatch=monkeypatch, init_time=init_time)

    table_existed: list[bool] = []
    real_build = assets._nwp_quality_check_result

    def _record_then_build(
        report: NwpQualityReport, upstream: UpstreamNullRate
    ) -> AssetCheckResult:
        table_existed.append(Path(Settings().nwp_data_path).exists())
        return real_build(report=report, upstream=upstream)

    monkeypatch.setattr(target=assets, name="_nwp_quality_check_result", value=_record_then_build)

    result = materialize([ecmwf_ens], partition_key="2024-12-01", instance=dagster_instance)
    assert result.success
    assert table_existed == [False]


def test_ecmwf_ens_re_materialising_a_partition_does_not_duplicate_rows(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """Re-running a landed partition replaces its rows instead of landing a second copy of the run.

    ``Nwp.validate`` sees only the frame in hand, so a duplicated key would reach the table
    silently and fan every later ``Nwp.scan_delta`` read out.
    """
    _write_h3_grid_weights(Settings().h3_grid_weights_path)
    init_time = datetime(year=2024, month=12, day=1, tzinfo=UTC)
    _stub_ecmwf_download(monkeypatch=monkeypatch, init_time=init_time)

    for _ in range(2):
        result = materialize([ecmwf_ens], partition_key="2024-12-01", instance=dagster_instance)
        assert result.success

    assert pl.read_delta(Settings().nwp_data_path).height == 4


def test_ecmwf_ens_publishes_both_null_populations(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """A run whose stored cells are clean still reports the corruption the feed sent.

    That combination — every H3 key zero, the grid-point keys non-zero — is the state this
    measure exists for, and the one that was indistinguishable from a perfect run before it. The
    converter is stubbed, so this pins the plumbing and the description, not the aggregation
    itself.
    """
    _write_h3_grid_weights(Settings().h3_grid_weights_path)
    init_time = datetime(year=2024, month=12, day=1, tzinfo=UTC)
    _stub_ecmwf_download(monkeypatch=monkeypatch, init_time=init_time)
    monkeypatch.setattr(
        target=assets,
        name="download_ecmwf_ens_data",
        value=lambda ds: _make_downloaded_ds(n_null_grid_points=1),
    )

    result = materialize([ecmwf_ens], partition_key="2024-12-01", instance=dagster_instance)
    assert result.success

    evaluation = _check_evaluations(result)["nwp_has_no_unexpected_nulls"]
    assert evaluation.passed  # no stored cell is null...
    assert evaluation.metadata["n_null_h3_cells"].value == 0
    assert evaluation.metadata["n_null_nwp_grid_points"].value == 1  # ...but the feed was corrupt
    assert evaluation.metadata["n_total_nwp_grid_points"].value == 9  # 3 variables x 1 step x 3
    assert evaluation.metadata["null_nwp_grid_point_fraction"].value == pytest.approx(1 / 9)
    assert evaluation.metadata["n_affected_nwp_slices"].value == 1
    assert evaluation.metadata["affected_nwp_variables"].value == ["precipitation_surface"]
    # The per-variable table is on this check too — the runbook sends the operator to it for the
    # de-accumulated population as well.
    per_variable = evaluation.metadata["per_nwp_variable"].value
    assert isinstance(per_variable, TableMetadataValue)
    assert len(per_variable.records) == len(Nwp.deaccumulated_var_names)

    # The description must name both populations, or it claims the health of the one it read, and
    # must lead with the grid-point clause — the signal this check exists to surface.
    description = str(evaluation.description)
    assert description.startswith("Raw NWP grid: 1 of 9 grid point(s) null")
    assert "(11.1111%)" in description  # rendered as a percentage, not a bare ratio
    assert "precipitation_surface" in description
    assert "Stored H3 cells: 0 null cell(s)" in description
    assert "Known upstream ECMWF ENS corruption" in description

    # Published on the materialisation too, so the trend plots on the asset timeline rather than
    # only appearing on the runs bad enough to warn.
    (materialisation,) = result.asset_materializations_for_node("ecmwf_ens")
    assert materialisation.metadata["n_null_nwp_grid_points"].value == 1
    assert materialisation.metadata["null_nwp_grid_point_fraction"].value == pytest.approx(1 / 9)


@pytest.mark.parametrize(
    ("n_nulls", "expected_passed", "expected_description", "expected_corrupt_rows"),
    [
        (0, True, "No nulls in 54 instantaneous-variable grid points.", []),
        # Two nulls in *one* slice, so the table's two count columns hold different numbers and
        # swapping them is visible. The nulls fill two of the step's three grid points, not all
        # three, because a slice empty at every grid point is retried before any check runs. The
        # stubbed converter never sees these nulls.
        (
            2,
            False,
            (
                "2 of 54 instantaneous-variable grid point(s) null (3.7037%) in temperature_2m, "
                "across 1 (variable, member, step) slice(s)."
            ),
            [
                {
                    "variable": "temperature_2m",
                    "n_null_grid_points": 2,
                    "n_affected_slices": 1,
                    "n_total_grid_points": 6,
                }
            ],
        ),
    ],
)
def test_ecmwf_ens_flags_instantaneous_nulls_the_aggregation_absorbed(
    env: Path,
    monkeypatch: pytest.MonkeyPatch,
    dagster_instance: DagsterInstance,
    n_nulls: int,
    expected_passed: bool,
    expected_description: str,
    expected_corrupt_rows: list[dict[str, object]],
) -> None:
    """A null in a variable that is never legitimately null fails its own check, and only its own.

    This is the whole reason the two populations get separate checks: the corrupt run below is a
    *pass* for the de-accumulated nulls, which are tolerated, and a *fail* for the instantaneous
    one, which is not. A single check over both would have to pick one of those answers. The
    clean case is here because a zero threshold that always failed would satisfy the corrupt case
    alone.
    """
    _write_h3_grid_weights(Settings().h3_grid_weights_path)
    init_time = datetime(year=2024, month=12, day=1, tzinfo=UTC)
    _stub_ecmwf_download(monkeypatch=monkeypatch, init_time=init_time)
    monkeypatch.setattr(
        target=assets,
        name="download_ecmwf_ens_data",
        value=lambda ds: _make_downloaded_ds(n_null_instantaneous_grid_points=n_nulls),
    )

    result = materialize([ecmwf_ens], partition_key="2024-12-01", instance=dagster_instance)
    assert result.success  # WARN, not a failure: the aggregation already absorbed this
    assert pl.read_delta(Settings().nwp_data_path).height == 4  # and the run still lands

    evaluations = _check_evaluations(result)
    assert evaluations["nwp_has_no_unexpected_nulls"].passed  # the tolerated population is clean
    evaluation = evaluations["nwp_instantaneous_variables_have_no_nulls"]
    assert evaluation.passed is expected_passed  # one null grid point is enough, unlike the other
    assert evaluation.severity == AssetCheckSeverity.WARN
    assert evaluation.metadata["n_null_nwp_grid_points"].value == n_nulls
    # 9 instantaneous variables x 2 steps x 3 grid points — lead-0 included, unlike the other check.
    # Counting the de-accumulated or categorical variables too, or reading the contract's names
    # rather than the download's, moves this number.
    assert evaluation.metadata["n_total_nwp_grid_points"].value == 54
    assert str(evaluation.description) == expected_description

    # The per-variable table separates one bad variable from nine, which the totals cannot.
    per_variable = evaluation.metadata["per_nwp_variable"].value
    assert isinstance(per_variable, TableMetadataValue)  # narrows before reading `.records`
    # The declared schema is what names the columns in the Dagster UI, and is what makes a clean
    # run render an empty table rather than nothing at all.
    assert [column.name for column in per_variable.schema.columns] == [
        "variable",
        "n_null_grid_points",
        "n_affected_slices",
        "n_total_grid_points",
    ]
    assert len(per_variable.records) == 9
    corrupt = [record.data for record in per_variable.records if record.data["n_null_grid_points"]]
    assert corrupt == expected_corrupt_rows


@pytest.mark.parametrize(
    ("check_name", "expected"),
    [
        ("nwp_has_no_unexpected_nulls", assets._NWP_QUALITY_CHECK_DESCRIPTION),
        ("nwp_instantaneous_variables_have_no_nulls", assets._NWP_INSTANTANEOUS_CHECK_DESCRIPTION),
    ],
)
def test_nwp_check_specs_carry_a_standing_description(check_name: str, expected: str) -> None:
    """Only the spec's description reaches the Checks view before any run has happened.

    ``blocking`` is asserted here too: no check in this repo may fail the thing it warns about
    (inherent stability, rule 6).
    """
    (spec,) = [spec for spec in ecmwf_ens.check_specs if spec.name == check_name]

    assert spec.description == expected
    assert spec.blocking is False


@pytest.mark.parametrize("raiser", [RuntimeError, _FakePanic])
@pytest.mark.parametrize(
    "assessment",
    ["assess_nwp_quality", "assess_upstream_grid_point_nulls", "assess_nwp_run_completeness"],
)
def test_ecmwf_ens_lands_the_run_when_an_assessment_fails(
    env: Path,
    monkeypatch: pytest.MonkeyPatch,
    dagster_instance: DagsterInstance,
    raiser: type[BaseException],
    assessment: str,
) -> None:
    """A bug in any per-run assessment degrades to failed WARN results; the run still lands.

    Every assessment is covered, because the guard only protects the calls *inside* it: one
    lifted above the ``try`` would fail the partition, and that is reachable — a download missing
    one de-accumulated variable makes the grid-point counter raise ``KeyError``.

    All three declared checks must still be emitted — Dagster fails the step for a missing result
    *or* for one carrying no ``check_name``. ``_FakePanic`` is the case that fails if someone
    narrows the guard to ``except Exception``: the assessments run Polars sorts and group-bys,
    and a pyo3 panic from one derives from ``BaseException``.
    """
    _write_h3_grid_weights(Settings().h3_grid_weights_path)
    init_time = datetime(year=2024, month=12, day=1, tzinfo=UTC)
    _stub_ecmwf_download(monkeypatch=monkeypatch, init_time=init_time)

    def _raise(*_: object, **__: object) -> NwpQualityReport:
        raise raiser("assessment is broken")

    monkeypatch.setattr(target=assets, name=assessment, value=_raise)

    result = materialize([ecmwf_ens], partition_key="2024-12-01", instance=dagster_instance)
    assert result.success  # the ingest is not failed by a bug in a *reporting* function
    assert pl.read_delta(Settings().nwp_data_path).height == 4  # written exactly once

    evaluations = _check_evaluations(result)
    assert set(evaluations) == {
        "nwp_has_no_unexpected_nulls",
        "nwp_instantaneous_variables_have_no_nulls",
        "nwp_run_is_complete",
    }
    for evaluation in evaluations.values():
        assert not evaluation.passed
        assert evaluation.severity == AssetCheckSeverity.WARN
        assert "assessment is broken" in str(evaluation.description)

    # The shape and upstream keys come from reports that never got built. Both fallbacks must be
    # assigned in the `except` branch: `MaterializeResult` is built *outside* the guard, so an
    # unbound name there would fail the partition — the failure the guard exists to prevent.
    (materialisation,) = result.asset_materializations_for_node("ecmwf_ens")
    assert materialisation.metadata["n_rows"].value == 4
    assert "n_ensemble_members" not in materialisation.metadata
    assert "null_nwp_grid_point_fraction" not in materialisation.metadata


def test_ecmwf_ens_reports_a_degraded_assessment_to_sentry_once(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """Sentry is told, exactly once, that the assessment could not run.

    Once because all three checks share one guard; at all because not failing the run means the
    ``sentry_capture_failure`` hook no longer fires.
    """
    _write_h3_grid_weights(Settings().h3_grid_weights_path)
    init_time = datetime(year=2024, month=12, day=1, tzinfo=UTC)
    _stub_ecmwf_download(monkeypatch=monkeypatch, init_time=init_time)

    def _raise(nwp: pt.DataFrame[Nwp]) -> NwpQualityReport:
        raise RuntimeError("assessment is broken")

    reported: list[tuple[str, BaseException]] = []
    monkeypatch.setattr(target=assets, name="assess_nwp_quality", value=_raise)
    monkeypatch.setattr(
        target=assets,
        name="report_check_degradation",
        value=lambda check_name, exc: reported.append((check_name, exc)),
    )

    result = materialize([ecmwf_ens], partition_key="2024-12-01", instance=dagster_instance)
    assert result.success
    assert [name for name, _ in reported] == ["nwp_has_no_unexpected_nulls"]


def test_ecmwf_ens_re_raises_a_cancelled_run_without_writing(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one thing the guard must *not* swallow — and here swallowing it would also write.

    A swallowed cancellation falls straight through into ``write_nwp``, landing rows under a
    partition Dagster then marks cancelled.
    """
    _write_h3_grid_weights(Settings().h3_grid_weights_path)
    init_time = datetime(year=2024, month=12, day=1, tzinfo=UTC)
    _stub_ecmwf_download(monkeypatch=monkeypatch, init_time=init_time)

    def _cancel(nwp: pt.DataFrame[Nwp]) -> NwpQualityReport:
        raise DagsterExecutionInterruptedError

    monkeypatch.setattr(target=assets, name="assess_nwp_quality", value=_cancel)
    monkeypatch.setattr(target=assets, name="report_check_degradation", value=_never_called)

    with (
        build_asset_context(partition_key="2024-12-01") as context,
        pytest.raises(DagsterExecutionInterruptedError),
    ):
        ecmwf_ens(context)

    assert not Path(Settings().nwp_data_path).exists()


def _today_key() -> str:
    """Today's partition key: always 0 to 24 hours old, so well inside the retry age limit."""
    return datetime.now(UTC).date().isoformat()


def _materialize_expecting_retries(
    monkeypatch: pytest.MonkeyPatch, instance: DagsterInstance, partition_key: str
) -> list[RetryRequested]:
    """Materialise ``ecmwf_ens`` until its retries run out, and return each retry it requested.

    Goes through ``materialize`` because a partition young enough to retry reads
    ``context.retry_number``, and ``context.retry_number`` raises ``AttributeError`` under direct
    invocation. The wait between retries is set to 0 in each request, after the request is recorded,
    so the recorded ``seconds_to_wait`` is the value the asset requested.
    """
    requests: list[RetryRequested] = []
    requested_kwargs: list[dict[str, Any]] = []

    class _RecordingRetryRequested(RetryRequested):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**{**kwargs, "seconds_to_wait": 0})
            requested_kwargs.append(kwargs)
            requests.append(self)

    monkeypatch.setattr(target=assets, name="RetryRequested", value=_RecordingRetryRequested)
    result = materialize(
        [ecmwf_ens], partition_key=partition_key, instance=instance, raise_on_error=False
    )
    assert not result.success
    for kwargs in requested_kwargs:
        assert kwargs == {
            "max_retries": _ECMWF_ENS_MAX_RETRIES,
            "seconds_to_wait": _ECMWF_ENS_RETRY_DELAY_SECONDS,
        }
    return requests


@pytest.mark.parametrize("raising_step", ["open", "empty_slice_check"])
def test_ecmwf_ens_retries_when_run_not_yet_available(
    env: Path,
    monkeypatch: pytest.MonkeyPatch,
    dagster_instance: DagsterInstance,
    raising_step: str,
) -> None:
    """``NwpRunNotYetAvailable`` → ``RetryRequested`` with the asset's configured retry budget,
    so a not-yet-published run waits rather than failing outright. The asset retries whether the
    run is absent from the catalog or the downloaded run has an empty slice."""
    _write_h3_grid_weights(Settings().h3_grid_weights_path)

    def _raise_not_available(*, nwp_init_time: datetime, h3_grid: object) -> None:
        raise NwpRunNotYetAvailable

    if raising_step == "open":
        monkeypatch.setattr(target=assets, name="open_ecmwf_ens_run", value=_raise_not_available)
    else:
        # The real check runs, on a downloaded run whose only slice at lead 6 h is empty at all
        # three grid points.
        monkeypatch.setattr(
            target=assets, name="open_ecmwf_ens_run", value=lambda **kwargs: object()
        )
        monkeypatch.setattr(
            target=assets,
            name="download_ecmwf_ens_data",
            value=lambda ds_lazy: _make_downloaded_ds(n_null_instantaneous_grid_points=3),
        )

    requests = _materialize_expecting_retries(
        monkeypatch=monkeypatch, instance=dagster_instance, partition_key=_today_key()
    )

    assert [type(request.__cause__) for request in requests] == [NwpRunNotYetAvailable] * (
        _ECMWF_ENS_MAX_RETRIES + 1
    )


def test_ecmwf_ens_warns_sentry_once_on_the_first_failed_attempt(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """A partition that will be retried sends one Sentry warning, on the first attempt and on no
    later attempt.

    The download count recorded when the warning is sent pins that the warning came from the first
    attempt: a warning sent on the last attempt would also be sent exactly once.
    """
    _write_h3_grid_weights(Settings().h3_grid_weights_path)
    downloads = 0
    warnings: list[tuple[str, BaseException, int]] = []

    def _fail_to_download(ds_lazy: object) -> None:
        nonlocal downloads
        downloads += 1
        raise NwpRunNotYetAvailable("not ready")

    def _record_warning(asset_name: str, exc: BaseException) -> None:
        warnings.append((asset_name, exc, downloads))

    monkeypatch.setattr(target=assets, name="open_ecmwf_ens_run", value=lambda **kwargs: object())
    monkeypatch.setattr(target=assets, name="download_ecmwf_ens_data", value=_fail_to_download)
    monkeypatch.setattr(target=assets, name="report_asset_retry", value=_record_warning)

    partition_key = _today_key()
    _materialize_expecting_retries(
        monkeypatch=monkeypatch, instance=dagster_instance, partition_key=partition_key
    )

    assert downloads == _ECMWF_ENS_MAX_RETRIES + 1
    ((asset_name, exc, downloads_when_warned),) = warnings
    assert asset_name == "ecmwf_ens"
    assert downloads_when_warned == 1
    assert f"partition {partition_key}" in "".join(exc.__notes__)


@pytest.mark.parametrize(
    "error", [NwpRunNotYetAvailable("not ready"), NwpVariableWhollyMissing("empty column")]
)
def test_ecmwf_ens_fails_at_once_for_a_run_too_old_to_retry(
    env: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """For a run more than 36 hours old, ``ecmwf_ens`` re-raises the original exception with the
    partition noted on it, and neither retries nor warns Sentry. This test fails on ``main``, where
    the asset requests a retry instead."""
    _write_h3_grid_weights(Settings().h3_grid_weights_path)
    warned: list[object] = []

    def _raise(ds_lazy: object) -> None:
        raise error

    monkeypatch.setattr(target=assets, name="open_ecmwf_ens_run", value=lambda **kwargs: object())
    monkeypatch.setattr(target=assets, name="download_ecmwf_ens_data", value=_raise)
    monkeypatch.setattr(
        target=assets, name="report_asset_retry", value=lambda **kw: warned.append(kw)
    )

    # `build_asset_context()` defaults to its own `DagsterInstance.ephemeral()`
    # (<https://openclimatefix.github.io/nged-substation-forecast/architecture/testing/>) and is
    # used as a context manager here for the same reason
    # `dagster_instance` is a fixture: entering it makes disposal happen deterministically at
    # `__exit__`, rather than depending on `__del__` running via garbage collection, which the
    # traceback captured by `pytest.raises` delays past this test — see the fixture's docstring.
    with (
        build_asset_context(partition_key="2024-05-01") as context,
        pytest.raises(type(error)) as exc_info,
    ):
        ecmwf_ens(context)

    assert exc_info.value is error
    assert "partition 2024-05-01" in "".join(error.__notes__)
    assert warned == []


def test_is_too_old_to_retry_is_exact_at_36_hours() -> None:
    init_time = datetime(2026, 10, 1, tzinfo=UTC)
    assert not assets._is_too_old_to_retry(
        nwp_init_time=init_time, now=init_time + timedelta(hours=36)
    )
    assert assets._is_too_old_to_retry(
        nwp_init_time=init_time, now=init_time + timedelta(hours=36, seconds=1)
    )


@pytest.mark.parametrize("error", [RuntimeError("bug"), ValueError("bad dtype")])
def test_ecmwf_ens_fails_at_once_on_an_error_that_waiting_cannot_fix(
    env: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """Only the three "run not ready yet" failures retry; a structural failure or a bug of ours
    propagates unchanged. ``ValueError`` is here because ``NwpVariableWhollyMissing`` subclasses
    it, so a retry rule widened to ``ValueError`` would silently retry every dtype failure."""
    _write_h3_grid_weights(Settings().h3_grid_weights_path)
    monkeypatch.setattr(target=assets, name="open_ecmwf_ens_run", value=lambda **kwargs: object())

    def _raise(ds_lazy: object) -> None:
        raise error

    monkeypatch.setattr(target=assets, name="download_ecmwf_ens_data", value=_raise)

    with (
        build_asset_context(partition_key=_today_key()) as context,
        pytest.raises(type(error), match=str(error)),
    ):
        ecmwf_ens(context)


# --- definitions load ----------------------------------------------------------------------------


def test_definitions_resolve(env: Path) -> None:
    """The whole asset graph resolves into a repository, the three ingest assets are present, the
    ``ecmwf_ens`` dependency edge is wired, and each asset job's selection resolves to its asset.

    Resolution alone (constructing ``Definitions`` + ``get_repository_def()``) catches
    import-time errors and duplicate asset keys, but *not* a broken ``deps=[…]`` string (Dagster
    silently treats an unknown key as an external asset) or a job ``AssetSelection`` pointing at
    a missing asset (resolved lazily) — so those are asserted explicitly below.

    Uses ``get_repository_def()`` rather than the stricter ``Definitions.validate_loadable``: the
    latter also runs ``validate_partitions``, which rejects the CV pipeline's deliberate
    static-fold-upstream / dynamic-experiment-fold-downstream ``deps`` mapping that ``dg dev``
    and the CV asset tests run against happily.
    """
    from dagster import AssetKey

    from nged_substation_forecast.definitions import defs
    from nged_substation_forecast.defs.assets import ecmwf_ens_partitions
    from nged_substation_forecast.defs.live_forecast_assets import live_forecast_partitions

    repo = defs.get_repository_def()
    asset_graph = repo.asset_graph

    asset_keys = {key.to_user_string() for key in asset_graph.get_all_asset_keys()}
    assert {
        "power_time_series_and_metadata",
        "clean_nged_power_data",
        "h3_grid_weights",
        "ecmwf_ens",
    } <= asset_keys

    for name, layer in [
        ("live_forecasts", "production"),
        ("promotable_model_runs", "research"),
        ("promoted_model", "research"),
    ]:
        node = asset_graph.get(AssetKey(name))
        assert node.is_materializable
        assert node.tags["layer"] == layer

    assert {
        key.to_user_string() for key in asset_graph.get(AssetKey("live_forecasts")).parent_keys
    } == {"ecmwf_ens", "clean_nged_power_data"}

    # A broken deps=[...] string would drop this edge (the unknown key becomes an external asset).
    ecmwf_parents = {
        key.to_user_string() for key in asset_graph.get(AssetKey("ecmwf_ens")).parent_keys
    }
    assert "h3_grid_weights" in ecmwf_parents

    # Every production asset's check is registered.
    check_keys = {key.name for key in asset_graph.asset_check_keys}
    assert "power_data_is_fresh" in check_keys
    assert "nwp_has_no_unexpected_nulls" in check_keys
    assert "nwp_run_is_complete" in check_keys
    assert "live_forecasts_are_healthy" in check_keys
    assert "cleaned_power_keeps_up_with_raw" in check_keys

    # ...and the 6-hourly scheduled job actually runs the live check: an AssetSelection includes
    # its assets' checks, so this is what makes the check evaluate on every production tick.
    live_job_checks = {
        key.name
        for key in repo.get_job("live_forecasts_job").asset_layer.asset_graph.asset_check_keys
    }
    assert live_job_checks == {"live_forecasts_are_healthy"}

    # A job whose AssetSelection names a missing asset resolves to an empty/wrong key set.
    for job_name, expected_assets in [
        (
            "power_time_series_and_metadata_job",
            {"power_time_series_and_metadata", "clean_nged_power_data"},
        ),
        ("ecmwf_ens_job", {"ecmwf_ens"}),
        ("live_forecasts_job", {"live_forecasts"}),
    ]:
        selected = {
            key.to_user_string() for key in repo.get_job(job_name).asset_layer.executable_asset_keys
        }
        assert selected == expected_assets

    # Neither partitioned job passes `partitions_def` to `define_asset_job` — Dagster infers it from
    # the selected asset at resolution time. Assert the inferred definition equals the one the asset
    # declares, so a job silently resolving to `None`, or to a different cadence or start, fails
    # here rather than at the next schedule tick. (Equality, not identity: what matters is that the
    # job targets the same partitions, and Dagster is free to hand back an equal copy.)
    assert repo.get_job("ecmwf_ens_job").partitions_def == ecmwf_ens_partitions
    assert repo.get_job("live_forecasts_job").partitions_def == live_forecast_partitions

    # `live_forecasts_schedule` is built by `build_schedule_from_partitioned_job`, so its cron is
    # *derived* from that inferred partitions_def — the one thing dropping the explicit argument
    # could plausibly have broken. Pin the resolved schedule, not just the job.
    live_schedule = repo.get_schedule_def("live_forecasts_schedule")
    assert live_schedule.cron_schedule == live_forecast_partitions.cron_schedule
    assert live_schedule.execution_timezone == "UTC"


# --- summary classes (pure, no Dagster) ----------------------------------------------------------


def _file_listing(
    n: int, time_series_ids: list[int] | None = None
) -> pt.DataFrame[_ProcessedFileListing]:
    base = datetime(year=2026, month=3, day=26, hour=8, tzinfo=UTC)
    ids = time_series_ids if time_series_ids is not None else list(range(9, 9 + n))
    return (
        _ProcessedFileListing.DataFrame(
            {
                "path": [f"p{i}" for i in range(n)],
                "filesize_bytes": [1000 + i for i in range(n)],
                "last_modified": [base] * n,
                "time_series_id": ids,
                "start_time": [base] * n,
                "end_time": [base + timedelta(hours=i) for i in range(n)],
            }
        )
        .cast()
        .validate()
    )


def test_file_listing_summary_non_empty() -> None:
    """Non-empty frame: the ``@field_validator`` formats the datetimes, and ``n_time_series_ids``
    is deduped (two of the three files share ``time_series_id`` 11), so it differs from
    ``n_files``."""
    summary = _FileListingSummary.from_data_frame(
        "Files with new data", _file_listing(3, time_series_ids=[11, 9, 11])
    )
    assert summary.n_files == 3
    assert summary.start_time == "2026-03-26 08:00"
    assert summary.end_time == "2026-03-26 10:00"
    assert summary.n_time_series_ids == 2
    assert summary.min_file_size_bytes == 1000
    assert summary.max_file_size_bytes == 1002


def test_power_time_series_summary_non_empty() -> None:
    """Non-empty frame, with a duplicate ``time_series_id`` across rows (as
    ``test_file_listing_summary_non_empty`` has), so ``n_time_series_ids`` is pinned against a
    count that just copies ``n_rows``."""
    base = datetime(year=2026, month=3, day=26, hour=8, tzinfo=UTC)
    df = (
        PowerTimeSeries.DataFrame(
            {
                "time_series_id": [1, 2, 2],
                "time": [base, base + timedelta(minutes=30), base + timedelta(minutes=60)],
                "power": [2.5, 1.5, 1.6],
            }
        )
        .cast()
        .validate()
    )
    summary = _PowerTimeSeriesSummary.from_data_frame("Downloaded timeseries", df)
    assert summary.n_rows == 3
    assert summary.start_time == "2026-03-26 08:00"
    assert summary.n_time_series_ids == 2


@pytest.mark.parametrize(
    ("summary_cls", "empty_df"),
    [
        (_FileListingSummary, _ProcessedFileListing.DataFrame(schema=_ProcessedFileListing.dtypes)),
        (_PowerTimeSeriesSummary, PowerTimeSeries.DataFrame(schema=PowerTimeSeries.dtypes)),
    ],
)
def test_summary_empty_frame_uses_defaults(
    summary_cls: type[_BaseSummary], empty_df: pt.DataFrame
) -> None:
    """Empty frame → the ``"N/A"`` start/end defaults survive (the validator passes them through
    untouched) and ``n_time_series_ids`` stays at its ``0`` default."""
    summary = summary_cls.from_data_frame("stage", empty_df)
    assert summary.start_time == "N/A"
    assert summary.end_time == "N/A"
    assert summary.n_time_series_ids == 0


def test_make_table_returns_one_record_per_stage() -> None:
    """``make_table`` wraps each stage's summary as a Dagster table row under the given key."""
    table_metadata = _FileListingSummary.make_table(
        "nged_s3_paths", {"stage_a": _file_listing(2), "stage_b": _file_listing(1)}
    )
    assert set(table_metadata) == {"nged_s3_paths"}
    assert len(table_metadata["nged_s3_paths"].value.records) == 2
