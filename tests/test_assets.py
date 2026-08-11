"""Materialisation tests for the three ingest Dagster assets, plus a definitions-load smoke test.

Fires up Dagster for each ingest asset — ``power_time_series_and_metadata``, ``h3_grid_weights``,
``ecmwf_ens`` — against temp Delta/parquet tables, and asserts the whole asset graph (assets +
jobs + schedules) resolves. The three leaf data pipelines (NGED JSON parsing, H3 weighting, ECMWF
download/convert) are unit-tested in their own packages; here we exercise only the asset *bodies* —
the wiring, branching, and metadata each asset owns — stubbing the S3/network boundary and the
~30-second GB-boundary buffer so the tests stay fast and offline.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import patito as pt
import polars as pl
import pytest
import shapely
from contracts.geo_schemas import H3GridWeights
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from contracts.settings import Settings
from contracts.weather_schemas import Nwp
from dagster import (
    AssetCheckEvaluation,
    AssetCheckSeverity,
    DagsterExecutionInterruptedError,
    DagsterInstance,
    ExecuteInProcessResult,
    build_asset_context,
    materialize,
)
from dynamical_data.ecmwf_ens.download import NwpRunNotYetAvailable
from nged_data.storage import NoNewData, _ProcessedFileListing

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

    def bytes(self) -> _JsonBytes:
        return self._data


class _FakeS3Store:
    """Minimal ``obstore`` store stand-in serving a fixed set of NGED JSON files.

    ``list_timeseries_json_files`` and ``download_and_parse_files`` only call ``.list()`` and
    ``.get()``, so duck-typing those two methods lets the real asset body run offline.
    """

    def __init__(self, files: dict[str, _JsonBytes]) -> None:
        self._files = files

    def list(self, prefix: str) -> _StoreListing:
        return [[{"path": path, "size": len(data)} for path, data in self._files.items()]]

    def get(self, path: str) -> _FakeGetResult:
        return _FakeGetResult(self._files[path])


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
        assets.Settings, "get_nged_s3_store", lambda self: _FakeS3Store(_NGED_FILES)
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


def test_power_time_series_and_metadata_writes_power_when_the_roster_upsert_fails(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """The headline property of #508: the roster is derived, re-delivered data, so losing one
    refresh of it must not cost the power stream an hour of telemetry.

    Also asserts the degradation is *reported*. Not failing the step means
    ``sentry_capture_failure`` no longer fires, and log-to-event capture is off, so without the
    explicit send this would be a silent hole rather than a degraded run.
    """
    monkeypatch.setattr(
        assets.Settings, "get_nged_s3_store", lambda self: _FakeS3Store(_NGED_FILES)
    )

    def boom(*_: object, **__: object) -> None:
        raise RuntimeError("roster upsert exploded")

    monkeypatch.setattr(assets, "upsert_metadata", boom)
    reported: list[tuple[str, object]] = []
    monkeypatch.setattr(
        assets,
        "report_asset_degradation",
        lambda asset_name, exc: reported.append((asset_name, exc)),
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
    assert isinstance(reported[0][1], RuntimeError)


def test_power_time_series_and_metadata_degrades_on_a_rust_panic_in_the_upsert(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """The reason the guard catches ``BaseException`` rather than ``Exception``.

    A panic in any of the pyo3 extensions ``upsert_metadata`` reads through — Polars or obstore (via
    ``object_exists``) — is not an ``Exception``, so a narrower guard would let it through and fail
    the hourly run, which is the exact failure this guard exists to prevent. The cancellation test
    below cannot catch that regression, because ``DagsterExecutionInterruptedError`` is not an
    ``Exception`` either and so propagates out of a narrow guard on its own.
    """
    monkeypatch.setattr(
        assets.Settings, "get_nged_s3_store", lambda self: _FakeS3Store(_NGED_FILES)
    )

    class _Panic(BaseException):
        """Stands in for a pyo3 `PanicException`, which derives from `BaseException`."""

    def _panic(*_: object, **__: object) -> None:
        raise _Panic("simulated rust panic inside the upsert")

    monkeypatch.setattr(assets, "upsert_metadata", _panic)
    reported: list[tuple[str, BaseException]] = []
    monkeypatch.setattr(
        assets,
        "report_asset_degradation",
        lambda asset_name, exc: reported.append((asset_name, exc)),
    )

    result = materialize([power_time_series_and_metadata], instance=dagster_instance)
    assert result.success
    assert set(pl.read_delta(str(env / "NGED" / "power_time_series.delta"))["time_series_id"]) == {
        10,
        11,
    }
    assert [name for name, _ in reported] == ["power_time_series_and_metadata"]
    assert isinstance(reported[0][1], _Panic)


def test_power_time_series_and_metadata_re_raises_a_cancelled_run(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """The one thing the guard must *not* swallow. Cancellation lands in the same ``BaseException``
    net as a panic, so the handler re-raises it explicitly: a run the operator cancelled has to
    stop, not finish green having quietly skipped the roster."""
    monkeypatch.setattr(
        assets.Settings, "get_nged_s3_store", lambda self: _FakeS3Store(_NGED_FILES)
    )

    def _cancel(*_: object, **__: object) -> None:
        raise DagsterExecutionInterruptedError

    monkeypatch.setattr(assets, "upsert_metadata", _cancel)
    reported: list[str] = []
    monkeypatch.setattr(
        assets, "report_asset_degradation", lambda asset_name, exc: reported.append(asset_name)
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
    monkeypatch.setattr(assets.Settings, "get_nged_s3_store", lambda self: _FakeS3Store(files))

    result = materialize([power_time_series_and_metadata], instance=dagster_instance)
    assert result.success

    power = pl.read_delta(str(env / "NGED" / "power_time_series.delta"))
    assert power.height == 1
    assert power["time"][0] == datetime(2026, 3, 5, 12, 30, tzinfo=UTC)

    materialisations = result.asset_materializations_for_node("power_time_series_and_metadata")
    metadata = {k: v for mat in materialisations for k, v in mat.metadata.items()}
    assert metadata["n_implausible_power_rows_dropped"].value == 1


def test_power_time_series_and_metadata_handles_no_new_data(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """``NoNewData`` from ``download_and_parse_files`` → the asset returns early, writes nothing."""
    monkeypatch.setattr(
        assets.Settings, "get_nged_s3_store", lambda self: _FakeS3Store(_NGED_FILES)
    )

    def _raise_no_new_data(store: object, paths_df: object) -> None:
        raise NoNewData

    monkeypatch.setattr(assets, "download_and_parse_files", _raise_no_new_data)

    result = materialize([power_time_series_and_metadata], instance=dagster_instance)
    assert result.success
    assert not (env / "NGED" / "metadata.parquet").exists()
    assert not (env / "NGED" / "power_time_series.delta").exists()


# --- h3_grid_weights -----------------------------------------------------------------------------


def test_h3_grid_weights_materialises_and_writes_parquet(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """Materialise ``h3_grid_weights`` against a small stand-in boundary, and assert a valid
    parquet lands on disk.

    The real GB boundary buffers for ~30 s, and is exercised in ``packages/geo`` instead.
    """
    # A 1×1-degree box over central GB — enough to yield several H3 cells, milliseconds to compute.
    monkeypatch.setattr(assets, "load_gb_boundary", lambda: shapely.box(-2.0, 52.0, -1.0, 53.0))

    result = materialize([h3_grid_weights], instance=dagster_instance)
    assert result.success

    weights = pl.read_parquet(env / "h3_grid_weights.parquet")
    H3GridWeights.validate(weights)
    assert weights.height > 0


# --- ecmwf_ens -----------------------------------------------------------------------------------


def _check_evaluations(result: ExecuteInProcessResult) -> dict[str, AssetCheckEvaluation]:
    """The run's asset-check evaluations, keyed by check name.

    ``ecmwf_ens`` emits two independent checks, so tests look theirs up by name rather than
    relying on the order Dagster happens to report them in.
    """
    return {
        evaluation.check_name: evaluation for evaluation in result.get_asset_check_evaluations()
    }


def test_ecmwf_ens_materialises_and_appends_nwp(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """Happy path with the download/convert pipeline stubbed: the partition key parses into
    ``nwp_init_time`` (passed to ``open_ecmwf_ens_run``) and the converted frame is written to the
    NWP Delta table via ``write_nwp``."""
    _write_h3_grid_weights(Settings().h3_grid_weights_path)
    # After 2024-11-12, when categorical_precipitation_type_surface became a non-null Nwp variable.
    init_time = datetime(2024, 12, 1, tzinfo=UTC)
    captured: dict[str, datetime] = {}

    def _open(*, nwp_init_time: datetime, h3_grid: object) -> object:
        captured["nwp_init_time"] = nwp_init_time
        return object()

    monkeypatch.setattr(assets, "open_ecmwf_ens_run", _open)
    monkeypatch.setattr(assets, "download_ecmwf_ens_data", lambda ds: object())
    monkeypatch.setattr(
        assets,
        "convert_nwp_xarray_dataset_to_polars_dataframe",
        lambda ds, h3_grid: _make_nwp(init_time),
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


def test_ecmwf_ens_warns_on_scattered_nulls_but_still_materialises(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """Scattered per-pixel nulls in a de-accumulated variable (the known upstream ECMWF ENS
    corruption) are tolerated: the run still materialises, and the data-quality check WARNs."""
    _write_h3_grid_weights(Settings().h3_grid_weights_path)
    init_time = datetime(2024, 12, 1, tzinfo=UTC)

    # One (member, valid_time) slice across three h3 cells, one cell's precipitation nulled.
    scattered = _make_nwp(init_time, n=3).with_columns(
        init_time=pl.lit(init_time),
        valid_time=pl.lit(init_time + timedelta(hours=3)),
        ensemble_member=pl.lit(0, dtype=pl.UInt8),
        precipitation_surface=pl.Series([0.001, None, 0.001], dtype=pl.Float32),
    )
    # `object` cannot be inlined in place of this stub: the real function is called with
    # keyword arguments, which `object()` rejects.
    monkeypatch.setattr(
        assets,
        "open_ecmwf_ens_run",
        lambda *, nwp_init_time, h3_grid: object(),  # noqa: PLW0108
    )
    monkeypatch.setattr(assets, "download_ecmwf_ens_data", lambda ds: object())
    monkeypatch.setattr(
        assets, "convert_nwp_xarray_dataset_to_polars_dataframe", lambda ds, h3_grid: scattered
    )

    result = materialize([ecmwf_ens], partition_key="2024-12-01", instance=dagster_instance)
    assert result.success  # tolerated — the run is NOT failed
    assert pl.read_delta(Settings().nwp_data_path).height == 3  # data was persisted
    evaluation = _check_evaluations(result)["nwp_has_no_unexpected_nulls"]
    assert not evaluation.passed  # WARN: the scatter is surfaced
    assert evaluation.metadata["n_null_cells"].value == 1
    assert evaluation.metadata["n_whole_null_slices"].value == 0
    # Both halves of the split are emitted, not just the whole-null one: the operations runbook
    # names `n_scattered_slices` as a number to read off this check.
    assert evaluation.metadata["n_scattered_slices"].value == 1


def test_ecmwf_ens_reports_whole_null_slices_in_its_quality_check(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """A wholly-null (member, valid_time) slice — the 2026-08-09 class — reaches the operator as a
    WARN that counts it, rather than passing silently.

    Scoped to the *reporting* half deliberately: the converter is stubbed here, so `Nwp.validate`
    never sees this frame, and that it no longer rejects such a slice is pinned by
    ``test_whole_slice_deaccumulated_null_beyond_lead0_is_tolerated`` in the contracts package.
    What fails on ``main`` is the count: ``assess_nwp_quality`` filtered wholly-null slices out of
    its report entirely, so the check passed and the missing field was surfaced nowhere.
    """
    _write_h3_grid_weights(Settings().h3_grid_weights_path)
    init_time = datetime(2024, 12, 1, tzinfo=UTC)

    # `_make_nwp` gives each row its own (member, valid_time), so nulling one row's precipitation
    # empties one whole slice of three while the other two stay intact.
    one_slice_missing = _make_nwp(init_time, n=3).with_columns(
        precipitation_surface=pl.Series([None, 0.001, 0.001], dtype=pl.Float32)
    )
    # `object` cannot be inlined in place of this stub: the real function is called with
    # keyword arguments, which `object()` rejects.
    monkeypatch.setattr(
        assets,
        "open_ecmwf_ens_run",
        lambda *, nwp_init_time, h3_grid: object(),  # noqa: PLW0108
    )
    monkeypatch.setattr(assets, "download_ecmwf_ens_data", lambda ds: object())
    monkeypatch.setattr(
        assets,
        "convert_nwp_xarray_dataset_to_polars_dataframe",
        lambda ds, h3_grid: one_slice_missing,
    )

    result = materialize([ecmwf_ens], partition_key="2024-12-01", instance=dagster_instance)
    assert result.success  # tolerated — the run is NOT failed
    assert pl.read_delta(Settings().nwp_data_path).height == 3  # data was persisted
    evaluation = _check_evaluations(result)["nwp_has_no_unexpected_nulls"]
    assert not evaluation.passed  # WARN: the missing slice is surfaced
    assert evaluation.metadata["n_whole_null_slices"].value == 1
    assert evaluation.metadata["n_null_cells"].value == 1
    # The mirror of the scattered test above: the same slice must be counted once, on one side of
    # the split, so the two metadata fields cannot both claim it.
    assert evaluation.metadata["n_scattered_slices"].value == 0


def test_ecmwf_ens_retries_when_a_variable_is_wholly_missing(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``NwpVariableWhollyMissing`` → ``RetryRequested``, not a failed partition.

    An all-null weather column is one way a half-published upstream run reads, and Dynamical.org
    republishes a defective one — the 2026-08-09 repair landed 3h25m later, inside this budget.

    The stub calls the *real* ``Nwp.validate``, so this pins the whole chain the widened ``try``
    exists for: an empty column raises from validation, which the converter calls, which sits past
    where ``main``'s ``try`` block ended. It fails on ``main``, where the exception escaped as a
    hard failure.
    """
    from dagster import RetryRequested

    _write_h3_grid_weights(Settings().h3_grid_weights_path)
    init_time = datetime(2024, 12, 1, tzinfo=UTC)
    # A run whose radiation column carries no weather at all, exactly as the converter would hand
    # it over: `_make_nwp` gives each row its own (member, valid_time), so nulling every row empties
    # the column across every slice beyond lead-0.
    wholly_missing = _make_nwp(init_time, n=3).with_columns(
        downward_short_wave_radiation_flux_surface=pl.Series([None] * 3, dtype=pl.Float32)
    )

    def _convert_via_real_validation(ds: object, h3_grid: object) -> pt.DataFrame[Nwp]:
        return Nwp.validate(wholly_missing)

    # `object` cannot be inlined in place of this stub: the real function is called with
    # keyword arguments, which `object()` rejects.
    monkeypatch.setattr(
        assets,
        "open_ecmwf_ens_run",
        lambda *, nwp_init_time, h3_grid: object(),  # noqa: PLW0108
    )
    monkeypatch.setattr(assets, "download_ecmwf_ens_data", lambda ds: object())
    monkeypatch.setattr(
        assets, "convert_nwp_xarray_dataset_to_polars_dataframe", _convert_via_real_validation
    )

    with (
        build_asset_context(partition_key="2024-12-01") as context,
        pytest.raises(RetryRequested) as exc_info,
    ):
        ecmwf_ens(context)

    assert exc_info.value.max_retries == _ECMWF_ENS_MAX_RETRIES
    assert exc_info.value.seconds_to_wait == _ECMWF_ENS_RETRY_DELAY_SECONDS
    # Validation runs before the Delta append, so a retry (or a later manual re-run) has no partial
    # partition to double-count against.
    assert not Path(Settings().nwp_data_path).exists()


def test_ecmwf_ens_warns_on_incomplete_run_but_still_materialises(
    env: Path, monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """A short run is landed anyway and surfaced as a WARN — an incomplete upstream run is absent
    input, so we keep the rows that arrived rather than discarding the whole partition.

    ``_make_nwp`` builds 4 rows carrying 4 distinct members, valid_times and cells (a diagonal, not
    a cross-product), which is nothing like a complete 51 x 85 x 1 ECMWF ENS run.
    """
    _write_h3_grid_weights(Settings().h3_grid_weights_path)
    init_time = datetime(2024, 12, 1, tzinfo=UTC)
    # `object` cannot be inlined in place of this stub: the real function is called with
    # keyword arguments, which `object()` rejects.
    monkeypatch.setattr(
        assets,
        "open_ecmwf_ens_run",
        lambda *, nwp_init_time, h3_grid: object(),  # noqa: PLW0108
    )
    monkeypatch.setattr(assets, "download_ecmwf_ens_data", lambda ds: object())
    monkeypatch.setattr(
        assets,
        "convert_nwp_xarray_dataset_to_polars_dataframe",
        lambda ds, h3_grid: _make_nwp(init_time),
    )

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


def test_ecmwf_ens_retries_when_run_not_yet_available(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``NwpRunNotYetAvailable`` → ``RetryRequested`` with the asset's configured retry budget,
    so a not-yet-published run waits rather than failing outright."""
    from dagster import RetryRequested

    _write_h3_grid_weights(Settings().h3_grid_weights_path)

    def _raise_not_available(*, nwp_init_time: datetime, h3_grid: object) -> None:
        raise NwpRunNotYetAvailable

    monkeypatch.setattr(assets, "open_ecmwf_ens_run", _raise_not_available)

    # `build_asset_context()` defaults to its own `DagsterInstance.ephemeral()`
    # (<https://openclimatefix.github.io/nged-substation-forecast/architecture/testing/>) and is
    # used as a context manager here for the same reason
    # `dagster_instance` is a fixture: entering it makes disposal happen deterministically at
    # `__exit__`, rather than depending on `__del__` running via garbage collection, which the
    # traceback captured by `pytest.raises` delays past this test — see the fixture's docstring.
    with (
        build_asset_context(partition_key="2024-05-01") as context,
        pytest.raises(RetryRequested) as exc_info,
    ):
        ecmwf_ens(context)

    assert exc_info.value.max_retries == _ECMWF_ENS_MAX_RETRIES
    assert exc_info.value.seconds_to_wait == _ECMWF_ENS_RETRY_DELAY_SECONDS


# --- definitions load ----------------------------------------------------------------------------


def test_definitions_resolve(env: Path) -> None:
    """The whole asset graph resolves into a repository, the three ingest assets are present, the
    ``ecmwf_ens`` dependency edge is wired, and each asset job's selection resolves to its asset.

    Resolution alone (constructing ``Definitions`` + ``get_repository_def()``) catches import-time
    errors and duplicate asset keys, but *not* a broken ``deps=[…]`` string (Dagster silently treats
    an unknown key as an external asset) or a job ``AssetSelection`` pointing at a missing asset
    (resolved lazily) — so those are asserted explicitly below.

    Uses ``get_repository_def()`` rather than the stricter ``Definitions.validate_loadable``: the
    latter also runs ``validate_partitions``, which rejects the CV pipeline's deliberate
    static-fold-upstream / dynamic-experiment-fold-downstream ``deps`` mapping that ``dg dev`` and
    the CV asset tests run against happily.
    """
    from dagster import AssetKey

    from nged_substation_forecast.definitions import defs
    from nged_substation_forecast.defs.assets import ecmwf_ens_partitions
    from nged_substation_forecast.defs.production_assets import live_forecast_partitions

    repo = defs.get_repository_def()
    asset_graph = repo.asset_graph

    asset_keys = {key.to_user_string() for key in asset_graph.get_all_asset_keys()}
    assert {"power_time_series_and_metadata", "h3_grid_weights", "ecmwf_ens"} <= asset_keys

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

    # ...and the 6-hourly scheduled job actually runs the live check: an AssetSelection includes
    # its assets' checks, so this is what makes the check evaluate on every production tick.
    live_job_checks = {
        key.name
        for key in repo.get_job("live_forecasts_job").asset_layer.asset_graph.asset_check_keys
    }
    assert live_job_checks == {"live_forecasts_are_healthy"}

    # A job whose AssetSelection names a missing asset resolves to an empty/wrong key set.
    for job_name, expected_asset in [
        ("power_time_series_and_metadata_job", "power_time_series_and_metadata"),
        ("ecmwf_ens_job", "ecmwf_ens"),
        ("live_forecasts_job", "live_forecasts"),
    ]:
        selected = {
            key.to_user_string() for key in repo.get_job(job_name).asset_layer.executable_asset_keys
        }
        assert selected == {expected_asset}

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
    live_schedule = repo.get_schedule_def("live_forecasts_job_schedule")
    assert live_schedule.cron_schedule == live_forecast_partitions.cron_schedule
    assert live_schedule.execution_timezone == "UTC"


# --- summary classes (pure, no Dagster) ----------------------------------------------------------


def _file_listing(
    n: int, time_series_ids: list[int] | None = None
) -> pt.DataFrame[_ProcessedFileListing]:
    base = datetime(2026, 3, 26, 8, tzinfo=UTC)
    ids = time_series_ids if time_series_ids is not None else list(range(9, 9 + n))
    return (
        _ProcessedFileListing.DataFrame(
            {
                "path": [f"p{i}" for i in range(n)],
                "filesize_bytes": [1000 + i for i in range(n)],
                "time_series_id": ids,
                "start_time": [base] * n,
                "end_time": [base + timedelta(hours=i) for i in range(n)],
            }
        )
        .cast()
        .validate()
    )


def test_file_listing_summary_non_empty() -> None:
    """Non-empty frame: the ``@field_validator``s format the datetime and dedup the IDs (two of the
    three files share ``time_series_id`` 11), and ``n_time_series_ids`` parses the resulting string
    back to a count distinct from ``n_files``."""
    summary = _FileListingSummary.from_data_frame(
        "Files with new data", _file_listing(3, time_series_ids=[11, 9, 11])
    )
    assert summary.n_files == 3
    assert summary.start_time == "2026-03-26 08:00"
    assert summary.end_time == "2026-03-26 10:00"
    assert summary.time_series_ids == "[9, 11]"  # deduped and sorted
    assert summary.n_time_series_ids == 2
    assert summary.min_file_size_bytes == 1000
    assert summary.max_file_size_bytes == 1002


def test_power_time_series_summary_non_empty() -> None:
    base = datetime(2026, 3, 26, 8, tzinfo=UTC)
    df = (
        PowerTimeSeries.DataFrame(
            {
                "time_series_id": [1, 2],
                "time": [base, base + timedelta(minutes=30)],
                "power": [2.5, 1.5],
            }
        )
        .cast()
        .validate()
    )
    summary = _PowerTimeSeriesSummary.from_data_frame("Downloaded timeseries", df)
    assert summary.n_rows == 2
    assert summary.start_time == "2026-03-26 08:00"
    assert summary.time_series_ids == "[1, 2]"
    assert summary.n_time_series_ids == 2


@pytest.mark.parametrize(
    ("summary_cls", "empty_df"),
    [
        (_FileListingSummary, _ProcessedFileListing.DataFrame(schema=_ProcessedFileListing.dtypes)),
        (_PowerTimeSeriesSummary, PowerTimeSeries.DataFrame(schema=PowerTimeSeries.dtypes)),
    ],
)
def test_summary_empty_frame_uses_na_defaults(
    summary_cls: type[_BaseSummary], empty_df: pt.DataFrame
) -> None:
    """Empty frame → the ``"N/A"`` defaults survive (the validators pass them through untouched) and
    ``n_time_series_ids`` short-circuits to 0 without calling ``ast.literal_eval``."""
    summary = summary_cls.from_data_frame("stage", empty_df)
    assert summary.start_time == "N/A"
    assert summary.end_time == "N/A"
    assert summary.time_series_ids == "N/A"
    assert summary.n_time_series_ids == 0


def test_make_table_returns_one_record_per_stage() -> None:
    """``make_table`` wraps each stage's summary as a Dagster table row under the given key."""
    table_metadata = _FileListingSummary.make_table(
        "nged_s3_paths", {"stage_a": _file_listing(2), "stage_b": _file_listing(1)}
    )
    assert set(table_metadata) == {"nged_s3_paths"}
    assert len(table_metadata["nged_s3_paths"].value.records) == 2
