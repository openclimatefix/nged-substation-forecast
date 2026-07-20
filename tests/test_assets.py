"""Materialisation tests for the three ingest Dagster assets, plus a definitions-load smoke test.

Fires up Dagster for each ingest asset — ``power_time_series_and_metadata``, ``h3_grid_weights``,
``ecmwf_ens`` — against temp Delta/parquet tables, and asserts the whole asset graph (assets +
jobs + schedules) resolves. The three leaf data pipelines (NGED JSON parsing, H3 weighting, ECMWF
download/convert) are unit-tested in their own packages; here we exercise only the asset *bodies* —
the wiring, branching, and metadata each asset owns — stubbing the S3/network boundary and the
~30-second GB-boundary buffer so the tests stay fast and offline.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import patito as pt
import polars as pl
import pytest
import shapely
from contracts.geo_schemas import H3GridWeights
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from contracts.weather_schemas import Nwp
from dagster import DagsterInstance, build_asset_context, materialize
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
    """A minimal valid ``H3GridWeights`` parquet — the ``ecmwf_ens`` asset reads it before download."""
    H3GridWeights.DataFrame(
        {"h3_index": [100], "nwp_lat": [52.5], "nwp_lon": [-1.0], "proportion": [1.0]}
    ).cast().validate().write_parquet(path)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every managed data-path root at a temp dir, fully isolating the assets from the
    developer's real configuration (dummy ``NGED_S3_BUCKET_*`` creds come from the autouse
    ``_dummy_nged_s3_creds`` fixture in conftest.py)."""
    monkeypatch.setenv("DATA_PATH_INTERNAL", str(tmp_path))
    monkeypatch.setenv("DATA_PATH_DELIVERY", str(tmp_path))
    monkeypatch.setenv("LOCAL_ARTIFACTS_PATH", str(tmp_path))
    return tmp_path


# --- power_time_series_and_metadata --------------------------------------------------------------


def test_power_time_series_and_metadata_ingests_and_writes(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Happy path: a fake S3 store serving two real NGED JSON files → metadata parquet + power
    Delta table both written, and the asset materialises successfully."""
    monkeypatch.setattr(
        assets.Settings, "get_nged_s3_store", lambda self: _FakeS3Store(_NGED_FILES)
    )

    result = materialize([power_time_series_and_metadata], instance=DagsterInstance.ephemeral())
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


def test_power_time_series_and_metadata_handles_no_new_data(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``NoNewData`` from ``download_and_parse_files`` → the asset returns early, writing nothing."""
    monkeypatch.setattr(
        assets.Settings, "get_nged_s3_store", lambda self: _FakeS3Store(_NGED_FILES)
    )

    def _raise_no_new_data(store: object, paths_df: object) -> None:
        raise NoNewData

    monkeypatch.setattr(assets, "download_and_parse_files", _raise_no_new_data)

    result = materialize([power_time_series_and_metadata], instance=DagsterInstance.ephemeral())
    assert result.success
    assert not (env / "NGED" / "metadata.parquet").exists()
    assert not (env / "NGED" / "power_time_series.delta").exists()


# --- h3_grid_weights -----------------------------------------------------------------------------


def test_h3_grid_weights_materialises_and_writes_parquet(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Materialise ``h3_grid_weights`` against a small stand-in boundary (the real GB boundary
    buffers for ~30 s and is exercised in ``packages/geo``); assert a valid parquet lands on disk."""
    # A 1×1-degree box over central GB — enough to yield several H3 cells, milliseconds to compute.
    monkeypatch.setattr(assets, "load_gb_boundary", lambda: shapely.box(-2.0, 52.0, -1.0, 53.0))

    result = materialize([h3_grid_weights], instance=DagsterInstance.ephemeral())
    assert result.success

    weights = pl.read_parquet(env / "h3_grid_weights.parquet")
    H3GridWeights.validate(weights)
    assert weights.height > 0


# --- ecmwf_ens -----------------------------------------------------------------------------------


def test_ecmwf_ens_materialises_and_appends_nwp(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path with the download/convert pipeline stubbed: the partition key parses into
    ``nwp_init_time`` (passed to ``open_ecmwf_ens_run``) and the converted frame is written to the
    NWP Delta table via ``write_nwp``."""
    from contracts.settings import Settings

    _write_h3_grid_weights(Settings().h3_grid_weights_path)
    # After 2024-11-12, when categorical_precipitation_type_surface became a non-null Nwp variable.
    init_time = datetime(2024, 12, 1, tzinfo=timezone.utc)
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

    result = materialize(
        [ecmwf_ens], partition_key="2024-12-01", instance=DagsterInstance.ephemeral()
    )
    assert result.success
    # The partition key is parsed into nwp_init_time and handed to open_ecmwf_ens_run...
    assert captured["nwp_init_time"] == init_time
    # ...and the converted frame is actually persisted via write_nwp (all 4 rows round-trip).
    written = pl.read_delta(Settings().nwp_data_path)
    assert written.height == 4


def test_ecmwf_ens_retries_when_run_not_yet_available(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``NwpRunNotYetAvailable`` → ``RetryRequested`` with the asset's configured retry budget,
    so a not-yet-published run waits rather than failing outright."""
    from contracts.settings import Settings
    from dagster import RetryRequested

    _write_h3_grid_weights(Settings().h3_grid_weights_path)

    def _raise_not_available(*, nwp_init_time: datetime, h3_grid: object) -> None:
        raise NwpRunNotYetAvailable

    monkeypatch.setattr(assets, "open_ecmwf_ens_run", _raise_not_available)

    with pytest.raises(RetryRequested) as exc_info:
        ecmwf_ens(build_asset_context(partition_key="2024-05-01"))

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

    repo = defs.get_repository_def()
    asset_graph = repo.asset_graph

    asset_keys = {key.to_user_string() for key in asset_graph.get_all_asset_keys()}
    assert {"power_time_series_and_metadata", "h3_grid_weights", "ecmwf_ens"} <= asset_keys

    # A broken deps=[...] string would drop this edge (the unknown key becomes an external asset).
    ecmwf_parents = {
        key.to_user_string() for key in asset_graph.get(AssetKey("ecmwf_ens")).parent_keys
    }
    assert "h3_grid_weights" in ecmwf_parents

    # A job whose AssetSelection names a missing asset resolves to an empty/wrong key set.
    for job_name, expected_asset in [
        ("power_time_series_and_metadata_job", "power_time_series_and_metadata"),
        ("ecmwf_ens_job", "ecmwf_ens"),
    ]:
        selected = {
            key.to_user_string() for key in repo.get_job(job_name).asset_layer.executable_asset_keys
        }
        assert selected == {expected_asset}


# --- summary classes (pure, no Dagster) ----------------------------------------------------------


def _file_listing(
    n: int, time_series_ids: list[int] | None = None
) -> pt.DataFrame[_ProcessedFileListing]:
    base = datetime(2026, 3, 26, 8, tzinfo=timezone.utc)
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
    base = datetime(2026, 3, 26, 8, tzinfo=timezone.utc)
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
    "summary_cls, empty_df",
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
