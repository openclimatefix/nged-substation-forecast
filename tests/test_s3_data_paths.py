"""S3 integration tests for the managed data tables (issue #121).

Exercises the data-table IO layer against a real S3 endpoint — an in-process ``moto`` server, so
no Docker or network — driven entirely through ``Settings`` pointed at an ``s3://`` data-path root.
Proves the two things the S3 migration must guarantee and that the local test suite cannot:

1. Delta and parquet tables round-trip over ``s3://`` through the production write/read helpers
   (``write_power_forecasts`` / ``write_nwp`` / ``Nwp.scan_delta`` / ``upsert_metadata``) using
   only ``Settings.storage_options``.
2. **Partition pruning survives over S3** — the ``.explain()`` plan for a filter on the
   ``experiment_name`` partition column lists only the matching partition's path. The whole
   memory-bounding design (single fold in RAM) depends on this holding on the object store too.

The coverage here goes through the functional ``write_deltalake`` + polars
``scan_delta``/``read_delta`` + obstore paths that production uses for the hot IO. It deliberately
does **not** drive the ``deltalake.DeltaTable`` *class* client (``delta_table_exists`` /
``DeltaTable(...).partitions()``): that client's connection handling hangs against moto's in-process
dev server (an emulator limitation — it is the standard API and works on real S3/MinIO), so
exercising it here would only add a 3-minute stall, not real signal.

Marked ``integration``; skipped automatically if ``moto`` is not installed.
"""

import socket
import urllib.request
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

import patito as pt
import polars as pl
import pytest
from contracts._uri import object_exists
from contracts.typing_utils import typeddict_to_dict
from contracts.power_schemas import PowerForecast, TimeSeriesMetadata
from contracts.settings import Settings
from contracts.weather_schemas import Nwp
from delta_store.nwp import write_nwp
from delta_store.power_forecasts import write_power_forecasts
from nged_data.storage import upsert_metadata

from nged_substation_forecast.defs.cv_assets import PopulationFilter

pytestmark = pytest.mark.integration

ThreadedMotoServer = pytest.importorskip("moto.server").ThreadedMotoServer

_T0 = datetime(2025, 7, 1, tzinfo=timezone.utc)
_BUCKET = "nged-test-bucket"

_CONTINUOUS_BASE_VALUES = {
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


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def s3_endpoint() -> Iterator[str]:
    """Start an in-process moto S3 server and create the test bucket (no boto3, no Docker)."""
    port = _free_port()
    server = ThreadedMotoServer(port=port)
    server.start()
    endpoint = f"http://127.0.0.1:{port}"
    # Create the bucket with a bare HTTP PUT — object_store never creates buckets itself.
    urllib.request.urlopen(urllib.request.Request(f"{endpoint}/{_BUCKET}", method="PUT")).close()
    try:
        yield endpoint
    finally:
        server.stop()


def _s3_settings(endpoint: str, prefix: str) -> Settings:
    """A ``Settings`` whose data tables live under ``s3://{_BUCKET}/{prefix}`` on the moto server.

    Only ``data_path_internal``/``data_path_delivery`` and the ``data_store_*`` credentials are
    set (both roots point at the same prefix, as in local dev); every ``*_data_path`` derives from
    one of the two roots through the normal validator, so the test drives the real derivation +
    ``storage_options`` chain rather than hand-built URIs.
    """
    return Settings(
        data_path_internal=f"s3://{_BUCKET}/{prefix}",
        data_path_delivery=f"s3://{_BUCKET}/{prefix}",
        data_store_endpoint_url=endpoint,
        data_store_access_key_id="test",
        data_store_secret_access_key="test",
        data_store_region="us-east-1",
        nged_s3_bucket_url="https://example.com",
        nged_s3_bucket_access_key="key",
        nged_s3_bucket_secret="secret",
    )


def _make_forecasts(experiment_name: str, power_fcst: list[float]) -> pt.DataFrame[PowerForecast]:
    n = len(power_fcst)
    return (
        PowerForecast.DataFrame(
            {
                "valid_time": [_T0 + timedelta(hours=i) for i in range(n)],
                "time_series_id": list(range(1, n + 1)),
                "ensemble_member": [i % 3 for i in range(n)],
                "ml_flow_experiment_id": [None] * n,
                "nwp_init_time": [_T0] * n,
                "power_fcst_model_name": ["xgboost"] * n,
                "experiment_name": [experiment_name] * n,
                "power_fcst_model_version": [1] * n,
                "power_fcst_init_time": [_T0] * n,
                "power_fcst": power_fcst,
                "fold_id": ["s3_test"] * n,
            }
        )
        .cast()
        .validate()
    )


def _make_nwp(n: int = 6) -> pt.DataFrame[Nwp]:
    return (
        Nwp.DataFrame(
            {
                "nwp_model_id": ["ECMWF_ENS_0_25_degree"] * n,
                "init_time": [_T0] * n,
                "valid_time": [_T0 + timedelta(hours=(i % 3) + 1) for i in range(n)],
                "ensemble_member": list(range(n - 1, -1, -1)),
                "h3_index": [100 + i for i in range(n)],
                "categorical_precipitation_type_surface": [1] * n,
                **{
                    var: [base * (1 + 0.003 * i) for i in range(n)]
                    for var, base in _CONTINUOUS_BASE_VALUES.items()
                },
            }
        )
        .cast()
        .validate()
    )


def _make_metadata() -> pt.DataFrame[TimeSeriesMetadata]:
    rows = [
        {
            "time_series_id": ts_id,
            "time_series_name": f"Test Substation {ts_id}",
            "time_series_type": "Disaggregated Demand",
            "units": "MW",
            "licence_area": "EMids",
            "substation_number": ts_id,
            "substation_type": "Primary",
            "latitude": 52.0,
            "longitude": -1.0,
            "h3_res_5": 599423199024775167,
        }
        for ts_id in (1, 2, 3)
    ]
    return pt.DataFrame(rows).set_model(TimeSeriesMetadata).cast().validate()


def test_power_forecasts_round_trip_and_pruning_over_s3(s3_endpoint: str) -> None:
    """``write_power_forecasts`` writes to S3 and the filtered scan prunes partitions over S3."""
    settings = _s3_settings(s3_endpoint, "forecasts")
    uri = settings.power_forecasts_data_path
    opts = settings.storage_options
    assert uri.startswith("s3://")

    # Two experiments = two on-disk partitions.
    write_power_forecasts(
        _make_forecasts("expA", [1.0, 2.0, 3.0, 4.0]),
        uri,
        replace_partition=("expA", "s3_test"),
        storage_options=opts,
    )
    write_power_forecasts(
        _make_forecasts("expB", [5.0, 6.0, 7.0, 8.0]),
        uri,
        replace_partition=("expB", "s3_test"),
        storage_options=opts,
    )

    both = pl.read_delta(uri, storage_options=typeddict_to_dict(opts))
    assert set(both["experiment_name"].unique()) == {"expA", "expB"}
    assert both.height == 8

    # Partition pruning must survive over S3: the explain plan lists only expA's path.
    scan = pt.LazyFrame.from_existing(
        pl.scan_delta(uri, storage_options=typeddict_to_dict(opts))
    ).set_model(PowerForecast)
    plan = PopulationFilter(experiment_name="expA").apply(scan).explain()
    assert "experiment_name=expA" in plan
    assert "experiment_name=expB" not in plan


def test_nwp_round_trip_over_s3(s3_endpoint: str) -> None:
    """``write_nwp`` writes to S3 and ``Nwp.scan_delta`` reads it back with the same storage opts."""
    settings = _s3_settings(s3_endpoint, "nwp")
    uri = settings.nwp_data_path
    opts = settings.storage_options

    write_nwp(_make_nwp(), uri, opts)

    back = Nwp.scan_delta(uri, storage_options=opts).collect()
    assert back.height == 6
    assert set(back["ensemble_member"]) == set(range(6))


def test_metadata_parquet_round_trip_over_s3(s3_endpoint: str) -> None:
    """``upsert_metadata`` creates then reads a parquet file on S3 (object_exists + parquet IO)."""
    settings = _s3_settings(s3_endpoint, "meta")
    uri = settings.metadata_path
    opts = settings.storage_options

    assert not object_exists(uri, opts)

    metadata = _make_metadata()
    stats = upsert_metadata(metadata, uri, opts)
    assert stats["metadata_n_new_TimeSeriesIDs"] == 3

    assert object_exists(uri, opts)
    back = pl.read_parquet(uri, storage_options=typeddict_to_dict(opts))
    assert back.height == 3

    # A second upsert of identical rows is a no-op (exercises the read-existing S3 path).
    stats2 = upsert_metadata(metadata, uri, opts)
    assert stats2["metadata_n_new_TimeSeriesIDs"] == 0
