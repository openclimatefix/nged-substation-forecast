"""Tests for ``write_nwp`` — the ``nwp`` storage format end-to-end.

Writes real (tiny) ``Nwp`` frames into a temp Delta table and asserts the on-disk format:
ZSTD with parquet's *default* encodings (measured better than ``BYTE_STREAM_SPLIT`` /
``DELTA_BINARY_PACKED`` for this table — see ``delta_store.nwp``), member-early sort within
each file, every continuous variable rounded to ``NWP_SIGNIFICAND_BITS``, and appends landing
as separate ``(nwp_model_id, init_time)`` Hive partitions.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import patito as pt
import polars as pl
import pyarrow.parquet as pq
from contracts.weather_schemas import Nwp
from delta_store.nwp import NWP_SIGNIFICAND_BITS, NWP_SORT_COLS, write_nwp

_T0 = datetime(2025, 6, 1, tzinfo=timezone.utc)

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
"""Per-variable bases well inside each field's ``ge``/``le`` bounds, with full-entropy
mantissas so the significand-rounding assertions have something to measure."""


def _make_nwp(n: int = 6, *, init_time: datetime = _T0) -> pt.DataFrame[Nwp]:
    """Build a valid ``Nwp`` frame with deliberately unsorted key columns."""
    rows = {
        # Reverse-ordered members and cycling valid_times so the writer's sort has work to do.
        "nwp_model_id": ["ECMWF_ENS_0_25_degree"] * n,
        "init_time": [init_time] * n,
        "valid_time": [init_time + timedelta(hours=(i % 3) + 1) for i in range(n)],
        "ensemble_member": list(range(n - 1, -1, -1)),
        "h3_index": [100 + i for i in range(n)],
        "categorical_precipitation_type_surface": [1] * n,
        **{
            var: [base * (1 + 0.003 * i) for i in range(n)]
            for var, base in _CONTINUOUS_BASE_VALUES.items()
        },
    }
    return Nwp.DataFrame(rows).cast().validate()


def test_on_disk_format(tmp_path: Path) -> None:
    table = tmp_path / "nwp"
    write_nwp(_make_nwp(), table)

    parquet_files = list(table.rglob("*.parquet"))
    assert parquet_files
    for parquet_file in parquet_files:
        row_group = pq.ParquetFile(parquet_file).metadata.row_group(0)
        columns = {
            row_group.column(i).path_in_schema: row_group.column(i)
            for i in range(row_group.num_columns)
        }
        for name, column in columns.items():
            assert column.compression == "ZSTD", f"{name} written as {column.compression}"
            # Unlike power_forecasts, this table measures *worse* with the special encodings —
            # write_nwp must stick to parquet's defaults (see the delta_store.nwp docstring).
            assert "BYTE_STREAM_SPLIT" not in column.encodings, name
            assert "DELTA_BINARY_PACKED" not in column.encodings, name

        rows = pl.read_parquet(parquet_file)
        # nwp_model_id and init_time are Hive partition values, not parquet columns.
        key = rows.select(k=pl.struct([c for c in NWP_SORT_COLS if c in rows.columns]))
        assert key["k"].is_sorted()


def test_continuous_vars_rounded_to_significand_bits(tmp_path: Path) -> None:
    table = tmp_path / "nwp"
    n = 6
    write_nwp(_make_nwp(n), table)

    stored_df = pl.read_delta(str(table))
    discarded = np.uint32((1 << (23 - (NWP_SIGNIFICAND_BITS - 1))) - 1)
    for var, base in _CONTINUOUS_BASE_VALUES.items():
        stored = np.sort(stored_df[var].to_numpy())
        expected = np.sort(np.array([base * (1 + 0.003 * i) for i in range(n)], dtype=np.float32))
        rel_err = np.abs(stored - expected) / np.abs(expected)
        assert (rel_err <= 2.0**-NWP_SIGNIFICAND_BITS).all(), var
        assert (stored.view(np.uint32) & discarded == 0).all(), var


def test_successive_appends_create_separate_partitions(tmp_path: Path) -> None:
    table = tmp_path / "nwp"
    n = 4
    t1 = _T0 + timedelta(days=1)
    write_nwp(_make_nwp(n), table)
    write_nwp(_make_nwp(n, init_time=t1), table)

    partition_dirs = sorted(
        p.name for p in (table / "nwp_model_id=ECMWF_ENS_0_25_degree").iterdir() if p.is_dir()
    )
    assert len(partition_dirs) == 2
    assert all(name.startswith("init_time=") for name in partition_dirs)

    # Both partitions' rows are intact, and the table round-trips through the read contract.
    scan = Nwp.scan_delta(table)
    collected = scan.collect()
    assert collected.height == 2 * n
    assert collected.filter(pl.col("init_time") == t1).height == n
    assert collected.schema["temperature_2m"] == pl.Float32
    assert collected.schema["nwp_model_id"] == pl.Enum(["ECMWF_ENS_0_25_degree"])
