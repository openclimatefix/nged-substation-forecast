"""Tests for ``write_power_forecasts`` — the ``power_forecasts`` storage format end-to-end.

Writes real (tiny) ``PowerForecast`` frames into a temp Delta table and asserts the on-disk
format: ZSTD + the per-column encodings, member-adjacent sort within each file, ``power_fcst``
rounded to ``POWER_FCST_SIGNIFICAND_BITS``, and the replace-partition / append semantics that
``cv_power_forecasts`` relies on for idempotent re-materialisation.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import patito as pt
import polars as pl
import pyarrow.parquet as pq
from contracts.power_schemas import PowerForecast
from delta_store.power_forecasts import (
    POWER_FCST_SIGNIFICAND_BITS,
    POWER_FORECASTS_SORT_COLS,
    write_power_forecasts,
)

_T0 = datetime(2025, 7, 1, tzinfo=timezone.utc)


def _make_forecasts(
    power_fcst: list[float], *, fold_id: str = "smoke_test"
) -> pt.DataFrame[PowerForecast]:
    """Build a valid ``PowerForecast`` frame with deliberately unsorted key columns."""
    n = len(power_fcst)
    rows = {
        # Reverse-ordered keys so the writer's sort has something to do.
        "valid_time": [_T0 + timedelta(hours=n - i) for i in range(n)],
        "time_series_id": list(range(n, 0, -1)),
        "ensemble_member": [i % 3 for i in range(n)],
        "ml_flow_experiment_id": [None] * n,
        "nwp_init_time": [_T0] * n,
        "power_fcst_model_name": ["xgboost"] * n,
        "experiment_name": ["exp_storage_test"] * n,
        "power_fcst_model_version": [1] * n,
        "power_fcst_init_time": [_T0] * n,
        "power_fcst": power_fcst,
        "fold_id": [fold_id] * n,
    }
    return PowerForecast.DataFrame(rows).cast().validate()


def test_on_disk_format(tmp_path: Path) -> None:
    table = tmp_path / "power_forecasts"
    write_power_forecasts(
        _make_forecasts([123.456, -0.0871, 3.14159, 987654.0]),
        table,
        replace_partition=("exp_storage_test", "smoke_test"),
    )

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
        assert "DELTA_BINARY_PACKED" in columns["valid_time"].encodings
        assert "BYTE_STREAM_SPLIT" in columns["power_fcst"].encodings

        rows = pl.read_parquet(parquet_file)
        key = rows.select(k=pl.struct([c for c in POWER_FORECASTS_SORT_COLS if c in rows.columns]))
        assert key["k"].is_sorted()


def test_power_fcst_rounded_to_significand_bits(tmp_path: Path) -> None:
    table = tmp_path / "power_forecasts"
    original = [123.456, -0.0871, 1e-6, 987654.0, -4321.99]
    write_power_forecasts(
        _make_forecasts(original),
        table,
        replace_partition=("exp_storage_test", "smoke_test"),
    )

    stored = np.sort(pl.read_delta(str(table))["power_fcst"].to_numpy())
    expected = np.sort(np.array(original, dtype=np.float32))
    rel_err = np.abs(stored - expected) / np.abs(expected)
    assert (rel_err <= 2.0**-POWER_FCST_SIGNIFICAND_BITS).all()
    discarded = np.uint32((1 << (23 - (POWER_FCST_SIGNIFICAND_BITS - 1))) - 1)
    assert (stored.view(np.uint32) & discarded == 0).all()


def test_replace_partition_then_append(tmp_path: Path) -> None:
    table = tmp_path / "power_forecasts"
    partition = ("exp_storage_test", "smoke_test")

    # First materialisation: replace + append lands both chunks.
    write_power_forecasts(_make_forecasts([1.0, 2.0]), table, replace_partition=partition)
    write_power_forecasts(_make_forecasts([3.0]), table)
    assert pl.read_delta(str(table)).height == 3

    # Re-materialisation: the replace-chunk clears the partition's prior three rows.
    write_power_forecasts(_make_forecasts([4.0]), table, replace_partition=partition)
    assert pl.read_delta(str(table)).height == 1

    # Other partitions are untouched by a replace.
    write_power_forecasts(
        _make_forecasts([5.0], fold_id="other_fold"),
        table,
        replace_partition=("exp_storage_test", "other_fold"),
    )
    write_power_forecasts(_make_forecasts([6.0]), table, replace_partition=partition)
    result = pl.read_delta(str(table))
    assert result.height == 2
    assert set(result["fold_id"].to_list()) == {"smoke_test", "other_fold"}


def test_empty_replace_chunk_clears_partition(tmp_path: Path) -> None:
    table = tmp_path / "power_forecasts"
    partition = ("exp_storage_test", "smoke_test")
    write_power_forecasts(_make_forecasts([1.0, 2.0]), table, replace_partition=partition)

    # An empty first chunk must still clear the partition (cv_power_forecasts relies on this
    # so a re-materialisation that produces no forecasts never leaves stale rows behind).
    write_power_forecasts(_make_forecasts([]), table, replace_partition=partition)
    assert pl.read_delta(str(table)).height == 0
