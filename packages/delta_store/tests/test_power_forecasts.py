"""Tests for ``write_power_forecasts`` — the ``power_forecasts`` storage format end-to-end.

Writes real (tiny) ``PowerForecast`` frames into a temp Delta table and asserts the on-disk
format: ZSTD + the per-column encodings, the primary-key sort within each file, the 13 quantile
columns rounded to ``POWER_FCST_SIGNIFICAND_BITS``, and the replace-partition / append semantics
that ``cv_power_forecasts`` relies on for idempotent re-materialisation.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import patito as pt
import polars as pl
import pyarrow.parquet as pq
from contracts.power_schemas import POWER_FCST_QUANTILE_COLUMNS, PowerForecast
from delta_store.power_forecasts import (
    POWER_FCST_SIGNIFICAND_BITS,
    POWER_FORECASTS_SORT_COLS,
    write_power_forecasts,
)

_T0 = datetime(2025, 7, 1, tzinfo=UTC)


def _make_forecasts(
    power_fcst: list[float], *, fold_id: str = "smoke_test"
) -> pt.DataFrame[PowerForecast]:
    """Build a valid ``PowerForecast`` frame with deliberately unsorted key columns.

    Quantile column ``k`` (0 to 12) of a row holds ``power_fcst + k``, so the 13 columns of every
    row are distinct and increasing.
    """
    n = len(power_fcst)
    rows = {
        # Reverse-ordered keys so the writer's sort has something to do.
        "valid_time": [_T0 + timedelta(hours=n - i) for i in range(n)],
        "time_series_id": list(range(n, 0, -1)),
        "ml_flow_experiment_id": [None] * n,
        "nwp_init_time": [_T0] * n,
        "power_fcst_model_name": ["climatology"] * n,
        "experiment_name": ["exp_storage_test"] * n,
        "power_fcst_model_version": [1] * n,
        "power_fcst_init_time": [_T0] * n,
        **{
            column: [value + position for value in power_fcst]
            for position, column in enumerate(POWER_FCST_QUANTILE_COLUMNS)
        },
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
        for column in POWER_FCST_QUANTILE_COLUMNS:
            assert "BYTE_STREAM_SPLIT" in columns[column].encodings, column

        rows = pl.read_parquet(parquet_file)
        key = rows.select(k=pl.struct([c for c in POWER_FORECASTS_SORT_COLS if c in rows.columns]))
        assert key["k"].is_sorted()


def test_every_quantile_column_rounded_to_significand_bits(tmp_path: Path) -> None:
    table = tmp_path / "power_forecasts"
    original = [123.456, -0.0871, 1e-6, 987654.0, -4321.99]
    forecasts = _make_forecasts(original)
    write_power_forecasts(
        forecasts,
        table,
        replace_partition=("exp_storage_test", "smoke_test"),
    )

    stored_frame = pl.read_delta(str(table))
    discarded = np.uint32((1 << (23 - (POWER_FCST_SIGNIFICAND_BITS - 1))) - 1)
    for column in POWER_FCST_QUANTILE_COLUMNS:
        stored = np.sort(stored_frame[column].to_numpy())
        expected = np.sort(forecasts[column].to_numpy())
        rel_err = np.abs(stored - expected) / np.abs(expected)
        assert (rel_err <= 2.0**-POWER_FCST_SIGNIFICAND_BITS).all(), column
        assert (stored.view(np.uint32) & discarded == 0).all(), column
    # Rounding to nearest never reverses two values, so the stored rows still validate.
    PowerForecast.validate(stored_frame)


def test_the_power_fcst_significand_bits_is_thirteen() -> None:
    """Pins the literal, not a value derived from the constant.

    ``test_every_quantile_column_rounded_to_significand_bits`` above computes its tolerance *from*
    ``POWER_FCST_SIGNIFICAND_BITS``, so it stays green no matter what the constant is set to.
    Each ``PowerForecast`` quantile column's description asserts "max relative error 2^-13" in
    prose that a changed constant would leave silently wrong.
    """
    assert POWER_FCST_SIGNIFICAND_BITS == 13


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


def test_sort_cols_lead_with_time_series_id() -> None:
    """The sort order is this module's choice, even though its columns come from the primary key.

    `POWER_FORECASTS_SORT_COLS` is `PowerForecast.PRIMARY_KEY` so the two cannot disagree about
    *which* columns matter. The order is a separate decision: leading with `time_series_id` is
    what lets row-group statistics prune a single-series scan, and putting `valid_time` last is
    what places the successive half-hours of one forecast run on adjacent rows for
    `BYTE_STREAM_SPLIT`. A primary key reordered for any other reason would still be a correct key
    and a worse layout.
    """
    assert POWER_FORECASTS_SORT_COLS == ("time_series_id", "power_fcst_init_time", "valid_time")
