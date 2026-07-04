"""Unit tests for the ``power_forecasts`` storage-format helper in ``cv_assets``.

``_prepare_forecasts_for_storage`` is the write-side companion of
``_POWER_FORECASTS_WRITER_PROPERTIES``: it truncates the ``power_fcst`` mantissa and sorts rows
into the compression-friendly order. These tests pin down its numeric contract (bounded relative
error, non-finite passthrough, bit-level truncation) and the row order, without touching Dagster
or Delta — the end-to-end on-disk format is asserted in ``test_cv_power_forecasts.py``.
"""

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import patito as pt
import polars as pl
import pyarrow as pa
from contracts.power_schemas import PowerForecast

from nged_substation_forecast.defs.cv_assets import (
    _POWER_FCST_KEPT_MANTISSA_BITS,
    _POWER_FORECASTS_SORT_COLS,
    _prepare_forecasts_for_storage,
)

_T0 = datetime(2025, 7, 1, tzinfo=timezone.utc)


def _make_forecasts(power_fcst: list[float]) -> pt.DataFrame[PowerForecast]:
    """Build a valid ``PowerForecast`` frame with deliberately unsorted key columns."""
    n = len(power_fcst)
    rows = {
        # Reverse-ordered keys so the helper's sort has something to do.
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
        "fold_id": ["smoke_test"] * n,
    }
    return PowerForecast.DataFrame(rows).cast().validate()


def test_truncation_bounds_relative_error() -> None:
    values = [123.456, -0.0871, 1e-6, 0.0, 987654.0, -4321.99]
    table = _prepare_forecasts_for_storage(_make_forecasts(values))
    stored = np.sort(np.asarray(table["power_fcst"]))
    original = np.sort(np.array(values, dtype=np.float32))

    # Truncation toward zero: |stored| <= |original| and relative error < 2**-kept_bits.
    assert (np.abs(stored) <= np.abs(original)).all()
    nonzero = original != 0
    rel_err = np.abs(stored[nonzero] - original[nonzero]) / np.abs(original[nonzero])
    assert (rel_err < 2.0**-_POWER_FCST_KEPT_MANTISSA_BITS).all()


def test_truncation_zeroes_low_mantissa_bits() -> None:
    table = _prepare_forecasts_for_storage(_make_forecasts([math.pi, -math.e, 0.123456789]))
    bits = np.asarray(table["power_fcst"], dtype=np.float32).view(np.uint32)
    discarded_bits = np.uint32((1 << (23 - _POWER_FCST_KEPT_MANTISSA_BITS)) - 1)
    assert (bits & discarded_bits == 0).all()


def test_non_finite_values_pass_through_unchanged() -> None:
    table = _prepare_forecasts_for_storage(_make_forecasts([math.nan, math.inf, -math.inf, 1.5]))
    stored = np.asarray(table["power_fcst"])
    assert np.isnan(stored).sum() == 1
    assert np.isposinf(stored).sum() == 1
    assert np.isneginf(stored).sum() == 1
    # 1.5 has an exactly-representable (all-zero) low mantissa, so it survives verbatim.
    assert (stored == 1.5).sum() == 1


def test_rows_are_sorted_member_adjacent() -> None:
    forecasts = _make_forecasts([float(i) for i in range(7)])
    table = _prepare_forecasts_for_storage(forecasts)

    result = pl.from_arrow(table)
    assert isinstance(result, pl.DataFrame)
    assert result.height == forecasts.height
    assert set(result.columns) == set(forecasts.columns)
    key = result.select(pl.struct(list(_POWER_FORECASTS_SORT_COLS)).alias("key"))["key"]
    assert key.is_sorted()


def test_returns_arrow_table_with_schema_intact() -> None:
    forecasts = _make_forecasts([1.0, 2.0])
    table = _prepare_forecasts_for_storage(forecasts)
    assert isinstance(table, pa.Table)
    assert table.num_rows == 2
    # Same column names and Arrow types as the untouched conversion.
    assert table.schema == forecasts.to_arrow().schema
