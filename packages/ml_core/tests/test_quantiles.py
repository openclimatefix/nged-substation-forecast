import polars as pl
import pytest
from contracts.common import DELIVERY_QUANTILES
from contracts.power_schemas import POWER_FCST_QUANTILE_COLUMNS
from ml_core.quantiles import to_stored_quantiles


def _members(time_series_ids: list[int], values: list[float | None]) -> pl.DataFrame:
    """One row per point value, numbered by ``ensemble_member`` within its series."""
    return pl.DataFrame(
        {
            "time_series_id": time_series_ids,
            "ensemble_member": list(range(len(values))),
            "power_fcst": values,
        },
        schema_overrides={"power_fcst": pl.Float64},
    )


def test_evenly_spaced_values_give_each_level_exactly():
    """101 values 0 … 100 put the linear empirical quantile at level q exactly on 100 q."""
    values: list[float | None] = [float(value) for value in range(101)]
    result = to_stored_quantiles(
        _members([1] * 101, values), value_col="power_fcst", drop_cols=["ensemble_member"]
    )
    assert result.columns == ["time_series_id", *POWER_FCST_QUANTILE_COLUMNS]
    assert result.height == 1
    for quantile, column in zip(DELIVERY_QUANTILES, POWER_FCST_QUANTILE_COLUMNS, strict=True):
        assert result[column].item() == pytest.approx(100 * quantile)
        assert result.schema[column] == pl.Float32


def test_one_value_fills_all_thirteen_columns():
    result = to_stored_quantiles(
        _members([1], [7.5]), value_col="power_fcst", drop_cols=["ensemble_member"]
    )
    assert result.select(POWER_FCST_QUANTILE_COLUMNS).row(0) == (7.5,) * 13


def test_a_null_value_is_dropped_before_the_reduction():
    """The group keeps its other values: the median of (1, 3) is 2, not a null."""
    result = to_stored_quantiles(
        _members([1, 1, 1], [1.0, None, 3.0]), value_col="power_fcst", drop_cols=["ensemble_member"]
    )
    assert result["power_fcst_p50"].item() == 2.0


def test_a_group_of_only_null_values_yields_no_row():
    result = to_stored_quantiles(
        _members([1, 2, 2], [4.0, None, None]),
        value_col="power_fcst",
        drop_cols=["ensemble_member"],
    )
    assert result["time_series_id"].to_list() == [1]
