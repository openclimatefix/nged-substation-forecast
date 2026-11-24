"""Reduce a set of point values per forecast row to the 13 quantile columns of ``PowerForecast``.

A forecaster that produces several point values for one forecast row — one per ECMWF ENS weather
member, or one per historical power lag — calls `to_stored_quantiles` before it returns, so that
every forecaster writes the same quantile columns.
"""

from collections.abc import Sequence

import polars as pl
from contracts.common import DELIVERY_QUANTILES
from contracts.power_schemas import POWER_FCST_QUANTILE_COLUMNS


def to_stored_quantiles(
    frame: pl.DataFrame, *, value_col: str, drop_cols: Sequence[str] = ()
) -> pl.DataFrame:
    """Reduce the point values in ``value_col`` to one row per group of the remaining columns.

    The group is every column except ``value_col`` and ``drop_cols``. For a forecast, those
    columns are the ``PowerForecast`` key (``time_series_id``, ``power_fcst_init_time``,
    ``valid_time``) plus columns that are constant within the key, such as
    ``power_fcst_model_name`` and ``fold_id``. ``drop_cols`` names the column that tells the point
    values apart, such as ``ensemble_member``, so that it does not split the group.

    Each output row carries the linear empirical quantile of the group's values at each of the 13
    ``DELIVERY_QUANTILES``, as ``Float32`` columns ``power_fcst_p1`` … ``power_fcst_p99``.
    Empirical quantiles never decrease from one level to the next, so the output passes
    ``PowerForecast``'s ordering constraint by construction. A group with one value gets that
    value in all 13 columns. Null values are dropped first, so a group keeps its other values, and
    a group whose values are all null yields no row.

    Args:
        frame: One row per point value. Holds ``value_col``, the columns in ``drop_cols``, and the
            grouping columns.
        value_col: The column holding the point values.
        drop_cols: Columns dropped before grouping.

    Returns:
        One row per group: the grouping columns, in their input order, then the 13 quantile
        columns. Groups appear in the order of their first row in ``frame``.
    """
    values = frame.drop(drop_cols).filter(pl.col(value_col).is_not_null())
    group_cols = [column for column in values.columns if column != value_col]
    return values.group_by(group_cols, maintain_order=True).agg(
        pl.col(value_col)
        .quantile(quantile=quantile, interpolation="linear")
        .cast(pl.Float32)
        .alias(column)
        for quantile, column in zip(DELIVERY_QUANTILES, POWER_FCST_QUANTILE_COLUMNS, strict=True)
    )
