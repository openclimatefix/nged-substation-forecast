"""Shared data splitting and loading logic for ML models."""

from datetime import datetime

import polars as pl


def split_lazyframe_by_time(
    lf: pl.LazyFrame,
    start_time: str | datetime,
    end_time: str | datetime,
    time_col: str = "valid_time",
) -> pl.LazyFrame:
    """Filter a LazyFrame by a time range.

    Args:
        lf: The input LazyFrame.
        start_time: The start of the time range (inclusive).
        end_time: The end of the time range (inclusive).
        time_col: The name of the time column.

    Returns:
        The filtered LazyFrame.
    """
    return lf.filter(pl.col(time_col).is_between(start_time, end_time))
