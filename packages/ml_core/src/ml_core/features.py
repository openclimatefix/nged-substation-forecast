"""Shared feature engineering logic for ML models."""

from typing import Any, TypeVar, cast

import numpy as np
import polars as pl

T = TypeVar("T", pl.DataFrame, pl.LazyFrame)


def add_cyclical_temporal_features(df: T, time_col: str = "valid_time") -> T:
    """Add cyclical and standard temporal features.

    Args:
        df: DataFrame or LazyFrame with a time column.
        time_col: Name of the time column.

    Returns:
        DataFrame or LazyFrame with added temporal features (hour_sin, hour_cos,
        day_of_year_sin, day_of_year_cos, day_of_week).
    """
    return cast(
        T,
        cast(Any, df).with_columns(
            # Cyclical Hour (24h)
            hour_sin=(
                (pl.col(time_col).dt.hour() + pl.col(time_col).dt.minute() / 60.0) * 2 * np.pi / 24
            )
            .sin()
            .cast(pl.Float32),
            hour_cos=(
                (pl.col(time_col).dt.hour() + pl.col(time_col).dt.minute() / 60.0) * 2 * np.pi / 24
            )
            .cos()
            .cast(pl.Float32),
            # Cyclical Day of Year (365.25d)
            day_of_year_sin=(pl.col(time_col).dt.ordinal_day() * 2 * np.pi / 365.25)
            .sin()
            .cast(pl.Float32),
            day_of_year_cos=(pl.col(time_col).dt.ordinal_day() * 2 * np.pi / 365.25)
            .cos()
            .cast(pl.Float32),
            # Day of week (1-7)
            day_of_week=pl.col(time_col).dt.weekday().cast(pl.Int8),
        ),
    )
