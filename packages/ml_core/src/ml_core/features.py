"""Shared feature engineering logic for ML models."""

import numpy as np
import polars as pl


def add_cyclical_temporal_features(df: pl.DataFrame, time_col: str = "valid_time") -> pl.DataFrame:
    """Add cyclical and standard temporal features.

    Args:
        df: DataFrame with a time column.
        time_col: Name of the time column.

    Returns:
        DataFrame with added temporal features (hour_sin, hour_cos, day_of_year_sin,
        day_of_year_cos, day_of_week).
    """
    return df.with_columns(
        # Cyclical Hour (24h)
        hour_sin=(pl.col(time_col).dt.hour() * 2 * np.pi / 24).sin().cast(pl.Float32),
        hour_cos=(pl.col(time_col).dt.hour() * 2 * np.pi / 24).cos().cast(pl.Float32),
        # Cyclical Day of Year (365.25d)
        day_of_year_sin=(pl.col(time_col).dt.ordinal_day() * 2 * np.pi / 365.25)
        .sin()
        .cast(pl.Float32),
        day_of_year_cos=(pl.col(time_col).dt.ordinal_day() * 2 * np.pi / 365.25)
        .cos()
        .cast(pl.Float32),
        # Day of week (0-6)
        day_of_week=pl.col(time_col).dt.weekday().cast(pl.Int8),
    )
