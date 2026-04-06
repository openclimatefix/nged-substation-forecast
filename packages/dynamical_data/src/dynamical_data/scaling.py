from pathlib import Path

import polars as pl


def load_scaling_params(csv_path: Path) -> pl.DataFrame:
    """Load scaling parameters from a CSV file."""
    return pl.read_csv(csv_path)


def scale_to_uint8(df: pl.DataFrame, scaling_params: pl.DataFrame) -> pl.DataFrame:
    """Scale numeric columns to uint8 [0, 255] based on scaling parameters.

    Args:
        df: Polars DataFrame with float32 columns.
        scaling_params: DataFrame with col_name, buffered_min, buffered_range.

    Returns:
        DataFrame with rescaled uint8 columns.
    """
    exprs = []

    for row in scaling_params.to_dicts():
        col_name = row["col_name"]
        if col_name not in df.columns:
            continue

        buffered_min = row["buffered_min"]
        buffered_max = row["buffered_max"]
        buffered_range = row["buffered_range"]

        # Handle NaNs first (Polars treats NaN as > any finite number)
        base_col = pl.col(col_name).fill_nan(None)

        clipped_col = base_col.clip(lower_bound=buffered_min, upper_bound=buffered_max)

        if buffered_range == 0.0:
            expr = pl.lit(0, dtype=pl.UInt8).alias(col_name)
        else:
            # Categorical variables like precipitation type are handled natively by this loop
            # because their `buffered_min` is 0.0 and `buffered_range` is 255.0 in the
            # scaling parameters, which naturally simplifies to a direct cast to UInt8
            # without scaling artifacts.
            expr = (
                (((clipped_col - buffered_min) / buffered_range) * 255)
                .round()
                .cast(pl.UInt8)
                .alias(col_name)
            )

        exprs.append(expr)

    return df.with_columns(exprs)


def recover_physical_units(df: pl.DataFrame, scaling_params: pl.DataFrame) -> pl.DataFrame:
    """Convert uint8 columns back to physical units.

    DATA TYPE TRANSITION RATIONALE:
    1. Disk (UInt8): Weather variables are stored as scaled 8-bit unsigned integers to save
       space and bandwidth.
    2. Interpolation (Float64): During spatial weighting and H3 grid joins, we use Float64
       to maintain precision and avoid rounding errors during aggregation.
    3. Memory/Model (Float32): After processing, we cast to Float32 for memory efficiency
       in the ML model.

    We do NOT cast back to UInt8 in memory because:
    - It would cause a "staircase" effect by quantizing interpolated values, defeating
      the purpose of 30-minute upsampling.
    - It risks silent underflow during differencing operations (e.g., calculating trends
      like temp_trend_6h = current - lagged).

    Args:
        df: Polars DataFrame with uint8 columns.
        scaling_params: DataFrame with col_name, buffered_min, buffered_range.

    Returns:
        DataFrame with recovered float32 columns.
    """
    exprs = []
    for row in scaling_params.to_dicts():
        col_name = row["col_name"]
        if col_name not in df.columns:
            continue

        buffered_min = row["buffered_min"]
        buffered_range = row["buffered_range"]

        if buffered_range == 0.0:
            expr = pl.lit(buffered_min, dtype=pl.Float32).alias(col_name)
        else:
            expr = (
                (pl.col(col_name).cast(pl.Float32) / 255.0) * buffered_range + buffered_min
            ).alias(col_name)

        exprs.append(expr)

    return df.with_columns(exprs)
