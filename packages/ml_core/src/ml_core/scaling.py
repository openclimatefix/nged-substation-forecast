"""Shared scaling and normalization utilities for ML models."""

import polars as pl


def uint8_to_physical_unit(params: pl.DataFrame) -> list[pl.Expr]:
    """Convert uint8 columns back to physical units (Float32).

    Args:
        params: DataFrame with scaling parameters (col_name, buffered_min, buffered_range).

    Returns:
        List of Polars expressions for the conversion.
    """
    exprs = []
    for row in params.iter_rows(named=True):
        col = row["col_name"]
        b_min = row["buffered_min"]
        b_range = row["buffered_range"]

        # UInt8 -> Raw (Float32)
        expr = ((pl.col(col).cast(pl.Float32) / 255 * b_range) + b_min).alias(col)
        exprs.append(expr)

    return exprs
