"""Shared scaling and normalization utilities for ML models."""

import patito as pt
import polars as pl
from contracts.data_schemas import ScalingParams


def uint8_to_physical_unit(params: pt.DataFrame[ScalingParams]) -> list[pl.Expr]:
    """Convert uint8 columns back to physical units (Float32).

    Args:
        params: Patito DataFrame with scaling parameters.

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
