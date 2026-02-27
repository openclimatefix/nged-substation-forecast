"""Utility for scaling NWP variables to uint8."""

import polars as pl
from pathlib import Path

# Load scaling params from the provided CSV
SCALING_PARAMS_PATH = Path("packages/dynamical_data/scaling/ecmwf_scaling_params.csv")


def load_scaling_params() -> pl.DataFrame:
    if not SCALING_PARAMS_PATH.exists():
        # Fallback for if we are running from root or package
        alt_path = Path("packages/xgboost_forecaster/scaling_params.csv")
        if not alt_path.exists():
            # If it still doesn't exist, we might be in trouble, but let's try to find it
            raise FileNotFoundError(f"Scaling params not found at {SCALING_PARAMS_PATH}")
        return pl.read_csv(alt_path)
    return pl.read_csv(SCALING_PARAMS_PATH)


def get_scaling_expressions(params: pl.DataFrame, reverse: bool = False) -> list[pl.Expr]:
    exprs = []
    dtype = pl.UInt8

    for row in params.iter_rows(named=True):
        col = row["col_name"]
        b_min = row["buffered_min"]
        b_range = row["buffered_range"]

        if not reverse:
            # Raw -> UInt8
            # clipped_col = pl.col(col).clip(lower_bound=b_min, upper_bound=row["buffered_max"])
            # expr = ((clipped_col - b_min) / b_range) * 255
            # Simplified for speed/memory in lazy mode:
            expr = (
                (
                    (
                        (
                            pl.col(col)
                            .fill_nan(None)
                            .clip(lower_bound=b_min, upper_bound=row["buffered_max"])
                            - b_min
                        )
                        / b_range
                    )
                    * 255
                )
                .round()
                .cast(dtype)
                .alias(col)
            )
        else:
            # UInt8 -> Raw (Float32)
            expr = ((pl.col(col).cast(pl.Float32) / 255 * b_range) + b_min).alias(col)
        exprs.append(expr)

    # Categorical precipitation is already roughly 0-7, just cast it
    if not reverse:
        exprs.append(pl.col("categorical_precipitation_type_surface").cast(pl.UInt8))

    return exprs
