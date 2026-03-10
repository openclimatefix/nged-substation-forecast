from pathlib import Path

import polars as pl

# TODO: Most of (maybe *all of*) this file should be moved to packages/dynamical_data/scaling/
scaling_params_path = Path("packages/dynamical_data/scaling/ecmwf_scaling_params.csv")


def load_scaling_params(path: Path | None = None) -> pl.DataFrame:
    if path is None:
        path = scaling_params_path
    if not path.exists():
        # Fallback for if we are running from root or package
        alt_path = Path("packages/xgboost_forecaster/scaling_params.csv")
        if not alt_path.exists():
            # If it still doesn't exist, we might be in trouble, but let's try to find it
            raise FileNotFoundError(
                f"Scaling params not found at {path}. Please set the XGBOOST_SCALING_PARAMS_PATH environment variable."
            )
        return pl.read_csv(alt_path)
    return pl.read_csv(path)


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
