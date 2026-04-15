from datetime import datetime
from typing import Optional

import numpy as np
import patito as pt
import polars as pl


class RawData(pt.Model):
    # Base columns
    timestamp: datetime
    power: float


class AllFeatures(RawData):
    # Optional engineered features
    power_lag_1: Optional[float] = pt.Field(ge=0)
    time_sin: Optional[float] = pt.Field(ge=-1.0, le=1.0)
    time_cos: Optional[float] = pt.Field(ge=-1.0, le=1.0)

    # Weather features...


STATIC_FEATURE_REGISTRY: dict[str, pl.Expr] = {
    "hour_of_day": pl.col("timestamp").dt.hour().alias("hour_of_day"),
    "time_sin": (pl.col("timestamp").dt.hour() / 24.0 * 2 * np.pi).sin().alias("time_sin"),
    "time_cos": (pl.col("timestamp").dt.hour() / 24.0 * 2 * np.pi).cos().alias("time_cos"),
    "is_weekend": pl.col("timestamp").dt.weekday().is_in([6, 7]).alias("is_weekend"),
}


# You don't want to hardcode "power_lag_1", "power_lag_2", etc., into a dictionary. For
# parameterized features, use small factory functions that return a pl.Expr.
def build_lag_expr(base_col: str, lag: int) -> pl.Expr:
    return pl.col(base_col).shift(lag).alias(f"{base_col}_lag_{lag}")


def build_rolling_mean_expr(base_col: str, window: int) -> pl.Expr:
    return (
        pl.col(base_col).rolling_mean(window_size=window).alias(f"{base_col}_rolling_mean_{window}")
    )
