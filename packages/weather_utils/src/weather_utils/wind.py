"""Wind quantities shared by NWP feature engineering and display."""

import math
from typing import Literal

import polars as pl


def wind_feature_expressions(height: Literal["10m", "100m"]) -> tuple[pl.Expr, pl.Expr]:
    """Derive vector speed and meteorological bearing from signed component columns.

    u is eastward velocity and v is northward velocity. Polars uses atan2(y, x); passing u first
    and v second measures the angle from the northward axis. The components describe travel
    towards a direction, so adding 180° gives the bearing the wind comes from, and modulo 360
    keeps the result in [0°, 360°). Feature engineering calls this after component interpolation;
    native-step study extracts and dashboard plots use the expressions without temporal resampling.

    Bearing is the direction the wind comes from: north is 0°, east is 90°. At exact calm,
    the atan2 convention yields 180°; the bearing has no physical meaning there. Null components
    propagate to null features.

    See <https://openclimatefix.github.io/nged-substation-forecast/architecture/nwp-variable-conventions/#interpolate-components-before-deriving-speed-and-bearing>.
    """
    u, v = pl.col(f"wind_u_{height}"), pl.col(f"wind_v_{height}")
    return (
        (u**2 + v**2).sqrt().cast(pl.Float32).alias(f"wind_speed_{height}"),
        ((pl.arctan2(u, v) * (180 / math.pi) + 180) % 360)
        .cast(pl.Float32)
        .alias(f"wind_direction_{height}"),
    )
