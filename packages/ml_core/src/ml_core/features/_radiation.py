"""Interval-aware radiation reconstruction on the half-hour grid.

Short-wave uses daylight midpoint indices to shape each native window before conserving its
energy. Long-wave fits PCHIP through native-window midpoints. Missing native windows stay missing.
Policies and worked plots:
<https://openclimatefix.github.io/nged-substation-forecast/architecture/nwp-variable-conventions/#radiation-resampling>.
"""

from datetime import timedelta
from itertools import pairwise
from math import pi
from typing import Final

import polars as pl
from contracts.weather_schemas import ECMWF_ENS_LEAD_TIME_HOURS

SHORT_WAVE: Final[str] = "downward_short_wave_radiation_flux_surface"
LONG_WAVE: Final[str] = "downward_long_wave_radiation_flux_surface"
RADIATION_COLUMNS: Final[frozenset[str]] = frozenset({SHORT_WAVE, LONG_WAVE})
DAYLIGHT_FLOOR_W_M2: Final[float] = 1.0


def _half_hour_clear_sky(*, targets: pl.LazyFrame, coordinates: pl.LazyFrame) -> pl.LazyFrame:
    """Evaluate Haurwitz means using two 15-minute midpoint samples per target.

    NOAA fractional-year solar geometry includes longitude and equation of time. Geometry is
    independent of run/member, so compute it once per series and target timestamp.
    """
    samples = (
        targets.select("time_series_id", "valid_time")
        .unique()
        .join(coordinates, on="time_series_id", how="left", validate="m:1")
        .with_columns(_offset=pl.lit([450, 1350]))
        .explode("_offset")
        .with_columns(_stamp=pl.col("valid_time") - pl.duration(seconds=pl.col("_offset")))
        .with_columns(
            _hour=pl.col("_stamp").dt.hour()
            + pl.col("_stamp").dt.minute() / 60
            + pl.col("_stamp").dt.second() / 3600,
            _latitude=pl.col("latitude").cast(pl.Float64).radians(),
        )
        .with_columns(
            _gamma=2
            * pi
            / pl.when(pl.col("_stamp").dt.is_leap_year()).then(366).otherwise(365)
            * (pl.col("_stamp").dt.ordinal_day() - 1 + (pl.col("_hour") - 12) / 24)
        )
    )
    gamma = pl.col("_gamma")
    samples = samples.with_columns(
        _declination=0.006918
        - 0.399912 * gamma.cos()
        + 0.070257 * gamma.sin()
        - 0.006758 * (2 * gamma).cos()
        + 0.000907 * (2 * gamma).sin()
        - 0.002697 * (3 * gamma).cos()
        + 0.00148 * (3 * gamma).sin(),
        _equation=229.18
        * (
            0.000075
            + 0.001868 * gamma.cos()
            - 0.032077 * gamma.sin()
            - 0.014615 * (2 * gamma).cos()
            - 0.040849 * (2 * gamma).sin()
        ),
    ).with_columns(
        _cosine=(
            pl.col("_latitude").sin() * pl.col("_declination").sin()
            + pl.col("_latitude").cos()
            * pl.col("_declination").cos()
            * ((pl.col("_hour") * 60 + pl.col("_equation") + 4 * pl.col("longitude")) / 4 - 180)
            .radians()
            .cos()
        ).clip(lower_bound=0)
    )
    sky = samples.group_by("time_series_id", "valid_time").agg(
        _clear_sky=pl.when(pl.col("_cosine") > 0)
        .then(1098 * pl.col("_cosine") * (-0.059 / pl.col("_cosine")).exp())
        .otherwise(0.0)
        .mean()
    )
    # Half-hour support, rather than native windows, detects short northern summer nights.
    return (
        sky.with_columns(
            _day_start=(pl.col("_clear_sky") > 0)
            & (
                (pl.col("_clear_sky").shift().over("time_series_id", order_by="valid_time") <= 0)
                | (
                    (
                        pl.col("valid_time")
                        - pl.col("valid_time").shift().over("time_series_id", order_by="valid_time")
                    )
                    != timedelta(minutes=30)
                )
            ).fill_null(True)
        )
        .with_columns(
            _daylight_block=pl.col("_day_start")
            .cast(pl.Int64)
            .cum_sum()
            .over("time_series_id", order_by="valid_time")
        )
        .drop("_day_start")
    )


def _native_targets(*, native: pl.LazyFrame, group_cols: list[str]) -> pl.LazyFrame:
    """Expand supported source windows, using scheduled widths even when rows are missing."""
    leads = ECMWF_ENS_LEAD_TIME_HOURS
    schedule = (
        pl.DataFrame(
            {
                "_lead": [timedelta(hours=lead) for lead in leads[1:]],
                "_width_minutes": [(end - start) * 60 for start, end in pairwise(leads)],
            }
        )
        .cast({"_lead": pl.Duration("us")})
        .lazy()
    )
    windows = (
        native.with_columns(
            _lead=(pl.col("valid_time") - pl.col("nwp_init_time")).cast(pl.Duration("us"))
        )
        .join(schedule, on="_lead", how="inner")
        .rename({"valid_time": "_window_end"})
    )
    windows = windows.with_columns(
        _window_start=pl.col("_window_end") - pl.duration(minutes=pl.col("_width_minutes")),
        _anchor_second=pl.col("_window_end").dt.epoch("s") - pl.col("_width_minutes") * 30,
    )
    if SHORT_WAVE in native.collect_schema():
        windows = windows.with_columns(
            _break=pl.col(SHORT_WAVE).is_null()
            | pl.col(SHORT_WAVE).shift().over(group_cols, order_by="_window_end").is_null()
            | (
                pl.col("_window_start")
                != pl.col("_window_end").shift().over(group_cols, order_by="_window_end")
            ).fill_null(True)
        ).with_columns(
            _segment=pl.col("_break")
            .cast(pl.Int64)
            .cum_sum()
            .over(group_cols, order_by="_window_end")
        )
    return windows.with_columns(
        valid_time=pl.datetime_ranges(
            start=pl.col("_window_start") + timedelta(minutes=30),
            end=pl.col("_window_end"),
            interval="30m",
        )
    ).explode("valid_time", empty_as_null=False)


def _shape_short_wave(
    *, targets: pl.LazyFrame, group_cols: list[str], coordinates: pl.LazyFrame
) -> pl.LazyFrame:
    """Interpolate indices in daylight blocks, then conserve complete native-window energy."""
    window_keys = [*group_cols, "_window_end"]
    sky = _half_hour_clear_sky(targets=targets, coordinates=coordinates)
    targets = targets.join(sky, on=["time_series_id", "valid_time"], how="left").with_columns(
        _native_clear_sky=pl.col("_clear_sky").mean().over(window_keys),
        _target_second=pl.col("valid_time").dt.epoch("s") - 900,
    )
    interpolation_keys = [*group_cols, "_segment", "_daylight_block"]
    anchors = (
        targets.filter(
            (pl.col("_clear_sky") > 0)
            & pl.col(SHORT_WAVE).is_not_null()
            & (pl.col("_native_clear_sky") >= DAYLIGHT_FLOOR_W_M2)
        )
        .select(
            *interpolation_keys,
            "_anchor_second",
            _ratio=pl.col(SHORT_WAVE) / pl.col("_native_clear_sky"),
        )
        .unique()
    )
    shaped = targets.sort("_target_second")
    for strategy, label in (("backward", "earlier"), ("forward", "later")):
        shaped = shaped.join_asof(
            anchors.rename(
                {"_anchor_second": f"_{label}_second", "_ratio": f"_{label}_ratio"}
            ).sort(f"_{label}_second"),
            left_on="_target_second",
            right_on=f"_{label}_second",
            by=interpolation_keys,
            strategy=strategy,
            check_sortedness=False,
        )
    span = pl.col("_later_second") - pl.col("_earlier_second")
    ratio = (
        pl.when(span > 0)
        .then(
            pl.col("_earlier_ratio")
            + (pl.col("_later_ratio") - pl.col("_earlier_ratio"))
            * (pl.col("_target_second") - pl.col("_earlier_second"))
            / span
        )
        .otherwise(pl.coalesce("_earlier_ratio", "_later_ratio"))
    )
    shaped = shaped.with_columns(_shape=ratio.fill_null(1.0) * pl.col("_clear_sky"))
    shaped = shaped.with_columns(_shape_mean=pl.col("_shape").mean().over(window_keys))
    # Local solar weights are the fallback if no usable index exists or the interpolated sum is
    # zero. A positive fully-dark mean has no solar shape; retain its energy as a constant.
    return shaped.with_columns(
        **{
            SHORT_WAVE: pl.col(SHORT_WAVE)
            * pl.when(pl.col("_shape_mean") > 0)
            .then(pl.col("_shape") / pl.col("_shape_mean"))
            .when(pl.col("_native_clear_sky") > 0)
            .then(pl.col("_clear_sky") / pl.col("_native_clear_sky"))
            .otherwise(1.0)
        }
    )


def _pchip_edge_slope(
    *, width: pl.Expr, next_width: pl.Expr, slope: pl.Expr, next_slope: pl.Expr
) -> pl.Expr:
    """Limit the one-sided endpoint derivative used by PCHIP."""
    estimate = ((2 * width + next_width) * slope - width * next_slope) / (width + next_width)
    return (
        pl.when(next_width.is_null())
        .then(slope)
        .when(estimate * slope <= 0)
        .then(0.0)
        .when((slope * next_slope <= 0) & (estimate.abs() > 3 * slope.abs()))
        .then(3 * slope)
        .otherwise(estimate)
    )


def _shape_long_wave(*, targets: pl.LazyFrame, group_cols: list[str]) -> pl.LazyFrame:
    """Evaluate midpoint PCHIP lazily, with constant edges and no interpolation across gaps.

    Native interval means act as midpoint estimates, without enforcing interval energy. PCHIP's
    harmonic interior slopes and limited endpoint slopes preserve the shape of the supplied means.
    """
    window_keys = [*group_cols, "_window_end"]
    windows = targets.select(*window_keys, "_window_start", "_anchor_second", LONG_WAVE).unique()
    for shift, label in ((1, "earlier"), (-1, "later")):
        neighbour = pl.col(LONG_WAVE).shift(shift).over(group_cols, order_by="_window_end")
        anchor = pl.col("_anchor_second").shift(shift).over(group_cols, order_by="_window_end")
        contiguous = (
            pl.col("_window_start")
            == pl.col("_window_end").shift(1).over(group_cols, order_by="_window_end")
            if shift == 1
            else pl.col("_window_end")
            == pl.col("_window_start").shift(-1).over(group_cols, order_by="_window_end")
        )
        windows = windows.with_columns(
            **{
                f"_{label}_value": neighbour.cast(pl.Float64),
                f"_{label}_width": pl.when(contiguous & neighbour.is_not_null()).then(
                    (anchor - pl.col("_anchor_second")).abs()
                ),
                f"_{label}_secant": (neighbour.cast(pl.Float64) - pl.col(LONG_WAVE))
                / (anchor - pl.col("_anchor_second")),
            }
        )
    left_h, right_h = pl.col("_earlier_width"), pl.col("_later_width")
    left_d, right_d = pl.col("_earlier_secant"), pl.col("_later_secant")
    left_weight, right_weight = 2 * right_h + left_h, right_h + 2 * left_h
    derivative = (
        pl.when(left_h.is_not_null() & right_h.is_not_null())
        .then(
            pl.when(left_d * right_d > 0)
            .then((left_weight + right_weight) / (left_weight / left_d + right_weight / right_d))
            .otherwise(0.0)
        )
        .when(left_h.is_null())
        .then(
            _pchip_edge_slope(
                width=right_h,
                next_width=right_h.shift(-1).over(group_cols, order_by="_window_end"),
                slope=right_d,
                next_slope=right_d.shift(-1).over(group_cols, order_by="_window_end"),
            )
        )
        .otherwise(
            _pchip_edge_slope(
                width=left_h,
                next_width=left_h.shift(1).over(group_cols, order_by="_window_end"),
                slope=left_d,
                next_slope=left_d.shift(1).over(group_cols, order_by="_window_end"),
            )
        )
        .fill_null(0.0)
    )
    windows = windows.with_columns(_derivative=derivative).with_columns(
        _earlier_derivative=pl.col("_derivative").shift(1).over(group_cols, order_by="_window_end"),
        _later_derivative=pl.col("_derivative").shift(-1).over(group_cols, order_by="_window_end"),
    )
    shaped = targets.join(
        windows.drop("_window_start", "_anchor_second", LONG_WAVE), on=window_keys, how="left"
    )
    offset = pl.col("valid_time").dt.epoch("s") - 900 - pl.col("_anchor_second")
    expressions = []
    for direction, label in ((-1, "earlier"), (1, "later")):
        width = pl.col(f"_{label}_width")
        fraction = offset.abs() / width
        curve = (
            (2 * fraction**3 - 3 * fraction**2 + 1) * pl.col(LONG_WAVE)
            + (fraction**3 - 2 * fraction**2 + fraction) * direction * width * pl.col("_derivative")
            + (-2 * fraction**3 + 3 * fraction**2) * pl.col(f"_{label}_value")
            + (fraction**3 - fraction**2) * direction * width * pl.col(f"_{label}_derivative")
        )
        expressions.append(pl.when(width.is_not_null()).then(curve).otherwise(pl.col(LONG_WAVE)))
    return shaped.with_columns(
        **{LONG_WAVE: pl.when(offset < 0).then(expressions[0]).otherwise(expressions[1])}
    )


def _resample_radiation(
    *, native: pl.LazyFrame, group_cols: list[str], solar_coordinates: pl.LazyFrame | None
) -> pl.LazyFrame:
    """Return half-hour radiation features over supported backward native windows.

    Args:
        native: Original native samples, with `nwp_init_time` and weather columns.
        group_cols: Identity columns separating independent weather trajectories.
        solar_coordinates: Unique latitude/longitude by time_series_id; required for short-wave.

    Returns:
        Group keys, target valid_time and reconstructed radiation, in the original numeric dtypes.
    """
    schema = native.collect_schema()
    radiation = sorted(RADIATION_COLUMNS.intersection(schema.names()))
    targets = _native_targets(
        native=native.select(*group_cols, "valid_time", *radiation), group_cols=group_cols
    )
    if LONG_WAVE in radiation:
        targets = _shape_long_wave(targets=targets, group_cols=group_cols)
    if SHORT_WAVE in radiation:
        if solar_coordinates is None:
            raise ValueError("Short-wave reconstruction requires solar_coordinates.")
        targets = _shape_short_wave(
            targets=targets, group_cols=group_cols, coordinates=solar_coordinates
        )
    return targets.select(*group_cols, "valid_time", *radiation).cast(
        {column: schema[column] for column in radiation}
    )
