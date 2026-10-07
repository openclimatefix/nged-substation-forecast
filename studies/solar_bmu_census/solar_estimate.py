"""Estimate the solar part of a BMU's AC capacity from its output alone.

A Balancing Mechanism Unit (BMU) that pools many sites registers one capacity for all of them, and
no source splits that capacity by technology. This module fits the AC capacity `a` of a solar
component to the BMU's half-hourly output, under a model of solar output that is flat at the top:
`output(t) = min(r * a * c(t), a)`. Here `c(t)` is the cosine of the solar zenith at the census
reference point, scaled so that its peak over the window is 1, and `r` is the DC:AC ratio. Output
is flat at `a` whenever `r * c(t)` exceeds 1, because the inverters cannot export more than `a`.

Clouds only reduce output, so the fit uses the upper envelope of output and not its mean: a high
quantile of output within each band of `c(t)`. The estimate is an upper bound only if the BMU's
solar part is the whole of what drives the envelope. A BMU that also holds other generation can
give an estimate that is too high, and the estimate of a BMU with a large load can be too low.
"""

from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from classify import analysis_series, drop_daytime_zeros
from studies.sources import REANALYSIS_DOWNLOADS_DIR

CAMS_PUBLIC_POINTS_PATH: Final[Path] = (
    REANALYSIS_DOWNLOADS_DIR / "CAMS_public_points" / "cams_public_points.parquet"
)
"""Hourly CAMS irradiance at the single-site solar BMUs' positions and at 18 grid points. Columns:
`point_id`, `bmu_ids`, `latitude`, `longitude`, `time`, `ghi_w_m2`, `clear_sky_ghi_w_m2`, and
`reliability`. A point named `gb_NN` is a grid point and has no `bmu_ids`."""
MIN_CAMS_RELIABILITY: Final[float] = 0.9
"""Point-hours that CAMS flags below this fraction of reliable inputs are dropped."""
DC_AC_RATIO: Final[float] = 1.4
"""The DC:AC ratio `r` of the base case.

Lawrence Berkeley National Laboratory's Utility-Scale Solar report for 2023 (Bolinger, Seel and
others) gives a median of 1.40 for fixed-tilt projects built in the United States in 2022, and 1.32
for all projects. The study found no figure for Great Britain.
"""
SENSITIVITY_RATIOS: Final[tuple[float, ...]] = (1.2, DC_AC_RATIO, 1.6)
"""The DC:AC ratios the report tries."""
ENVELOPE_QUANTILE: Final[float] = 0.99
"""The quantile of output, within each band of `c(t)`, that stands for the upper envelope."""
BAND_EDGES: Final[tuple[float, ...]] = tuple(round(0.1 * k, 1) for k in range(1, 11))
"""Edges of the nine bands of `c(t)`, from 0.1 to 1.0. Below 0.1 the sun is in twilight."""
MIN_HALF_HOURS_PER_BAND: Final[int] = 20
"""A band with fewer half-hours than this is left out of the fit."""


def fit_ac_capacity_mw(
    *,
    sun: np.ndarray,
    output_mw: np.ndarray,
    dc_ac_ratio: float,
    quantile: float = ENVELOPE_QUANTILE,
) -> float | None:
    """Fit the AC capacity `a` of a flat-topped solar model to the upper envelope of output.

    In each band of `sun`, the envelope is the `quantile` of output, and the model's value there is
    `min(r * a * c, a)`, with `c` the same `quantile` of `sun` in the band. The model is linear in
    `a`, so the least-squares fit over the bands has a closed form.

    Args:
        sun: `c(t)` for each half-hour, scaled so that its peak is 1.
        output_mw: The BMU's output in each half-hour, in megawatts.
        dc_ac_ratio: The DC:AC ratio `r`.
        quantile: The quantile that stands for the upper envelope.

    Returns:
        The fitted `a` in megawatts, or None when no band holds `MIN_HALF_HOURS_PER_BAND`
        half-hours or the fit is not above zero.
    """
    shape: list[float] = []
    envelope: list[float] = []
    for low, high in pairwise(BAND_EDGES):
        in_band = (sun >= low) & ((sun < high) | (high == BAND_EDGES[-1]))
        if in_band.sum() < MIN_HALF_HOURS_PER_BAND:
            continue
        shape.append(min(dc_ac_ratio * float(np.quantile(sun[in_band], quantile)), 1.0))
        envelope.append(float(np.quantile(output_mw[in_band], quantile)))
    if not shape:
        return None
    shape_array, envelope_array = np.array(shape), np.array(envelope)
    fitted = float(shape_array @ envelope_array / (shape_array @ shape_array))
    return fitted if fitted > 0 else None


def estimate_solar_ac_capacity_mw(
    *, output: pl.DataFrame, window_start: datetime, dc_ac_ratio: float
) -> float | None:
    """Estimate a BMU's solar AC capacity in megawatts from its settled half-hourly output.

    The half-hours are those the classifier judges: after the commissioning period, and without the
    exact zeros while the sun is clearly up. The sun's cosine is scaled so that its peak over those
    half-hours is 1.

    Args:
        output: Columns `half_hour_end_time` (UTC) and `output_mwh`.
        window_start: The start of the study window, in UTC.
        dc_ac_ratio: The DC:AC ratio `r`.

    Returns:
        The estimate, or None when the BMU has too little output to fit.
    """
    judged = drop_daytime_zeros(series=analysis_series(output=output, window_start=window_start))
    if judged.is_empty():
        return None
    sun = judged["cos_zenith"].to_numpy()
    peak = float(sun.max())
    if peak <= 0:
        return None
    return fit_ac_capacity_mw(
        sun=sun / peak, output_mw=judged["output_mwh"].to_numpy() * 2, dc_ac_ratio=dc_ac_ratio
    )


MIN_SUN_FOR_FIT: Final[float] = 0.05
"""The least-squares fit with a measured irradiance uses only half-hours with a scaled irradiance
above this, so that the many night-time half-hours of zero do not dominate."""
FIT_GRID_POINTS: Final[int] = 3000
"""The number of candidate capacities the least-squares fit tries between zero and 1.5 times the
largest output."""


def mean_hourly_sun(*, cams: pl.DataFrame, min_reliability: float) -> pl.DataFrame:
    """Average the CAMS irradiance over some points, and scale it to the clear-sky peak.

    Args:
        cams: Columns `time` (UTC, the end of the hour), `ghi_w_m2`, `clear_sky_ghi_w_m2`, and
            `reliability`, for one point or for several.
        min_reliability: Point-hours flagged below this fraction are dropped before averaging.

    Returns:
        Columns `hour_end_time` and `sun`: the mean of the reliable points' irradiance in each hour,
        divided by the highest mean clear-sky irradiance of any hour. An hour with no reliable
        point is absent.
    """
    hourly = (
        cams.filter(pl.col("reliability") >= min_reliability)
        .group_by("time")
        .agg(pl.col("ghi_w_m2").mean(), pl.col("clear_sky_ghi_w_m2").mean())
        .sort("time")
    )
    peak = float(hourly["clear_sky_ghi_w_m2"].max())  # ty: ignore[invalid-argument-type]
    return hourly.select(hour_end_time=pl.col("time"), sun=pl.col("ghi_w_m2") / peak)


def fit_ac_capacity_least_squares_mw(
    *, sun: np.ndarray, output_mw: np.ndarray, dc_ac_ratio: float
) -> float | None:
    """Fit `a` in `min(r * a * c, a)` to output by least squares, with a measured shape `c`.

    Args:
        sun: The scaled irradiance `c` in each half-hour, including its clouds.
        output_mw: The BMU's output in each half-hour, in megawatts.
        dc_ac_ratio: The DC:AC ratio `r`.

    Returns:
        The capacity in megawatts that minimises the sum of squared errors over the half-hours with
        `sun` above `MIN_SUN_FOR_FIT`, on a grid of `FIT_GRID_POINTS` values between zero and 1.5
        times the largest output; None when no such half-hour exists or the best capacity is zero.
    """
    daytime = sun > MIN_SUN_FOR_FIT
    if not daytime.any() or float(output_mw[daytime].max()) <= 0:
        return None
    shape, observed = sun[daytime], output_mw[daytime]
    candidates = np.linspace(0.0, 1.5 * float(observed.max()), FIT_GRID_POINTS)
    squared_error = np.array(
        [
            float(np.sum((np.minimum(dc_ac_ratio * a * shape, a) - observed) ** 2))
            for a in candidates
        ]
    )
    best = float(candidates[int(np.argmin(squared_error))])
    return best if best > 0 else None


def estimate_solar_ac_capacity_with_cams_mw(
    *,
    output: pl.DataFrame,
    window_start: datetime,
    dc_ac_ratio: float,
    hourly_sun: pl.DataFrame,
) -> float | None:
    """Estimate a BMU's solar AC capacity using the regional CAMS irradiance as the shape.

    The hour's mean applies to both half-hours inside the hour. A half-hour ending at 10:30 or at
    11:00 lies in the hour that ends at 11:00, so the half-hour's end time is rounded up to the
    hour. Half-hours in an hour with no CAMS value are left out.

    Args:
        output: Columns `half_hour_end_time` (UTC) and `output_mwh`.
        window_start: The start of the study window, in UTC.
        dc_ac_ratio: The DC:AC ratio `r`.
        hourly_sun: `mean_hourly_sun`'s output.

    Returns:
        The estimate, or None when the BMU has too little output to fit.
    """
    judged = drop_daytime_zeros(series=analysis_series(output=output, window_start=window_start))
    joined = (
        judged.with_columns(
            hour_end_time=pl.col("half_hour_end_time")
            .dt.offset_by("-1us")
            .dt.truncate("1h")
            .dt.offset_by("1h")
        )
        .join(hourly_sun, on="hour_end_time", how="inner")
        .drop_nulls("sun")
    )
    if joined.is_empty():
        return None
    return fit_ac_capacity_least_squares_mw(
        sun=joined["sun"].to_numpy(),
        output_mw=joined["output_mwh"].to_numpy() * 2,
        dc_ac_ratio=dc_ac_ratio,
    )
