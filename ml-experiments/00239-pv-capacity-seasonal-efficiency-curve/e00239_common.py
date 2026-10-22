"""Constants and loaders shared by the scripts of experiment 239.

Experiment 239 multiplies the satellite-based solar shape of experiment 129's capacity fit by a
seasonal efficiency curve of 12 monthly factors fitted on the metered farms' own output, then
repeats experiment 231's injection test. The plan is in
`docs/ml-experiments/00239-pv-capacity-seasonal-efficiency-curve/result.md`.

**Privacy.** The farms (labels A to F) are metered generators. Their output enters only as a
series scaled to a known capacity, and only ratios of recovered to injected capacity and the
12 shared factors may leave the private data folder.

**Hold-out.** Every fit trains on rows that end no later than `CUT`, 2026-07-01, which is the cut of
experiment 129's known-answer fits.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.dp_features import FeatureCache, default_input_paths, load_or_build_feature_cache
from studies.dp_gam import (
    CELL_TEMPERATURE_RISE_K,
    DERATING_PER_KELVIN,
    REFERENCE_CELL_TEMPERATURE_C,
    TrainingWindow,
)
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 239
SLUG: Final[str] = "pv-capacity-seasonal-efficiency-curve"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
"""Small result files; private and never committed."""
CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
"""The feature cache of experiment 106, read from there; it holds data to 2026-07-01."""
INJECTION_FRAME_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00039-pv-disaggregation-cams-injection" / "work" / "frame.parquet"
)
"""Experiment 39's frame, which holds the solar farms' power beside the CAMS-driven unit shape."""
KNOWN_ANSWER_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00129-differentiable-physics-pv-capacity-known-answer-test"
    / "work"
    / "kat_arm2.parquet"
)
"""Experiment 129's arm 2 known-answer table, which rung 4 must reproduce."""

CUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
"""Every fit uses rows that end no later than this instant."""
READ_OUT_SEASON_YEAR: Final[int] = 2024
"""The season-year (July 2024 to June 2025) at which the recovery is read: the last complete
season-year in experiment 129's known-answer table."""
DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
"""The five Primaries counted for the decision."""
ALL_SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
"""The decision Primaries and D9 beside them."""
FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "F")
"""The farms with data from 2019-09."""
INJECTED_SHARE: Final[float] = 0.5
"""The injected capacity, as a share of the Primary's pinned capacity."""
SEASON_YEARS: Final[range] = range(2019, 2026)
"""Season-years (July to June, named by the July's year) with a capacity increment."""
READ_OUT_COLUMNS: Final[int] = READ_OUT_SEASON_YEAR - SEASON_YEARS[0] + 1
"""The leading capacity increments whose sum is the capacity at the read-out season-year."""
RIDGE: Final[float] = 0.1
"""Experiment 104's tuned ridge penalty, as experiments 106, 129 and 157 used it."""
N_RESAMPLES: Final[int] = 200
"""Month-block bootstrap resamples."""
SUMMER_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8)
"""April to August, for the summer-rows-only fit."""
WINTER_MONTHS: Final[tuple[int, ...]] = (11, 12, 1, 2)
"""November to February, for the winter-rows-only fit."""
SEASONS: Final[dict[str, tuple[int, ...]]] = {
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
    "winter": (12, 1, 2),
}
"""Calendar months of each season, for the divisor check."""
CLEAR_INDEX: Final[float] = 0.9
"""A half-hour is clear when the Primary's CAMS global irradiance is at least this share of its
clear-sky value."""
CLEAR_SKY_MIN_W_M2: Final[float] = 200.0
"""... and the clear-sky irradiance is at least this."""
DAYLIGHT_UNIT_SHAPE: Final[float] = 0.01
"""A half-hour is daylight for the divisor check when the unit shape exceeds this."""
FARM_MAX_ZENITH_DEG: Final[float] = 75.0
FARM_DROPOUT_GHI_W_M2: Final[float] = 100.0
"""A farm half-hour at or below zero power while the irradiance exceeds this is a dropout."""
SUN_UP_UNIT_SHAPE: Final[float] = 0.01
"""A grid half-hour with no frame row counts as sunlit when a unit shape exceeds this."""
TRAINING_WINDOW: Final[TrainingWindow] = TrainingWindow()
"""Every row from the substation's first usable instant to the cut."""


def load_cache() -> FeatureCache:
    """Return the feature cache of experiment 106, which holds data to 2026-07-01."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)


def farm_norm(
    *, cache: FeatureCache, farm: str, max_zenith_deg: float = FARM_MAX_ZENITH_DEG
) -> np.ndarray:
    """Return a farm's power divided by its 99th percentile, on the cache's grid.

    The 99th percentile is over the farm's daylight half-hours in all history, as in experiments 39
    and 129. A half-hour returns `NaN` when the farm's frame has no row for it while the sun is up,
    or has a daylight row (zenith under 75 degrees) with no reading or a dropout (power at or below
    zero while the irradiance exceeds `FARM_DROPOUT_GHI_W_M2`), so that the row drops out of
    training. Remaining negative readings are set to zero. A half-hour with the sun below 75 degrees
    of zenith returns 0, and so does a half-hour with no frame row while the sun is down.

    Args:
        cache: The cache, for the grid.
        farm: The farm label (A to F).
        max_zenith_deg: Half-hours with the sun lower than this (larger zenith) return 0. The
            default is experiment 129's; 90 keeps every half-hour with the sun up. The 99th
            percentile is always over half-hours under `FARM_MAX_ZENITH_DEG`, so the unit does
            not change.

    Returns:
        The normalised power on the grid.
    """
    everything = pl.read_parquet(INJECTION_FRAME_PATH).filter(
        pl.col("site") == farm, pl.col("kind") == "farm"
    )
    frame = everything.filter(pl.col("zenith") < max_zenith_deg)
    power = everything.filter(pl.col("zenith") < FARM_MAX_ZENITH_DEG, pl.col("power").is_not_null())
    p99 = float(np.nanpercentile(power["power"].to_numpy(), 99))
    dropout = (pl.col("power") <= 0.0) & (pl.col("ghi_u") > FARM_DROPOUT_GHI_W_M2)
    daylight = frame.select(
        "time",
        norm=pl.when(pl.col("power").is_null() | dropout)
        .then(float("nan"))
        .otherwise(pl.col("power").clip(lower_bound=0.0) / p99),
    )
    grid = pl.DataFrame({"time": cache.times_series()})
    joined = grid.join(daylight, on="time", how="left", validate="1:1", maintain_order="left")
    norm = joined["norm"].fill_null(0.0).to_numpy().astype(np.float64)
    in_frame = grid.join(everything.select("time"), on="time", how="semi")["time"]
    missing = ~np.isin(cache.times_series().to_numpy(), in_frame.to_numpy())
    sun_up = np.max(
        [np.nan_to_num(site.pv_unit_shape, nan=0.0) for site in cache.sites.values()], axis=0
    )
    norm[missing & (sun_up > SUN_UP_UNIT_SHAPE)] = np.nan
    return norm


def clear_sky_index(*, cache: FeatureCache, site: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the Primary's CAMS global irradiance and clear-sky irradiance on the cache's grid.

    Each hour-ending CAMS value is repeated on the two half-hours that end inside the hour, as in
    experiment 129. A half-hour with no CAMS row returns `NaN`.

    Args:
        cache: The cache.
        site: The substation's label.

    Returns:
        `(ghi, clear_sky_ghi)` in W/m2.
    """
    table = cache.sites[site].table
    cams = (
        pl.read_parquet(default_input_paths().cams)
        .filter(pl.col("site") == table.site)
        .select("time", "ghi_w_m2", "clear_sky_ghi_w_m2")
    )
    grid = pl.DataFrame({"time": cache.times_series()}).with_columns(
        hour_end=(pl.col("time") - pl.duration(microseconds=1)).dt.truncate("1h")
        + pl.duration(hours=1)
    )
    joined = grid.join(
        cams.rename({"time": "hour_end"}), on="hour_end", how="left", maintain_order="left"
    )
    return (
        joined["ghi_w_m2"].to_numpy().astype(np.float64),
        joined["clear_sky_ghi_w_m2"].to_numpy().astype(np.float64),
    )


DERATING_ITERATIONS: Final[int] = 12
"""Fixed-point steps that divide the temperature derating out of a derated shape."""
DERATING_TOLERANCE: Final[float] = 1e-6
"""The largest allowed gap between a re-derated shape and its target."""
FULL_SUN_ZENITH_DEG: Final[float] = 90.0
"""The zenith limit of the full-sun injection: every half-hour with the sun above the horizon."""
MIN_DAYLIGHT_SHAPE: Final[float] = 0.01
"""A half-hour is daylight for the curve fit when the farm's derated shape exceeds this."""
FLAT_CURVE_SPREAD: Final[float] = 0.03
"""A curve whose largest and smallest factors differ by less than this share is flat."""
AUDIT_LADDER_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00231-pv-capacity-audit-where-recovery-goes" / "work"
)
"""Experiment 231's result files, which the reproduction check reads."""


def derate(*, unit: np.ndarray, temperature_c: np.ndarray) -> np.ndarray:
    """Return the unit shape times its temperature derating, as `studies.dp_gam` computes it."""
    cell = temperature_c + CELL_TEMPERATURE_RISE_K * unit
    return unit * np.clip(
        1.0 - DERATING_PER_KELVIN * (cell - REFERENCE_CELL_TEMPERATURE_C), 0, None
    )


def invert_derating(*, derated: np.ndarray, temperature_c: np.ndarray) -> np.ndarray:
    """Return the unit shape whose derated value is `derated`, by fixed-point steps.

    Args:
        derated: The wanted derated shape; `NaN` stays `NaN`.
        temperature_c: The air temperature in degrees Celsius.

    Returns:
        The unit shape `u` with `u * (1 - 0.004 * (T + 25 u - 25)) = derated`.
    """
    unit = derated.copy()
    for _ in range(DERATING_ITERATIONS):
        cell = temperature_c + CELL_TEMPERATURE_RISE_K * unit
        factor = np.clip(
            1.0 - DERATING_PER_KELVIN * (cell - REFERENCE_CELL_TEMPERATURE_C), 0.2, None
        )
        unit = derated / factor
    return unit


def calendar_months(*, cache: FeatureCache) -> np.ndarray:
    """Return each grid half-hour's calendar month (1 to 12), from the half-hour's start."""
    start = cache.end_times - np.timedelta64(30, "m")
    return start.astype("datetime64[M]").astype(np.int64) % 12 + 1


def farm_own_shape(*, cache: FeatureCache, farm: str) -> np.ndarray:
    """Return the farm's temperature-derated satellite shape at its own site, on the grid.

    The frame holds no temperature for the farms, so the derating uses the mean temperature of the
    six Primaries' sites, which lie within about 50 km of the farms. A half-hour with no frame row
    returns `NaN`.
    """
    frame = pl.read_parquet(INJECTION_FRAME_PATH).filter(
        pl.col("site") == farm, pl.col("kind") == "farm"
    )
    grid = pl.DataFrame({"time": cache.times_series()})
    joined = grid.join(
        frame.select("time", "u"),
        on="time",
        how="left",
        validate="1:1",
        maintain_order="left",
    )
    unit = joined["u"].to_numpy().astype(np.float64)
    temperature = np.nanmean([site.temperature_c for site in cache.sites.values()], axis=0)
    return derate(unit=unit, temperature_c=temperature)


def monthly_scales(*, cache: FeatureCache, farm: str) -> np.ndarray:
    """Return a farm's least-squares scale of its output on its own derated shape, per month.

    The scale of calendar month `k` is `sum(x v) / sum(v^2)` over the farm's daylight half-hours
    that end no later than `CUT`, for the farm's normalised output `x` and derated shape `v`.

    Args:
        cache: The cache, for the grid.
        farm: The farm label.

    Returns:
        12 scales, January first.
    """
    norm = farm_norm(cache=cache, farm=farm, max_zenith_deg=FULL_SUN_ZENITH_DEG)
    shape = farm_own_shape(cache=cache, farm=farm)
    usable = (
        (cache.end_times <= np.datetime64(CUT.replace(tzinfo=None)))
        & np.isfinite(norm)
        & np.isfinite(shape)
        & (shape > MIN_DAYLIGHT_SHAPE)
    )
    month = calendar_months(cache=cache)
    cross = np.bincount(month[usable], weights=(norm * shape)[usable], minlength=13)[1:]
    square = np.bincount(month[usable], weights=(shape**2)[usable], minlength=13)[1:]
    return cross / square


def curve_excluding(*, cache: FeatureCache, injected: str) -> np.ndarray:
    """Return the 12 monthly factors fitted on every farm in `FARMS` except `injected`.

    The factors are the mean over those farms of their monthly scales, divided by the mean of the
    12 means so that the curve has mean 1.
    """
    scales = np.stack(
        [monthly_scales(cache=cache, farm=farm) for farm in FARMS if farm != injected]
    )
    curve = scales.mean(axis=0)
    return curve / curve.mean()


def apply_curve(*, cache: FeatureCache, site: str, curve: np.ndarray) -> np.ndarray:
    """Return the unit shape whose derated value is the satellite shape times the curve."""
    features = cache.sites[site]
    unit = features.pv_unit_shape
    target = (
        derate(unit=unit, temperature_c=features.temperature_c)
        * curve[calendar_months(cache=cache) - 1]
    )
    return invert_derating(derated=target, temperature_c=features.temperature_c)
