"""Constants and loaders shared by the scripts of experiment 235.

Experiment 235 asks whether the capacity of the unmetered solar behind a Primary can be read from
the difference in net power between a sunny and a cloudy day of the same day type and temperature,
so that ordinary demand cancels. It adds a real farm's output at 50% of a Primary's pinned
capacity and measures how much of it the pair estimator finds. The plan is in
`docs/ml-experiments/00235-pv-capacity-matched-day-pairs/result.md`.

**Privacy.** The injected farms (labels A to F) are metered generators. Their output enters only as
a series scaled to a known capacity, and only ratios of recovered to injected capacity may leave the
private data folder.

**Hold-out.** Every fit trains on rows that end no later than `CUT`, 2026-07-01, which is the cut of
experiment 129's known-answer fits.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.dp_features import FeatureCache, default_input_paths, load_or_build_feature_cache
from studies.dp_gam import TrainingWindow
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 235
SLUG: Final[str] = "pv-capacity-matched-day-pairs"
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
AUDIT_LADDER_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00231-pv-capacity-audit-where-recovery-goes" / "work"
)
"""Experiment 231's ladder tables, whose capacities and rung 4 are the benchmark."""

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
PAIR_MIN_CLEAR_INDEX_GAP: Final[float] = 0.4
"""A sunny day's daylight clear-sky index exceeds the cloudy day's by at least this."""
PAIR_MAX_TEMPERATURE_GAP_K: Final[float] = 1.0
"""The two days' mean daylight temperatures differ by at most this."""
PAIR_MAX_DAY_GAP: Final[int] = 21
"""The two days are at most this many days apart."""
PARTNERS_PER_DAY: Final[int] = 3
"""Each sunny day takes at most this many cloudy partners, the smallest temperature gaps first."""
MIN_DAYLIGHT_ROWS: Final[int] = 10
"""A day needs at least this many daylight training half-hours to be paired."""
DAYLIGHT_CLEAR_SKY_W_M2: Final[float] = 10.0
"""A half-hour is daylight, for the day's clear-sky index, above this clear-sky irradiance."""
DAYLIGHT_UNIT_SHAPE: Final[float] = 0.01
"""A pair's half-hour counts when either day's unit shape exceeds this."""
MIN_PAIRS: Final[int] = 30
"""A Primary with fewer pairs is reported and left out of the count."""
CLOUD_LOAD_SHARE: Final[float] = 0.10
"""The day-long cloud-driven load's mean over daylight half-hours, as a share of the injected
capacity (experiment 129)."""
PLACEBO_MONTHS: Final[tuple[int, ...]] = (12, 1)
"""The calendar months of the placebo pairs."""
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


def farm_norm(*, cache: FeatureCache, farm: str) -> np.ndarray:
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

    Returns:
        The normalised power on the grid.
    """
    everything = pl.read_parquet(INJECTION_FRAME_PATH).filter(
        pl.col("site") == farm, pl.col("kind") == "farm"
    )
    frame = everything.filter(pl.col("zenith") < FARM_MAX_ZENITH_DEG)
    power = frame.filter(pl.col("power").is_not_null())
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


def cloud_load(*, cache: FeatureCache, site: str, capacity_points: float) -> np.ndarray:
    """Return a day-long cloud-driven load on the cache's grid, in points (experiment 129's load).

    The cloud response of a daylight half-hour is `1 - GHI / clear-sky GHI`, clipped to 0 to 1, at
    the substation's CAMS point. A day's response is its mean over the day's daylight half-hours.
    The load on a daylight half-hour is that day's response, scaled so that the load averages
    `CLOUD_LOAD_SHARE` of `capacity_points` over all daylight half-hours, and it is zero at night.

    Args:
        cache: The cache, for the grid.
        site: The substation's label.
        capacity_points: The injected capacity the load is a share of, in points.

    Returns:
        The load on the grid; zero where CAMS is missing.
    """
    ghi, clear = clear_sky_index(cache=cache, site=site)
    daylight = clear > DAYLIGHT_CLEAR_SKY_W_M2
    cloud = np.where(daylight, np.clip(1.0 - ghi / np.maximum(clear, 1e-9), 0, 1), 0)
    cloud = np.nan_to_num(cloud, nan=0.0)
    day = (cache.end_times - np.timedelta64(30, "m")).astype("datetime64[D]")
    frame = pl.DataFrame({"day": day, "cloud": cloud, "daylight": daylight})
    day_response = (
        frame.with_columns(response=pl.col("cloud").filter(pl.col("daylight")).mean().over("day"))[
            "response"
        ]
        .fill_null(0.0)
        .to_numpy()
    )
    response = np.where(daylight, day_response, 0.0)
    scale = CLOUD_LOAD_SHARE * capacity_points / response[daylight].mean()
    return response * scale
