"""Constants, injections and loaders shared by the scripts of experiment 129.

Experiment 129 asks whether the AC capacity of the unmetered solar behind a Primary substation can
be inferred offline, with an honest range, by adding known solar output to a Primary's real power
and checking what the convex model of `studies.dp_gam` finds. The plan is in
`docs/ml-experiments/00129-differentiable-physics-pv-capacity-known-answer-test/result.md`.

**Privacy.** The injected farms (labels A to F) are metered generators. Their output enters only as
a series scaled to a known capacity, and only ratios of recovered to injected capacity may leave
the private data folder.

**Hold-out.** The feature cache of experiment 106 stops at 2026-07-01, and `DATA_CUT` is that
instant. The maintainer approved fitting to 2026-08-31 for the final hindsight table, which needs
the feature cache rebuilt; nothing here reads data after `DATA_CUT`.
"""

import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import torch
from studies.dp_features import (
    FeatureCache,
    default_input_paths,
    load_or_build_feature_cache,
)
from studies.dp_gam import GamFit, GamSpec, TrainingWindow, fit_gam
from studies.dp_pinned_capacity import pinned_capacity, read_vintages
from studies.pv_physics import PlantParameters, half_hour_sun_and_sky, plant_power_mw
from studies.sources import ML_EXPERIMENTS_DATA_DIR

from studies import dp_features, dp_gam

EXPERIMENT_ID: Final[int] = 129
SLUG: Final[str] = "differentiable-physics-pv-capacity-known-answer-test"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
"""Small result files (pilot tables, gate tables, capacity tables); private and never committed."""
CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
"""The feature cache of experiment 106, read from there; it holds data to `DATA_CUT`."""
INJECTION_FRAME_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00039-pv-disaggregation-cams-injection" / "work" / "frame.parquet"
)
"""Experiment 39's frame, which holds the solar farms' power beside the CAMS-driven unit shape."""
FARM_CAMS_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR.parent
    / "studies"
    / "weather"
    / "CAMS"
    / "site_points"
    / "beam_diffuse_cams.parquet"
)
"""Hourly CAMS at the six farms (global, beam, diffuse); `time` ends the hour."""

DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
"""The five Primaries counted for the decision."""
PILOT_SITES: Final[tuple[str, ...]] = ("D10", "D16")
"""The Primaries of the stage B pilot: D10 has 6 labelled switching events, D16 has 14."""
FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "F")
"""The farms with data from 2019-09; farm E starts in 2024-02 and is left out."""
IRRADIANCE_FARM: Final[str] = "A"
"""The farm whose CAMS irradiance builds the synthetic plants, so that their irradiance error
differs from the Primary's own CAMS."""
SEASON_YEARS: Final[range] = range(2019, 2026)
"""Season-years (September to August, named by the September's year) with a capacity increment."""
COMPLETE_SEASON_YEARS: Final[range] = range(2019, 2025)
"""The season-years complete before `DATA_CUT`: 2019/20 to 2024/25."""
THROUGH_JULY: Final[bool] = os.environ.get("E00129_THROUGH_JULY") == "1"
"""Set `E00129_THROUGH_JULY=1` to fit to 2026-08-01 on the rows `e00129_july.py` builds. This is the
maintainer-approved hold-out exception: season-year 2025/26 ends in July because August's final
ERA5 is not out. It moves the two `HOLDOUT_START` constants and `GRID_END` in this process only."""
DATA_CUT: Final[datetime] = (
    datetime(2026, 8, 1, tzinfo=UTC) if THROUGH_JULY else datetime(2026, 7, 1, tzinfo=UTC)
)
"""The cut of every fit: the feature cache holds no later row."""
if THROUGH_JULY:
    dp_features.HOLDOUT_START = DATA_CUT
    dp_features.GRID_END = DATA_CUT - timedelta(minutes=30)
    dp_gam.HOLDOUT_START = DATA_CUT
TRAINING_WINDOW: Final[TrainingWindow] = TrainingWindow()
"""Every row from the substation's first usable instant to the cut."""
RIDGE: Final[float] = 0.1
"""Experiment 104's tuned ridge penalty, as experiments 106 and 110 used it."""
CLOUD_LOAD_SHARE: Final[float] = 0.10
"""The day-long cloud-driven load's mean over daylight half-hours, as a share of the injected
capacity."""
FARM_MAX_ZENITH_DEG: Final[float] = 75.0
FARM_DROPOUT_GHI_W_M2: Final[float] = 100.0
"""A farm half-hour at or below zero power while the irradiance exceeds this is a dropout."""
SUN_UP_UNIT_SHAPE: Final[float] = 0.01
"""A grid half-hour with no frame row counts as sunlit when a unit shape exceeds this."""
DAYLIGHT_CLEAR_SKY_W_M2: Final[float] = 10.0
"""A half-hour is daylight, for the cloud-driven load, above this clear-sky irradiance."""

EAST_WEST_PLANTS: Final[tuple[PlantParameters, ...]] = (
    PlantParameters(tilt_deg=15.0, azimuth_deg=90.0, dc_capacity_mw=0.65, ac_capacity_mw=0.5),
    PlantParameters(tilt_deg=15.0, azimuth_deg=270.0, dc_capacity_mw=0.65, ac_capacity_mw=0.5),
)
"""An east-west mix: two halves of one AC megawatt, DC:AC 1.3, clipped at the inverter."""
SOUTH_WEST_PLANTS: Final[tuple[PlantParameters, ...]] = (
    PlantParameters(tilt_deg=25.0, azimuth_deg=225.0, dc_capacity_mw=1.3, ac_capacity_mw=1.0),
)
"""A plant facing south-west at tilt 25 degrees, DC:AC 1.3, clipped at the inverter."""


def load_cache() -> FeatureCache:
    """Return the feature cache that holds data to `DATA_CUT`.

    Experiment 106's cache ends at 2026-07-01. Through July, the cache is rebuilt from
    `work/rows_to_july.parquet` in this experiment's own folder.
    """
    paths = default_input_paths()
    if not THROUGH_JULY:
        return load_or_build_feature_cache(paths=paths, cache_dir=CACHE_DIR)
    return load_or_build_feature_cache(
        paths=replace(paths, rows=WORK_DIR / "rows_to_july.parquet"),
        cache_dir=WORK_DIR / "cache_to_july",
    )


def fit_site(*, cache: FeatureCache, site: str, spec: GamSpec, device: str) -> GamFit:
    """Fit the convex model of one substation at `DATA_CUT` on every usable row."""
    return fit_gam(
        cache=cache,
        site=site,
        specs=[spec],
        cut=DATA_CUT,
        window=TRAINING_WINDOW,
        season_years=SEASON_YEARS,
        device=device,
        dtype=torch.float64,
    )


def pinned_points(*, cache: FeatureCache, site: str) -> float:
    """Return the Primary's pinned capacity at `DATA_CUT`, in points; it scales the injections."""
    return pinned_capacity(
        vintages=read_vintages(), table=cache.sites[site].table, cut=DATA_CUT
    ).held_points


def with_reading(*, cache: FeatureCache, site: str, y: np.ndarray) -> FeatureCache:
    """Return the cache with one substation's reading replaced."""
    return replace(cache, sites={**cache.sites, site: replace(cache.sites[site], y=y)})


def inject(
    *, cache: FeatureCache, site: str, output: np.ndarray, load: np.ndarray | None = None
) -> FeatureCache:
    """Subtract a solar output from a substation's power, and add a cloud-driven load.

    A `NaN` in `output` makes the power `NaN` there, so the row drops out of training.

    Args:
        cache: The cache.
        site: The substation's label.
        output: The solar output on the grid, in points of effective capacity.
        load: The load added to the power, in points, or `None` for none.

    Returns:
        The cache with the changed reading.
    """
    y = cache.sites[site].y - output
    if load is not None:
        y = y + load
    return with_reading(cache=cache, site=site, y=y)


def farm_norm(*, cache: FeatureCache, farm: str) -> np.ndarray:
    """Return a farm's power divided by its 99th percentile, on the cache's grid.

    The 99th percentile is over the farm's daylight half-hours in all history, as in experiment 39.
    A half-hour returns `NaN`, so a substation's power becomes `NaN` there and the row drops out of
    training instead of carrying a farm that produced nothing, when the farm's frame has no row for
    it while the sun is up, or has a daylight row (zenith under 75 degrees) with no reading or a
    dropout (power at or below zero while the irradiance exceeds `FARM_DROPOUT_GHI_W_M2`).
    Remaining negative readings are set to zero. A half-hour with the sun below 75 degrees of
    zenith returns 0, and so does a half-hour with no frame row while the sun is down.

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


def synthetic_plant_shape(
    *, cache: FeatureCache, site: str, plants: tuple[PlantParameters, ...]
) -> np.ndarray:
    """Return a synthetic plant's AC output per MW of AC capacity, on the cache's grid.

    The output uses the CAMS irradiance at the farm `IRRADIANCE_FARM`, not at the substation, so
    that the plant carries an irradiance error against the shape the fit assumes. The sun's
    position is the substation's. The plant has its own tilt, azimuth, DC:AC ratio, and inverter
    clip, and no temperature derating. Several plants are summed, so their AC capacities should
    add to 1 MW.

    Args:
        cache: The cache, for the grid.
        site: The substation's label, for the sun's position.
        plants: The plant, or the parts of a mix.

    Returns:
        The shape on the grid, `NaN` where the farm's CAMS is missing.
    """
    table = cache.sites[site].table
    hourly = (
        pl.read_parquet(FARM_CAMS_PATH)
        .filter(pl.col("site") == IRRADIANCE_FARM)
        .select("time", pl.col("ghi_w_m2").cast(pl.Float64))
        .drop_nulls()
        .sort("time")
    )
    sky = half_hour_sun_and_sky(hourly=hourly, latitude=table.latitude, longitude=table.longitude)
    total = sum(plant_power_mw(sky=sky, parameters=plant) for plant in plants)
    frame = pl.DataFrame(
        {
            "time": pl.Series(sky.half_hour_end_time).dt.replace_time_zone("UTC"),
            "shape": np.asarray(total, dtype=np.float64),
        }
    )
    grid = pl.DataFrame({"time": cache.times_series()})
    joined = grid.join(frame, on="time", how="left", validate="1:1", maintain_order="left")
    return joined["shape"].to_numpy().astype(np.float64)


def cloud_load(*, cache: FeatureCache, site: str, capacity_points: float) -> np.ndarray:
    """Return a day-long cloud-driven load on the cache's grid, in points.

    The cloud response of a daylight half-hour is `1 - GHI / clear-sky GHI`, clipped to 0 to 1, at
    the substation's CAMS point (experiment 40). A day's response is its mean over the day's
    daylight half-hours. The load on a daylight half-hour is that day's response, scaled so that
    the load averages `CLOUD_LOAD_SHARE` of `capacity_points` over all daylight half-hours, and it
    is zero at night. It is the demand that rises on overcast days, which looks like lost solar.

    Args:
        cache: The cache, for the grid.
        site: The substation's label.
        capacity_points: The injected capacity the load is a share of, in points.

    Returns:
        The load on the grid; zero where CAMS is missing.
    """
    sky = _half_hour_clear_sky(cache=cache, site=site)
    daylight = sky["clear"] > DAYLIGHT_CLEAR_SKY_W_M2
    cloud = np.where(daylight, np.clip(1.0 - sky["ghi"] / np.maximum(sky["clear"], 1e-9), 0, 1), 0)
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


def _half_hour_clear_sky(*, cache: FeatureCache, site: str) -> dict[str, np.ndarray]:
    """Return the substation's CAMS global and clear-sky irradiance on the cache's grid.

    Each hour-ending value is repeated on the two half-hours that end inside the hour.
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
    return {
        "ghi": joined["ghi_w_m2"].to_numpy().astype(np.float64),
        "clear": joined["clear_sky_ghi_w_m2"].to_numpy().astype(np.float64),
    }
