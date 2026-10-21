"""Constants and loaders shared by the scripts of experiment 139.

Experiment 139 feeds the convex model with the Kalman filter (GAM-K) an as-of solar capacity in
place of experiment 45's pinned capacity. The plan is in
`docs/ml-experiments/00139-differentiable-physics-pv-as-of-capacity-in-gam-k/result.md`.

**Privacy.** The injected farms (labels A to F) are metered generators. Their output enters only as
a series scaled to a known capacity, and only ratios of recovered to injected capacity may leave
the private data folder.
"""

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.dp_features import (
    FeatureCache,
    default_input_paths,
    load_or_build_feature_cache,
)
from studies.dp_gam import GamSpec, TrainingWindow
from studies.pv_physics import PlantParameters, half_hour_sun_and_sky, plant_power_mw
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 139
SLUG: Final[str] = "differentiable-physics-pv-as-of-capacity-in-gam-k"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
"""Small result files (capacity tables, gate tables); private and never committed."""
CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
"""The feature cache of experiment 106 (data to 2026-07-01), shared with experiment 111."""
INJECTION_FRAME_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00039-pv-disaggregation-cams-injection" / "work" / "frame.parquet"
)
"""Experiment 39's frame, which holds the solar farms' power beside the CAMS-driven unit shape."""

DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
"""The five Primaries with an as-of capacity table (experiment 129's)."""
FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "F")
"""The farms with data from 2019-09; farm E starts in 2024-02 and is left out."""
SEASON_YEARS: Final[range] = range(2019, 2026)
"""Season-years (July to June, named by the July's year) with a capacity increment."""
ORIGINS: Final[tuple[datetime, ...]] = (
    datetime(2023, 7, 1, tzinfo=UTC),
    datetime(2024, 7, 1, tzinfo=UTC),
    datetime(2025, 7, 1, tzinfo=UTC),
)
"""The as-of origins: the scored year's and the replicate year's first instants, and the year
before, which serves the monthly cuts from 2024-04 to 2024-06 (they forecast no scored row)."""
TRAINING_WINDOW: Final[TrainingWindow] = TrainingWindow()
RIDGE: Final[float] = 0.1
"""Experiment 104's tuned ridge penalty, as experiments 106, 110, and 129 used it."""
MIN_YEAR_ROWS: Final[int] = 12000
"""A season-year counts as a data year at a Primary when it has at least this many training
half-hours (a full season-year has 17,520), experiment 129's rule."""
FARM_MAX_ZENITH_DEG: Final[float] = 75.0
FARM_DROPOUT_GHI_W_M2: Final[float] = 100.0
"""A farm half-hour at or below zero power while the irradiance exceeds this is a dropout."""
SUN_UP_UNIT_SHAPE: Final[float] = 0.01
"""A grid half-hour with no frame row counts as sunlit when a unit shape exceeds this."""

ARM2_SPEC: Final[GamSpec] = GamSpec(ridge=RIDGE, capacity="fitted")
"""Experiment 129's arm 2: the monotone season-year capacity."""


def load_cache() -> FeatureCache:
    """Return the feature cache that holds data to 2026-07-01."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)


def write_parquet_atomic(*, frame: pl.DataFrame, path: Path) -> None:
    """Write a parquet file through a temporary name, so a crash leaves no half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.write_parquet(temporary)
    temporary.rename(path)


def farm_norm(*, cache: FeatureCache, farm: str) -> np.ndarray:
    """Return a farm's power divided by its 99th percentile, on the cache's grid.

    Copied from experiment 129. The 99th percentile is over the farm's daylight half-hours in all
    history. A half-hour returns `NaN` (so the substation's power becomes `NaN` and the row drops
    out of training) when the farm's frame has no row for it while the sun is up, or has a daylight
    row with no reading or a dropout. Remaining negative readings are set to zero.

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


FARM_CAMS_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR.parent
    / "studies"
    / "weather"
    / "CAMS"
    / "site_points"
    / "beam_diffuse_cams.parquet"
)
"""Hourly CAMS at the six farms (global, beam, diffuse); `time` ends the hour."""
IRRADIANCE_FARM: Final[str] = "A"
"""The farm whose CAMS irradiance builds the synthetic plants, so that their irradiance error
differs from the Primary's own CAMS."""
EAST_WEST_PLANTS: Final[tuple[PlantParameters, ...]] = (
    PlantParameters(tilt_deg=15.0, azimuth_deg=90.0, dc_capacity_mw=0.65, ac_capacity_mw=0.5),
    PlantParameters(tilt_deg=15.0, azimuth_deg=270.0, dc_capacity_mw=0.65, ac_capacity_mw=0.5),
)
"""An east-west mix: two halves of one AC megawatt, DC:AC 1.3, clipped at the inverter."""
SOUTH_WEST_PLANTS: Final[tuple[PlantParameters, ...]] = (
    PlantParameters(tilt_deg=25.0, azimuth_deg=225.0, dc_capacity_mw=1.3, ac_capacity_mw=1.0),
)
"""A plant facing south-west at tilt 25 degrees, DC:AC 1.3, clipped at the inverter."""


def synthetic_plant_shape(
    *, cache: FeatureCache, site: str, plants: tuple[PlantParameters, ...]
) -> np.ndarray:
    """Return a synthetic plant's AC output per MW of AC capacity, on the cache's grid.

    Copied from experiment 129. The output uses the CAMS irradiance at the farm `IRRADIANCE_FARM`,
    not at the substation, so that the plant carries an irradiance error against the shape the fit
    assumes. The sun's position is the substation's. The plant has its own tilt, azimuth, DC:AC
    ratio, and inverter clip, and no temperature derating.

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
