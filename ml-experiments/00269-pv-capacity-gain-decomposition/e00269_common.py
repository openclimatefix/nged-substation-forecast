"""Constants and loaders shared by the scripts of experiment 269.

Experiment 269 replaces the satellite-based solar shape of experiment 129's capacity fit by the
recorded output of a different metered farm near the Primary, then repeats experiment 231's
injection
test. The plan is in
`docs/ml-experiments/00269-pv-capacity-gain-decomposition/result.md`.

**Privacy.** The farms (labels A to F) are metered generators. Their output enters only as a
series scaled to a known capacity, and only ratios of recovered to injected capacity and rounded
distances may leave the private data folder.

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
from studies.pv_dataset import pv_sites
from studies.pv_plant_fit import ResponseGrid, ResponseTable, fit_response_table
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 269
SLUG: Final[str] = "pv-capacity-gain-decomposition"
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


MAX_DISTANCE_KM: Final[float] = 30.0
"""A reference farm must lie within this distance of the Primary and of the injected farm."""
MIN_ELIGIBLE_FARMS: Final[int] = 3
"""A Primary with fewer eligible injected farms is not testable."""
EARTH_RADIUS_KM: Final[float] = 6371.0
DERATING_ITERATIONS: Final[int] = 12
"""Fixed-point steps that divide the temperature derating out of a measured farm output."""


def distance_km(*, lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    """Return the great-circle distance in kilometres between two points in degrees."""
    to_rad = np.pi / 180.0
    half = (
        np.sin((lat_b - lat_a) * to_rad / 2) ** 2
        + np.cos(lat_a * to_rad)
        * np.cos(lat_b * to_rad)
        * np.sin((lon_b - lon_a) * to_rad / 2) ** 2
    )
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(half)))


def reference_farm(*, cache: FeatureCache, site: str, injected: str) -> tuple[str, float] | None:
    """Return the reference farm for an injection and its distance to the Primary, in km.

    The reference is the Primary's nearest farm among `FARMS` other than the injected farm, and
    it must lie within `MAX_DISTANCE_KM` of the Primary and of the injected farm.

    Args:
        cache: The cache, for the Primary's coordinates.
        site: The Primary's label.
        injected: The injected farm's label.

    Returns:
        `(reference label, distance to the Primary in km)`, or `None` when no farm is eligible.
    """
    table = cache.sites[site].table
    farms = {row["site"]: row for row in pv_sites().to_dicts()}
    injected_row = farms[injected]
    candidates = []
    for label in FARMS:
        if label == injected:
            continue
        row = farms[label]
        to_primary = distance_km(
            lat_a=table.latitude,
            lon_a=table.longitude,
            lat_b=row["latitude"],
            lon_b=row["longitude"],
        )
        to_injected = distance_km(
            lat_a=injected_row["latitude"],
            lon_a=injected_row["longitude"],
            lat_b=row["latitude"],
            lon_b=row["longitude"],
        )
        if to_primary <= MAX_DISTANCE_KM and to_injected <= MAX_DISTANCE_KM:
            candidates.append((to_primary, label))
    if not candidates:
        return None
    distance, label = min(candidates)
    return label, distance


def reference_unit_shape(*, cache: FeatureCache, site: str, reference: str) -> np.ndarray:
    """Return the Primary's unit shape built from a reference farm's measured output.

    The farm's normalised output already carries its own heat loss, and the fit multiplies the unit
    shape by a temperature derating, so the derating is divided out first: `u` solves
    `u * (1 - 0.004 * (T + 25 u - 25)) = r` for the farm's normalised output `r` and the Primary's
    temperature `T`, by fixed-point steps. Half-hours at which the reference farm has a dropout or
    no reading stay `NaN`, so that the fit drops them.

    Args:
        cache: The cache, for the grid and the Primary's temperature.
        site: The Primary's label.
        reference: The reference farm's label.

    Returns:
        The shape on the cache's grid.
    """
    measured = farm_norm(cache=cache, farm=reference)
    temperature = cache.sites[site].temperature_c
    unit = measured.copy()
    for _ in range(DERATING_ITERATIONS):
        cell = temperature + CELL_TEMPERATURE_RISE_K * unit
        factor = np.clip(
            1.0 - DERATING_PER_KELVIN * (cell - REFERENCE_CELL_TEMPERATURE_C), 0.2, None
        )
        unit = measured / factor
    return unit


DERATING_TOLERANCE: Final[float] = 1e-6
"""The largest allowed gap between a derated reference shape and the farm's normalised output."""
AUDIT_LADDER_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00231-pv-capacity-audit-where-recovery-goes" / "work"
)
"""Experiment 231's result files, which the reproduction check reads."""


ALL_FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F")
"""Every metered farm; the pooled empirical shape averages the five other than the injected one."""
RESPONSE_GRID: Final[ResponseGrid] = ResponseGrid(
    irradiance_w_m2=np.arange(0.0, 1101.0, 50.0), elevation_deg=np.arange(5.0, 66.0, 5.0)
)
"""The nodes of the learned response table, as in experiment 218."""
RESPONSE_MIN_ELEVATION_DEG: Final[float] = 0.0
"""Half-hours with the sun at or below this elevation have a shape of zero."""


def farm_sky(*, cache: FeatureCache, farm: str) -> tuple[np.ndarray, np.ndarray]:
    """Return a farm's CAMS global irradiance and sun elevation on the cache's grid.

    Both come from the farm's row of experiment 39's frame. A half-hour with no row returns `NaN`.

    Args:
        cache: The cache, for the grid.
        farm: The farm label (A to F).

    Returns:
        `(ghi in W/m2, elevation in degrees)`.
    """
    frame = (
        pl.read_parquet(INJECTION_FRAME_PATH)
        .filter(pl.col("site") == farm, pl.col("kind") == "farm")
        .select("time", "ghi_u", elevation=90.0 - pl.col("zenith"))
    )
    grid = pl.DataFrame({"time": cache.times_series()})
    joined = grid.join(frame, on="time", how="left", validate="1:1", maintain_order="left")
    return (
        joined["ghi_u"].to_numpy().astype(np.float64),
        joined["elevation"].to_numpy().astype(np.float64),
    )


def learned_response(*, cache: FeatureCache, site: str, farm: str) -> ResponseTable:
    """Fit a farm's learned response table: its unit shape against its own irradiance and sun.

    The shape is the farm's output with the temperature derating divided out at the Primary's
    temperature (`reference_unit_shape`), so that the fit's derating is not applied a second time.
    The table is monotone in irradiance, smooth, and anchored to nothing (`penalty` 0), as in
    experiment 218's arm 3.

    Args:
        cache: The cache.
        site: The Primary's label, for the temperature.
        farm: The farm label (A to F).

    Returns:
        The fitted table.
    """
    unit = reference_unit_shape(cache=cache, site=site, reference=farm)
    ghi, elevation = farm_sky(cache=cache, farm=farm)
    rows = np.isfinite(unit) & np.isfinite(ghi) & (elevation > RESPONSE_MIN_ELEVATION_DEG)
    return fit_response_table(
        grid=RESPONSE_GRID,
        irradiance_w_m2=ghi[rows],
        elevation_deg=elevation[rows],
        output=unit[rows],
        prior_output=np.zeros(int(rows.sum())),
        penalty=0.0,
    )


def learned_unit_shape(*, cache: FeatureCache, site: str, farms: tuple[str, ...]) -> np.ndarray:
    """Return the mean of farms' learned responses, read at the Primary's irradiance and sun.

    Half-hours with no CAMS row at the Primary return `NaN`, and half-hours with the sun at or
    below the horizon return zero.

    Args:
        cache: The cache.
        site: The Primary's label.
        farms: The farms whose responses are averaged.

    Returns:
        The shape on the cache's grid.
    """
    features = cache.sites[site]
    shapes = [
        learned_response(cache=cache, site=site, farm=farm).predict(
            irradiance_w_m2=np.nan_to_num(features.ghi_w_m2), elevation_deg=features.elevation_deg
        )
        for farm in farms
    ]
    shape = np.mean(shapes, axis=0)
    shape[features.elevation_deg <= RESPONSE_MIN_ELEVATION_DEG] = 0.0
    shape[~np.isfinite(features.ghi_w_m2)] = np.nan
    return shape
