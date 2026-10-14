"""Paths, series, arms, and helpers shared by the scripts of experiment 00012.

The experiment tests which inputs and which training start give the lowest error for the stage 1a
XGBoost model of the 6 solar farms and 3 wind farms, applied to ENS members
(`docs/ml-experiments/00012-solar-wind-stage1a-inputs-cams-and-training-start/result.md`). The
inputs it reads were built by the stage 1 pilot and are read only.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

import numpy as np
import polars as pl
import pvlib
from studies.anonymise import LABEL_PERMUTATION_SEED, SITE_LABELS
from studies.power import scan_power
from studies.pv_dataset import pv_sites, wind_sites
from studies.sources import ML_EXPERIMENTS_DATA_DIR, PER_STUDY_DIR

EXPERIMENT_ID: Final[int] = 12
SLUG: Final[str] = "solar-wind-stage1a-inputs-cams-and-training-start"
ROW_SET_ID: Final[str] = "fold_a_tuesdays_v1"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)

TechnologyType = Literal["solar", "wind"]
ArmType = Literal[
    "w",
    "start_2023",
    "start_2024",
    "small_block",
    "speed_block",
    "csr",
    "base_margin",
    "cams",
    "cams_stage0",
]
SOLAR_ARMS: Final[tuple[ArmType, ...]] = (
    "w",
    "start_2023",
    "start_2024",
    "small_block",
    "csr",
    "cams",
    "cams_stage0",
)
WIND_ARMS: Final[tuple[ArmType, ...]] = (
    "w",
    "start_2023",
    "start_2024",
    "speed_block",
    "base_margin",
)

PILOT_DIR: Final[Path] = PER_STUDY_DIR / "stage1_pilot"
"""The stage 1 pilot's outputs (experiment 00001), read only."""
ANALYSIS_PATH: Final[Path] = PILOT_DIR / "inputs" / "analysis_features.parquet"
ENS_TRAIN_PATH: Final[Path] = PILOT_DIR / "inputs" / "ens_features" / "train.parquet"
ENS_VALIDATION_DIR: Final[Path] = PILOT_DIR / "inputs" / "ens_features" / "val"
PILOT_SCORES_DIR: Final[Path] = PILOT_DIR / "scores"
FRAMES_DIR: Final[Path] = (
    PER_STUDY_DIR / "quick_wins_fixed_pipeline_1156" / "frames" / "combined" / "fold_a"
)
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables; private and never committed."""

ERA5_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
TRAIN_END: Final[datetime] = datetime(2025, 6, 17, tzinfo=UTC)
"""Training uses valid times before this: 14 days before the first validation run."""
VALIDATION_START: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
HOLDOUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
STARTS: Final[dict[str, datetime]] = {
    "start_2023": datetime(2023, 9, 1, tzinfo=UTC),
    "start_2024": datetime(2024, 4, 1, tzinfo=UTC),
}
"""The training start of the two start arms; every other arm starts at `ERA5_START`."""

SOLAR_FEATURES: Final[tuple[str, ...]] = (
    "shortwave",
    "longwave",
    "temperature_2m",
    "dew_point_temperature_2m",
    "pressure_surface",
    "precipitation",
    "wind_speed_10m",
    "solar_zenith",
    "solar_azimuth",
)
WIND_FEATURES: Final[tuple[str, ...]] = (
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_dir_sin_10m",
    "wind_dir_cos_10m",
    "wind_dir_sin_100m",
    "wind_dir_cos_100m",
)
SMALL_SOLAR_FEATURES: Final[tuple[str, ...]] = (
    "shortwave",
    "temperature_2m",
    "solar_zenith",
    "solar_azimuth",
)
SPEED_WIND_FEATURES: Final[tuple[str, ...]] = (
    "wind_speed_100m",
    "wind_dir_sin_100m",
    "wind_dir_cos_100m",
)
STAGE0_FEATURES: Final[tuple[str, ...]] = (
    "shortwave",
    "longwave",
    "temperature_2m",
    "dew_point_temperature_2m",
    "solar_zenith",
    "solar_azimuth",
    "lead_hours",
)
"""Inputs of the stage 0 model, all served by ENS at forecast time."""

XGBOOST_PARAMETERS: Final[dict[str, float | int | str]] = {
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "device": "cuda",
    "nthread": 10,
}
"""Experiment 00005's settings with the thread count of this experiment's slot."""
N_ROUNDS: Final[int] = 500
STAGE0_PARAMETERS: Final[dict[str, float | int | str]] = {
    **XGBOOST_PARAMETERS,
    "max_depth": 4,
    "subsample": 1.0,
}
STAGE0_ROUNDS: Final[int] = 200
MAX_ZENITH_DEGREES: Final[float] = 85.0
"""Daylight rows for stage 0 have a solar zenith below this."""

CLEAR_SKY_FIT_MIN_W_M2: Final[float] = 300.0
CLEAR_SKY_QUANTILE: Final[float] = 0.99
MIN_CLEAR_SKY_FRACTION: Final[float] = 0.01
"""Rows whose clear-sky power is below this fraction of capacity get a ratio of zero."""
MAX_RATIO: Final[float] = 2.0
SPEED_BIN_WIDTH_M_S: Final[float] = 0.5

SETTLING_LEVEL_FRACTION: Final[float] = 0.05
SETTLING_RUN_HALF_HOURS: Final[int] = 48
SETTLING_WEEKS: Final[int] = 6
"""The `new_meter_settling` rule, applied as the stage 1 pilot applied it."""

MEMBERS: Final[int] = 51
KEY_COLUMNS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")


def site_table() -> pl.DataFrame:
    """Return the nine generator series with anonymous label, technology, and capacity.

    Returns:
        One row per series with `time_series_id`, `site` (A to F, W1 to W3), `technology`,
        `latitude`, `longitude`, and `effective_capacity_mw`. The coordinates never leave this
        process.

    Raises:
        ValueError: If the anonymisation helper's labels have changed.
    """
    capacity = pl.read_parquet(FRAMES_DIR / "capacity.parquet").select(
        "time_series_id", "effective_capacity_mw"
    )
    solar = pv_sites().select("time_series_id", "site", "latitude", "longitude")
    wind = wind_sites().select("time_series_id", "site", "latitude", "longitude")
    if set(solar["site"].to_list()) != set(SITE_LABELS) or LABEL_PERMUTATION_SEED != 784:
        msg = "the anonymisation helper's solar labels have changed"
        raise ValueError(msg)
    return (
        pl.concat(
            [
                solar.with_columns(technology=pl.lit("solar")),
                wind.with_columns(technology=pl.lit("wind")),
            ]
        )
        .join(capacity, on="time_series_id", how="left", validate="1:1")
        .sort("technology", "site")
    )


def observed_training_power(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return cleaned power over capacity from the ERA5 era to the validation start.

    Rows before each series' settling cutoff are dropped, as the stage 1 pilot did.

    Args:
        sites: `site_table()`.

    Returns:
        Columns `time_series_id`, `valid_time`, and `target` (a fraction of capacity).
    """
    power = (
        scan_power()
        .filter(
            pl.col("time_series_id").is_in(sites["time_series_id"].to_list()),
            pl.col("time") >= ERA5_START,
            pl.col("time") < VALIDATION_START,
        )
        .select("time_series_id", pl.col("time").alias("valid_time"), "power")
        .collect()
        .join(sites.select("time_series_id", "effective_capacity_mw"), on="time_series_id")
        .sort("time_series_id", "valid_time")
    )
    live = (
        pl.col("power").abs() > SETTLING_LEVEL_FRACTION * pl.col("effective_capacity_mw")
    ).fill_null(value=False)
    run_end = (
        live.cast(pl.Int32).rolling_sum(window_size=SETTLING_RUN_HALF_HOURS).over("time_series_id")
        == SETTLING_RUN_HALF_HOURS
    )
    cutoffs = (
        power.filter(run_end)
        .group_by("time_series_id")
        .agg(
            cutoff=pl.col("valid_time")
            .min()
            .dt.offset_by(f"-{SETTLING_RUN_HALF_HOURS * 30 - 30}m")
            .dt.offset_by(f"{SETTLING_WEEKS}w")
        )
    )
    return (
        power.join(cutoffs, on="time_series_id", how="left")
        .filter(pl.col("cutoff").is_null() | (pl.col("valid_time") >= pl.col("cutoff")))
        .filter(pl.col("power").is_finite())
        .select(
            "time_series_id", "valid_time", target=pl.col("power") / pl.col("effective_capacity_mw")
        )
    )


def clear_sky_ghi(*, solar_zenith: np.ndarray) -> np.ndarray:
    """Return clear-sky global horizontal irradiance (W/m2) from the solar zenith.

    Uses `pvlib`'s simplified Solis model with its default aerosol and water-vapour settings, which
    depend on no weather input. Night-time values are zero.

    Args:
        solar_zenith: Solar zenith in degrees at the half-hour midpoint.

    Returns:
        The irradiance, one value per zenith.
    """
    elevation = 90.0 - np.asarray(solar_zenith, dtype=np.float64)
    ghi = np.asarray(
        pvlib.clearsky.simplified_solis(apparent_elevation=np.clip(elevation, 0.0, 90.0))["ghi"]
    )
    return np.where(elevation > 0.0, np.nan_to_num(ghi), 0.0)


def clear_sky_constant(*, power_fraction: np.ndarray, clear_sky: np.ndarray) -> float:
    """Return the farm's clear-sky power over clear-sky irradiance, from training rows only.

    Args:
        power_fraction: Observed power over capacity of the training rows.
        clear_sky: Clear-sky irradiance of the same rows, W/m2.

    Returns:
        The 99th percentile of power over clear-sky irradiance, over rows with clear-sky
        irradiance of at least `CLEAR_SKY_FIT_MIN_W_M2`.

    Raises:
        ValueError: If no training row is bright enough.
    """
    bright = clear_sky >= CLEAR_SKY_FIT_MIN_W_M2
    if not bright.any():
        msg = "no training row has clear-sky irradiance above the fitting threshold"
        raise ValueError(msg)
    return float(np.quantile(power_fraction[bright] / clear_sky[bright], CLEAR_SKY_QUANTILE))


def power_curve(*, speed: np.ndarray, power_fraction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a non-decreasing curve of power fraction on 100 m wind speed.

    Args:
        speed: 100 m wind speeds of the training rows, m/s.
        power_fraction: Observed power over capacity of the same rows.

    Returns:
        The bin centres and the running maximum of each bin's mean power fraction, for
        `np.interp`.
    """
    bins = np.floor(speed / SPEED_BIN_WIDTH_M_S).astype(np.int64)
    frame = (
        pl.DataFrame({"bin": bins, "power": power_fraction})
        .group_by("bin")
        .agg(pl.col("power").mean())
    )
    ordered = frame.sort("bin")
    centres = (ordered["bin"].to_numpy() + 0.5) * SPEED_BIN_WIDTH_M_S
    return centres, np.maximum.accumulate(ordered["power"].to_numpy())
