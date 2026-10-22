"""Constants and loaders shared by the scripts of experiment 206 (E1, meter outages in a Primary).

Experiment 206 asks whether a metered solar farm's zero-output runs in clear-sky hours show up in
the supplied power of the Primary substation it sits under. The plan, with every rule below fixed
before any shift was computed, is in
`docs/ml-experiments/00206-dp-meter-regime-steps-e1-farm-outages-in-primary/result.md`.

**Privacy.** Farms appear only under the labels A to F, and Primaries under their D-labels.
"""

from datetime import timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import pvlib
from studies.dp_features import FeatureCache, default_input_paths, load_or_build_feature_cache
from studies.pv_dataset import pv_sites
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

EXPERIMENT_ID: Final[int] = 206
SLUG: Final[str] = "dp-meter-regime-steps-e1-farm-outages-in-primary"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
"""The feature cache of experiment 106, read from there."""
CLEANED_FARMS_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00179-metered-solar-farm-exploration" / "cleaned_solar_farms.parquet"
)
HIERARCHY_DIR: Final[Path] = REPO_DATA_DIR / "NGED" / "substation_hierarchy_2026-10-15"
DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16", "D9")
"""The decision Primaries, with D9 beside them."""
FARM_COMMISSIONED: Final[dict[str, str]] = {"E": "2024-10-06"}
"""Farm E counts only from its commissioning date (`studies.commissioning`)."""
FARM_CLEAR_SKY_W_M2: Final[float] = 500.0
"""A farm half-hour is a clear-sky half-hour when clear-sky irradiance exceeds this."""
MIN_CLEAR_HALF_HOURS_PER_DAY: Final[int] = 4
"""A day with fewer clear-sky half-hours is neutral: it neither extends nor breaks an outage."""
BAD_DAY_SHARE: Final[float] = 0.8
"""A day is bad when at least this share of its clear-sky half-hours is flagged or missing."""
MIN_OUTAGE_DAYS: Final[int] = 3
MAX_NEUTRAL_GAP_DAYS: Final[int] = 4
"""An outage run bridges neutral days only when the next eligible day is this close."""
SUNNY_CLEAR_SKY_W_M2: Final[float] = 300.0
SUNNY_CAMS_SHARE: Final[float] = 0.8
"""A Primary half-hour is sunny daylight when clear-sky irradiance exceeds
`SUNNY_CLEAR_SKY_W_M2` and CAMS irradiance is at least this share of clear-sky."""
REFERENCE_DAYS: Final[int] = 14
MIN_SUNNY_IN_OUTAGE: Final[int] = 20
MIN_SUNNY_IN_REFERENCE: Final[int] = 60
MIN_SAMPLES_PER_HALF_HOUR: Final[int] = 2
SHIFT_RATIO_LOW: Final[float] = 0.5
SHIFT_RATIO_HIGH: Final[float] = 1.5
N_RESAMPLES: Final[int] = 400
NEAR_KM: Final[float] = 10.0
N_PLACEBO: Final[int] = 200
SEED: Final[int] = 206


def load_cache() -> FeatureCache:
    """Return the feature cache of experiment 106."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)


def clear_sky_ghi(*, end_times: np.ndarray, latitude: float, longitude: float) -> np.ndarray:
    """Return Ineichen clear-sky GHI at the mid-point of each period-ending stamp.

    Args:
        end_times: Period-end stamps as naive UTC `datetime64`.
        latitude: Degrees north.
        longitude: Degrees east.

    Returns:
        Clear-sky global horizontal irradiance in W/m².
    """
    import pandas as pd  # noqa: TID251  (pvlib's clear-sky API takes a pandas index)

    middle = pd.DatetimeIndex(end_times) - timedelta(minutes=15)
    location = pvlib.location.Location(latitude, longitude)
    return location.get_clearsky(middle.tz_localize("UTC"), model="ineichen")["ghi"].to_numpy()


def farm_table() -> pl.DataFrame:
    """Return the six farms with label, series id, location and effective capacity."""
    return pv_sites().sort("site")
