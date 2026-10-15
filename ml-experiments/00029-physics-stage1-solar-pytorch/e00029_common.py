"""Paths, series, constants, and scoring helpers shared by the scripts of experiment 00029.

The experiment replaces the XGBoost stage 1 of the 6 solar farms with fitted PV physics
(`docs/ml-experiments/00029-physics-stage1-solar-pytorch/result.md`). The mixture arithmetic, the
series table, and the training-power helper copy experiment 00020's; an experiment may not import
another experiment. The inputs it reads were built by the stage 1 pilot and are read only.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.anonymise import LABEL_PERMUTATION_SEED, SITE_LABELS
from studies.power import scan_power
from studies.pv_dataset import pv_sites
from studies.sources import ML_EXPERIMENTS_DATA_DIR, PER_STUDY_DIR

EXPERIMENT_ID: Final[int] = 29
SLUG: Final[str] = "physics-stage1-solar-pytorch"
ROW_SET_ID: Final[str] = "fold_a_tuesdays_v1"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
MEMBERS: Final[int] = 51
KEY_COLUMNS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")
CASE_ORDER: Final[tuple[str, ...]] = ("power_fcst_init_time", "time_series_id", "valid_time")

PILOT_DIR: Final[Path] = PER_STUDY_DIR / "stage1_pilot"
ANALYSIS_PATH: Final[Path] = PILOT_DIR / "inputs" / "analysis_features.parquet"
ENS_VALIDATION_DIR: Final[Path] = PILOT_DIR / "inputs" / "ens_features" / "val"
STAGE2_DIR: Final[Path] = PER_STUDY_DIR / "stage2_pilot"
SCORES_DIR: Final[Path] = STAGE2_DIR / "scores"
FRAMES_DIR: Final[Path] = (
    PER_STUDY_DIR / "quick_wins_fixed_pipeline_1156" / "frames" / "combined" / "fold_a"
)
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables; private and never committed."""

ERA5_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
TRAIN_END: Final[datetime] = datetime(2025, 6, 17, tzinfo=UTC)
"""Every fit uses valid times before this: 14 days before the first validation run."""
MAX_ZENITH_DEGREES: Final[float] = 85.0
"""Daylight rows for the physics fits have a solar zenith below this."""

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
"""The nine weather columns of `base` (experiment 00020), used by the hybrid arm."""
XGBOOST_PARAMETERS: Final[dict[str, object]] = {
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "tree_method": "hist",
    "device": "cuda",
    "nthread": 12,
}
"""Experiment 00002's multi-quantile settings with the thread count of this experiment's slot."""
N_ROUNDS: Final[int] = 500
SETTLING_LEVEL_FRACTION: Final[float] = 0.05
SETTLING_RUN_HALF_HOURS: Final[int] = 48
SETTLING_WEEKS: Final[int] = 6


LEVELS: Final[np.ndarray] = np.array([0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95])
LEVEL_COLUMNS: Final[tuple[str, ...]] = (
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
)
MEDIAN_INDEX: Final[int] = 4
LOWER_INDEX: Final[int] = 0
UPPER_INDEX: Final[int] = 8
ATOM_MASSES: Final[np.ndarray] = np.diff(
    np.concatenate([[0.0], (LEVELS[:-1] + LEVELS[1:]) / 2, [1.0]])
)
"""Probability mass each level carries: the interval between the mid-points to its neighbours."""
FINE_LEVELS: Final[np.ndarray] = np.arange(1, 200) / 200
"""The 199 levels on which each mixture's quantile function is stored."""


LEAD_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0 to 24 h", 0.0, 24.0),
    ("1 to 2 days", 24.0, 72.0),
    ("3 to 6 days", 72.0, 168.0),
    ("7 to 10 days", 168.0, 264.0),
    ("11 to 15 days", 264.0, 10_000.0),
)


def band_names(*, lead_hours: np.ndarray) -> np.ndarray:
    """Return the lead band name of each lead in hours."""
    names = np.full(lead_hours.shape, "none", dtype=object)
    for name, low, high in LEAD_BANDS:
        names[(lead_hours >= low) & (lead_hours < high)] = name
    return names


def mixture_quantiles(*, member_quantiles: np.ndarray, chunk: int = 20_000) -> np.ndarray:
    """Return each case's mixture quantile function on `FINE_LEVELS`.

    Args:
        member_quantiles: Shape (cases, members, 9) member quantiles, each member's levels sorted.
        chunk: Cases sorted at a time.

    Returns:
        Shape (cases, 199) float32 quantiles.
    """
    cases, members, levels = member_quantiles.shape
    weights = np.tile(ATOM_MASSES / members, members)
    out = np.empty((cases, FINE_LEVELS.size), dtype=np.float32)
    for start in range(0, cases, chunk):
        flat = member_quantiles[start : start + chunk].reshape(-1, members * levels)
        order = np.argsort(flat, axis=1, kind="stable")
        values = np.take_along_axis(flat, order, axis=1)
        mass = weights[order]
        midpoint = np.cumsum(mass, axis=1) - mass / 2
        for row in range(flat.shape[0]):
            out[start + row] = np.interp(FINE_LEVELS, midpoint[row], values[row])
    return out


def read_levels(*, fine: np.ndarray) -> np.ndarray:
    """Return the 9 output quantiles of each row of `fine`, by linear interpolation."""
    position = np.clip(LEVELS * 200 - 1, 0, FINE_LEVELS.size - 1)
    low = np.floor(position).astype(int)
    high = np.minimum(low + 1, FINE_LEVELS.size - 1)
    fraction = (position - low).astype(np.float32)
    return fine[:, low] * (1 - fraction) + fine[:, high] * fraction


def crps_estimate(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return twice the mass-weighted mean pinball loss of the 9 levels, per case (a fraction)."""
    difference = actual[:, None] - quantiles
    pinball = np.where(difference >= 0, LEVELS * difference, (LEVELS - 1) * difference)
    return 2.0 * (pinball * ATOM_MASSES).sum(axis=1)


def covered(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return whether the observed power lies in the 5% to 95% band, per case."""
    return (actual >= quantiles[:, LOWER_INDEX]) & (actual <= quantiles[:, UPPER_INDEX])


def site_table() -> pl.DataFrame:
    """Return the 6 solar farms with anonymous label, coordinates, and capacity.

    Returns:
        One row per farm with `time_series_id`, `site` (A to F), `latitude`, `longitude`, and
        `effective_capacity_mw`, sorted by `site`. The coordinates never leave this process.

    Raises:
        ValueError: If the anonymisation helper's labels have changed.
    """
    capacity = pl.read_parquet(FRAMES_DIR / "capacity.parquet").select(
        "time_series_id", "effective_capacity_mw"
    )
    solar = pv_sites().select("time_series_id", "site", "latitude", "longitude")
    if set(solar["site"].to_list()) != set(SITE_LABELS) or LABEL_PERMUTATION_SEED != 784:
        msg = "the anonymisation helper's solar labels have changed"
        raise ValueError(msg)
    return solar.join(capacity, on="time_series_id", how="left", validate="1:1").sort("site")


def observed_training_power(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return cleaned power over capacity from the ERA5 era to the end of training.

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
            pl.col("time") < TRAIN_END,
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


def validation_cases() -> pl.DataFrame:
    """Return experiment 00002's scored cases with technology and lead band, in `CASE_ORDER`.

    Columns: the key columns, `lead_hours`, `actual_norm` (a fraction of capacity), `technology`,
    and `band`.
    """
    cases = (
        pl.read_parquet(SCORES_DIR / "rows_seed0.parquet")
        .filter(pl.col("arm") == "b_mixture_full")
        .select(*KEY_COLUMNS, "lead_hours", "actual_norm")
        .filter(pl.col("time_series_id").is_in(site_table()["time_series_id"].to_list()))
        .sort(*CASE_ORDER)
    )
    return cases.with_columns(band=pl.Series(band_names(lead_hours=cases["lead_hours"].to_numpy())))
