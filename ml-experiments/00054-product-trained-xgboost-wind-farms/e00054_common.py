"""Paths, constants, and helpers shared by the scripts of experiment 00054.

Copied from experiment 00033 (`e00033_common.py`, `e00033_gen_common.py`, and
`e00033_fit_generators.py`) and experiment 00052; an experiment may not import another
experiment. The inputs it reads are read only.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.anonymise import LABEL_PERMUTATION_SEED, SITE_LABELS
from studies.power import scan_power
from studies.pv_dataset import wind_sites
from studies.sources import ML_EXPERIMENTS_DATA_DIR, PER_STUDY_DIR

EXPERIMENT_ID: Final[int] = 54
SLUG: Final[str] = "product-trained-xgboost-wind-farms"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables; private and never committed."""
SOURCE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00033-best-guess-architecture-vs-production" / "work"
)
REFERENCE_PATHS: Final[dict[str, Path]] = {
    "ref_best_guess": SOURCE_DIR / "generator_tables" / "bg_full.parquet",
    "ref_n1": SOURCE_DIR / "generator_tables" / "n1.parquet",
}
NWP_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR.parent / "studies" / "downloads" / "NWP"
UKV_STORES: Final[tuple[str, ...]] = ("UKV-CEDA", "UKV-CEDA-part2", "UKV-CEDA-part3")
ENS_DIR: Final[Path] = PER_STUDY_DIR / "stage1_pilot" / "inputs" / "ens_features"
ENS_TRAIN_PATH: Final[Path] = ENS_DIR / "train.parquet"
ENS_VALIDATION_DIR: Final[Path] = ENS_DIR / "val"
CAPACITY_PATH: Final[Path] = (
    PER_STUDY_DIR / "quick_wins_fixed_pipeline_1156" / "frames" / "combined" / "fold_a"
) / "capacity.parquet"

SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
N_ROUNDS: Final[int] = 500
MEMBERS: Final[int] = 51
KEY_COLUMNS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")
HOLDOUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
GRID_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
GRID_END: Final[datetime] = datetime(2026, 7, 31, 23, 30, tzinfo=UTC)
STEP: Final[timedelta] = timedelta(minutes=30)
STEPS_PER_DAY: Final[int] = 48
LAST_TRAINING_VALID: Final[datetime] = datetime(2025, 6, 30, 23, 30, tzinfo=UTC)
TRAIN_START: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
RECENCY_HALF_LIFE_DAYS: Final[float] = 2 * 365.25
TELEMETRY_DELAY: Final[timedelta] = timedelta(hours=3)
UKV_UPGRADE: Final[datetime] = datetime(2026, 1, 21, tzinfo=UTC)
MAX_LEAD_HOURS: Final[int] = 54
"""UKV-CEDA's longest lead from a run's 00 UTC start."""
SCORED_FROM_HOURS: Final[int] = 24
"""C1 scores valid times 24 to 54 hours after the 00 UTC start of the ENS run's date."""
UKV_LEADS: Final[int] = 55
"""The length of the store's lead axis (0 to 54 hours)."""
HOURLY_LAST_LEAD: Final[int] = 48
NATIVE_LEADS: Final[tuple[int, ...]] = (*range(HOURLY_LAST_LEAD + 1), 51, 54)

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
UPPER_INDEX: Final[int] = 8
LOWER_INDEX: Final[int] = 0
ATOM_MASSES: Final[np.ndarray] = np.diff(
    np.concatenate([[0.0], (LEVELS[:-1] + LEVELS[1:]) / 2, [1.0]])
)
FINE_LEVELS: Final[np.ndarray] = np.arange(1, 200) / 200
ENS_WIND: Final[tuple[str, ...]] = (
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_dir_sin_10m",
    "wind_dir_cos_10m",
    "wind_dir_sin_100m",
    "wind_dir_cos_100m",
)
UKV_WIND: Final[tuple[str, ...]] = (
    "ukv_speed_10m",
    "ukv_sin_10m",
    "ukv_cos_10m",
    "ukv_speed_925hpa",
)
LAG_COLUMNS: Final[tuple[str, ...]] = (
    "lag_origin",
    "lag_same_time_latest_day",
    "lag_same_time_latest_week",
)
ARM_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "a": (*ENS_WIND, "lead_hours", *LAG_COLUMNS),
    "b": (*ENS_WIND, *UKV_WIND, "lead_hours", *LAG_COLUMNS),
    "c": (*UKV_WIND, "lead_hours", *LAG_COLUMNS),
    "b_placebo": (*ENS_WIND, *UKV_WIND, "lead_hours", *LAG_COLUMNS),
}
XGBOOST_PARAMETERS: Final[dict[str, object]] = {
    "objective": "reg:quantileerror",
    "quantile_alpha": LEVELS,
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "tree_method": "hist",
    "device": "cuda",
    "nthread": 6,
}
TARGET_COVERAGE: Final[float] = 0.90
PLACEBO_SEED: Final[int] = 54


def site_table() -> pl.DataFrame:
    """Return the three wind farms with anonymous label, capacity, and (private) coordinates."""
    wind = wind_sites().select("time_series_id", "site", "latitude", "longitude")
    if LABEL_PERMUTATION_SEED != 784 or not SITE_LABELS:
        msg = "the anonymisation helper has changed"
        raise ValueError(msg)
    capacity = pl.read_parquet(CAPACITY_PATH).select("time_series_id", "effective_capacity_mw")
    return wind.join(capacity, on="time_series_id", how="left", validate="1:1").sort("site")


def grid_times() -> pl.Series:
    """Return the dense half-hourly grid every per-series array is indexed on."""
    return pl.datetime_range(
        GRID_START, GRID_END, interval="30m", time_unit="us", time_zone="UTC", eager=True
    )


def grid_index(*, times: pl.Series) -> np.ndarray:
    """Return the grid index of each timestamp of a series."""
    return ((times.dt.epoch("us") - int(GRID_START.timestamp() * 1e6)) // 1_800_000_000).to_numpy()


def take(*, values: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Return `values[index]`, NaN where the index falls off either end of the grid."""
    inside = (index >= 0) & (index < values.size)
    out = np.full(index.shape, np.nan, dtype=np.float32)
    out[inside] = values[index[inside]]
    return out


def origin_lags(*, power: np.ndarray, target: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Return the three lags measured from the forecast origin, using only usable observations.

    Raises:
        ValueError: If a case's valid time is not after its origin.
    """
    if (target <= origin).any():
        msg = "every valid time must be after its initialisation time"
        raise ValueError(msg)
    newest = origin - int(TELEMETRY_DELAY / STEP) - 1
    gap = target - newest
    day = target - STEPS_PER_DAY * np.ceil(gap / STEPS_PER_DAY).astype(int)
    week = target - 7 * STEPS_PER_DAY * np.ceil(gap / (7 * STEPS_PER_DAY)).astype(int)
    return np.stack(
        [
            take(values=power, index=newest),
            take(values=power, index=day),
            take(values=power, index=week),
        ],
        axis=1,
    )


def full_power(*, sites: pl.DataFrame) -> dict[int, np.ndarray]:
    """Return each farm's observed power over capacity on the dense half-hourly grid."""
    grid = pl.DataFrame({"valid_time": grid_times()})
    power = (
        scan_power()
        .filter(
            pl.col("time_series_id").is_in(sites["time_series_id"].to_list()),
            pl.col("time") >= GRID_START,
        )
        .select("time_series_id", pl.col("time").alias("valid_time"), "power")
        .collect()
    )
    out = {}
    for row in sites.iter_rows(named=True):
        joined = grid.join(
            power.filter(pl.col("time_series_id") == row["time_series_id"]).select(
                "valid_time", "power"
            ),
            on="valid_time",
            how="left",
            validate="1:1",
        )
        out[int(row["time_series_id"])] = (
            joined["power"].to_numpy().astype(np.float64) / row["effective_capacity_mw"]
        ).astype(np.float32)
    return out


def add_lags(*, frame: pl.DataFrame, power: np.ndarray) -> pl.DataFrame:
    """Add `lead_hours` (from the forecast origin) and the three origin lags."""
    lags = origin_lags(
        power=power,
        target=grid_index(times=frame["valid_time"]),
        origin=grid_index(times=frame["power_fcst_init_time"]),
    )
    return frame.with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
    ).with_columns(pl.DataFrame(lags, schema=list(LAG_COLUMNS), orient="row"))


def hours_after_run(*, valid_time: pl.Expr, power_fcst_init_time: pl.Expr) -> pl.Expr:
    """Return the hours from the 00 UTC start of the ENS run's date to the valid time."""
    return (valid_time - power_fcst_init_time.dt.truncate("1d")).dt.total_minutes() / 60.0


def mixture_quantiles(*, member_quantiles: np.ndarray, chunk: int = 20_000) -> np.ndarray:
    """Return each case's mixture quantile function on `FINE_LEVELS` (shape (cases, 199))."""
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


def quantile_score(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return twice the mass-weighted mean pinball loss of the nine levels, per case."""
    difference = actual[:, None] - quantiles
    pinball = np.where(difference >= 0, LEVELS * difference, (LEVELS - 1) * difference)
    return 2.0 * (pinball * ATOM_MASSES).sum(axis=1)


def covered(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return whether the observation lies in the 5% to 95% band, per case."""
    return (actual >= quantiles[:, LOWER_INDEX]) & (actual <= quantiles[:, UPPER_INDEX])
