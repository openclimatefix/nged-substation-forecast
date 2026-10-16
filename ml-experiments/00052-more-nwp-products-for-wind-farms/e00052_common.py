"""Paths, constants, and helpers shared by the scripts of experiment 00052.

Copied from experiment 00033 (`e00033_gen_common.py`, `e00033_fit_generators.py`); an experiment
may not import another experiment. The inputs it reads are read only.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import xgboost as xgb
from studies.anonymise import LABEL_PERMUTATION_SEED, SITE_LABELS
from studies.power import scan_power
from studies.pv_dataset import wind_sites
from studies.sources import ML_EXPERIMENTS_DATA_DIR, PER_STUDY_DIR

EXPERIMENT_ID: Final[int] = 52
SLUG: Final[str] = "more-nwp-products-for-wind-farms"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables; private and never committed."""
SOURCE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00033-best-guess-architecture-vs-production" / "work"
)
BEST_GUESS_PATH: Final[Path] = SOURCE_DIR / "generator_tables" / "bg_full.parquet"
NWP_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR.parent / "studies" / "downloads" / "NWP" / "OPEN-METEO-PREVIOUS-RUNS"
)
PILOT_INPUTS: Final[Path] = PER_STUDY_DIR / "stage1_pilot" / "inputs"
ANALYSIS_PATH: Final[Path] = PILOT_INPUTS / "analysis_features.parquet"
CAPACITY_PATH: Final[Path] = (
    PER_STUDY_DIR / "quick_wins_fixed_pipeline_1156" / "frames" / "combined" / "fold_a"
) / "capacity.parquet"

SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
N_ROUNDS: Final[int] = 500
ERA5_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
TRAIN_END: Final[datetime] = datetime(2025, 6, 17, tzinfo=UTC)
"""Training uses valid times before this: 14 days before the first validation run."""
HOLDOUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
KEY_COLUMNS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")

PRODUCTS: Final[dict[str, str]] = {
    "icon": "ICON-EU",
    "ifs": "ECMWF-IFS-025",
    "gfs": "GFS-SEAMLESS",
}
"""Product key to the folder of its Open-Meteo Previous Runs table."""
MAX_DAY: Final[dict[str, int]] = {"icon": 4, "ifs": 7, "gfs": 7}
"""The largest `previous_day` column each product's table holds with values."""
PUBLICATION_DELAY_HOURS: Final[dict[str, int]] = {"icon": 4, "ifs": 8, "gfs": 8}
"""Upper bounds on the hours from a run's initialisation to its publication."""
SERVICE_READ_HOUR: Final[int] = 9
KM_H_PER_M_S: Final[float] = 3.6

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
WIND_FEATURES: Final[tuple[str, ...]] = (
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_dir_sin_10m",
    "wind_dir_cos_10m",
    "wind_dir_sin_100m",
    "wind_dir_cos_100m",
)
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
SPEED_BIN_WIDTH_M_S: Final[float] = 0.5
SETTLING_LEVEL_FRACTION: Final[float] = 0.05
SETTLING_RUN_HALF_HOURS: Final[int] = 48
SETTLING_WEEKS: Final[int] = 6
TARGET_COVERAGE: Final[float] = 0.90
SPREAD_GRID: Final[np.ndarray] = np.round(np.arange(0.60, 3.001, 0.05), 2)
"""Wider than experiment 33's grid, which stopped at 2.0, because a product's forecast is narrow."""
MIN_FIT_CASES: Final[int] = 5_000
TRIVIAL_LEVEL: Final[float] = 0.01
TELEMETRY_DELAY: Final = np.timedelta64(210, "m")
"""A case is known to the spread-factor fit 3.5 hours after its valid time."""

LEAD_DAY_BANDS: Final[tuple[tuple[str, int, int], ...]] = (
    ("day 0", 0, 0),
    ("days 1-2", 1, 2),
    ("days 3-6", 3, 6),
    ("days 7+", 7, 10_000),
)
"""Lead-day bands, by calendar days from the ENS run's date to the valid time (inclusive)."""


def site_table() -> pl.DataFrame:
    """Return the 3 wind farms with anonymous label and capacity (coordinates stay private)."""
    wind = wind_sites().select("time_series_id", "site", "latitude", "longitude")
    if LABEL_PERMUTATION_SEED != 784 or not SITE_LABELS:
        msg = "the anonymisation helper has changed"
        raise ValueError(msg)
    capacity = pl.read_parquet(CAPACITY_PATH).select("time_series_id", "effective_capacity_mw")
    return wind.join(capacity, on="time_series_id", how="left", validate="1:1").sort("site")


def lead_day(*, power_fcst_init_time: pl.Expr, valid_time: pl.Expr) -> pl.Expr:
    """Return calendar days from the ENS run's date to the valid time's date."""
    return (valid_time.dt.date() - power_fcst_init_time.dt.date()).dt.total_days().cast(pl.Int32)


def band_of(*, days: np.ndarray) -> np.ndarray:
    """Return the lead-day band name of each lead day."""
    names = np.full(days.shape, "none", dtype=object)
    for name, low, high in LEAD_DAY_BANDS:
        names[(days >= low) & (days <= high)] = name
    return names


def quantile_score(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return twice the mass-weighted mean pinball loss of the nine levels, per case."""
    difference = actual[:, None] - quantiles
    pinball = np.where(difference >= 0, LEVELS * difference, (LEVELS - 1) * difference)
    return 2.0 * (pinball * ATOM_MASSES).sum(axis=1)


def covered(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return whether the observation lies in the 5% to 95% band, per case."""
    return (actual >= quantiles[:, LOWER_INDEX]) & (actual <= quantiles[:, UPPER_INDEX])


def scale_about_median(*, quantiles: np.ndarray, factor: float) -> np.ndarray:
    """Return `median + factor * (q - median)`, clipped to 0 to 1 and sorted."""
    median = quantiles[:, MEDIAN_INDEX : MEDIAN_INDEX + 1]
    return np.sort(np.clip(median + factor * (quantiles - median), 0.0, 1.0), axis=1)


def fit_spread_factor(*, quantiles: np.ndarray, actual: np.ndarray) -> float:
    """Return the grid factor whose 5% to 95% coverage is nearest `TARGET_COVERAGE`."""
    gaps = [
        abs(
            covered(
                quantiles=scale_about_median(quantiles=quantiles, factor=float(k)), actual=actual
            ).mean()
            - TARGET_COVERAGE
        )
        for k in SPREAD_GRID
    ]
    return float(SPREAD_GRID[int(np.argmin(gaps))])


def observed_training_power(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return cleaned power over capacity from the ERA5 era to the end of training."""
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


def power_curve(*, speed: np.ndarray, power_fraction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a non-decreasing curve of power fraction on 100 m wind speed (bin centres, curve)."""
    bins = np.floor(speed / SPEED_BIN_WIDTH_M_S).astype(np.int64)
    ordered = (
        pl.DataFrame({"bin": bins, "power": power_fraction})
        .group_by("bin")
        .agg(pl.col("power").mean())
        .sort("bin")
    )
    centres = (ordered["bin"].to_numpy() + 0.5) * SPEED_BIN_WIDTH_M_S
    return centres, np.maximum.accumulate(ordered["power"].to_numpy())


class WindModel:
    """One farm's ERA5-trained nine-quantile model with the power curve as a base margin."""

    def __init__(self, *, rows: pl.DataFrame, seed: int, rounds: int) -> None:
        """Fit the model on `rows`, which hold `WIND_FEATURES` and `target`."""
        features = self._matrix(rows)
        target = rows["target"].to_numpy()
        self.curve = power_curve(speed=rows["wind_speed_100m"].to_numpy(), power_fraction=target)
        margin = self._margin(speed=rows["wind_speed_100m"].to_numpy())
        self.booster = xgb.train(
            {**XGBOOST_PARAMETERS, "seed": seed},
            xgb.DMatrix(features, label=target, base_margin=margin),
            num_boost_round=rounds,
        )

    @staticmethod
    def _matrix(frame: pl.DataFrame) -> np.ndarray:
        return frame.select([pl.col(c).cast(pl.Float32) for c in WIND_FEATURES]).to_numpy()

    def _margin(self, *, speed: np.ndarray) -> np.ndarray:
        centres, curve = self.curve
        return np.tile(np.interp(speed, centres, curve)[:, None], (1, len(LEVEL_COLUMNS)))

    def predict(self, *, frame: pl.DataFrame) -> np.ndarray:
        """Return the 9 sorted quantile columns for every row (a fraction of capacity)."""
        matrix = xgb.DMatrix(
            self._matrix(frame), base_margin=self._margin(speed=frame["wind_speed_100m"].to_numpy())
        )
        return np.sort(self.booster.predict(matrix), axis=1)
