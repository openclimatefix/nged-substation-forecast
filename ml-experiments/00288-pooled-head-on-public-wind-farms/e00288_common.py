"""Constants and scoring functions shared by the scripts of experiment 00288.

Copied from experiment 00259's `e00259_common.py` (the levels, the member mixture, the spread
factor, and the quantile score), which copied experiments 00230, 00033, and 00052; an experiment may
not import another experiment. W1 to W3 stay anonymous. The public farms are named by their BMU
identifier, which is public.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 288
SLUG: Final[str] = "pooled-head-on-public-wind-farms"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables; private and never committed."""

SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
MEMBERS: Final[int] = 51
ERA5_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
TRAIN_END: Final[datetime] = datetime(2025, 6, 17, tzinfo=UTC)
"""Training uses valid times before this: 14 days before the first validation run."""
KEY_COLUMNS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")
POINTS_PER_FRACTION: Final[float] = 100.0
"""Scores are in points of capacity: a fraction of capacity times this."""

LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
"""The 11 levels every arm forecasts: experiment 33's nine, with the 0.01 and 0.99 tails."""
LEVEL_COLUMNS: Final[tuple[str, ...]] = (
    "q01",
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
    "q99",
)
NINE_INDEX: Final[np.ndarray] = np.arange(1, 10)
"""Positions in `LEVELS` of experiment 33's nine levels."""
MEDIAN_INDEX: Final[int] = 5
LOWER_INDEX: Final[int] = 1
UPPER_INDEX: Final[int] = 9
NINE_LEVELS: Final[np.ndarray] = LEVELS[NINE_INDEX]
NINE_MASSES: Final[np.ndarray] = np.diff(
    np.concatenate([[0.0], (NINE_LEVELS[:-1] + NINE_LEVELS[1:]) / 2, [1.0]])
)
"""Probability mass each of the nine levels carries (experiment 33's quantile score weights)."""
ATOM_MASSES: Final[np.ndarray] = np.diff(
    np.concatenate([[0.0], (LEVELS[:-1] + LEVELS[1:]) / 2, [1.0]])
)
"""Mass each of the 11 levels carries in a member mixture."""
FINE_LEVELS: Final[np.ndarray] = np.arange(1, 200) / 200
"""The 199 levels on which each mixture's quantile function is stored."""
NAMED_LEVELS: Final[tuple[str, ...]] = ("q01", "q05", "q50", "q95", "q99")

TARGET_COVERAGE: Final[float] = 0.90
SPREAD_GRID: Final[np.ndarray] = np.round(np.arange(0.60, 2.001, 0.05), 2)
MIN_FIT_CASES: Final[int] = 5_000
TRIVIAL_LEVEL: Final[float] = 0.01
TELEMETRY_DELAY: Final = np.timedelta64(210, "m")
"""A case is known to the spread-factor fit 3.5 hours after its valid time."""
SHORT_BAND_HOURS: Final[float] = 72.0
"""The spread factor is fitted for leads under this many hours and, separately, for later leads."""

LEAD_DAY_BANDS: Final[tuple[tuple[str, int, int], ...]] = (
    ("days 0-2", 0, 2),
    ("days 1-2", 1, 2),
    ("days 3-6", 3, 6),
    ("days 7+", 7, 10_000),
    ("all leads", 0, 10_000),
)
"""Lead-day bands, by calendar days from the ENS run's date to the valid time (inclusive)."""
STANDARD_AIR_DENSITY_KG_M3: Final[float] = 1.225
DRY_AIR_GAS_CONSTANT: Final[float] = 287.05
FIXED_SURFACE_PRESSURE_PA: Final[float] = 100_000.0
"""ERA5 surface pressure at the farms is not on disk, so every arm uses this fixed value."""
WEATHER_COLUMNS: Final[tuple[str, ...]] = (
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_dir_sin_10m",
    "wind_dir_cos_10m",
    "wind_dir_sin_100m",
    "wind_dir_cos_100m",
    "temperature_c",
)
XGBOOST_FEATURES: Final[tuple[str, ...]] = WEATHER_COLUMNS[:-1]
"""Experiment 52's six wind features, which carry no temperature."""
IDLE_WIND_M_S: Final[float] = 2.0
"""Experiment 210's idle floor: the power distribution at ERA5 100 m wind below this speed."""


def lead_day(*, power_fcst_init_time: pl.Expr, valid_time: pl.Expr) -> pl.Expr:
    """Return calendar days from the ENS run's date to the valid time's date."""
    return (valid_time.dt.date() - power_fcst_init_time.dt.date()).dt.total_days().cast(pl.Int32)


def quantile_score(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return experiment 33's score per case: twice the mass-weighted pinball loss of nine levels.

    Args:
        quantiles: Shape (cases, 11) quantiles at `LEVELS`; only the nine levels of experiment 33
            are used.
        actual: The observed fractions of capacity.

    Returns:
        Shape (cases,) scores in fractions of capacity.
    """
    difference = actual[:, None] - quantiles[:, NINE_INDEX]
    pinball = np.where(difference >= 0, NINE_LEVELS * difference, (NINE_LEVELS - 1) * difference)
    return 2.0 * (pinball * NINE_MASSES).sum(axis=1)


def pinball_at(*, quantiles: np.ndarray, actual: np.ndarray, column: str) -> np.ndarray:
    """Return the pinball loss of one named level per case."""
    index = LEVEL_COLUMNS.index(column)
    level = LEVELS[index]
    difference = actual - quantiles[:, index]
    return np.where(difference >= 0, level * difference, (level - 1) * difference)


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


def mixture_quantiles(*, member_quantiles: np.ndarray, chunk: int = 20_000) -> np.ndarray:
    """Return each case's equal-weight member mixture, read at `LEVELS`.

    Args:
        member_quantiles: Shape (cases, members, 11) member quantiles, each member's levels sorted.
        chunk: Cases sorted at a time.

    Returns:
        Shape (cases, 11) float32 quantiles of the mixture, by linear interpolation of the mixture's
        quantile function on `FINE_LEVELS`.
    """
    cases, members, levels = member_quantiles.shape
    weights = np.tile(ATOM_MASSES / members, members)
    fine = np.empty((cases, FINE_LEVELS.size), dtype=np.float32)
    for start in range(0, cases, chunk):
        flat = member_quantiles[start : start + chunk].reshape(-1, members * levels)
        order = np.argsort(flat, axis=1, kind="stable")
        values = np.take_along_axis(flat, order, axis=1)
        mass = weights[order]
        midpoint = np.cumsum(mass, axis=1) - mass / 2
        for row in range(flat.shape[0]):
            fine[start + row] = np.interp(FINE_LEVELS, midpoint[row], values[row])
    position = LEVELS * 200 - 1
    low = np.floor(position).astype(int)
    high = np.minimum(low + 1, FINE_LEVELS.size - 1)
    fraction = (position - low).astype(np.float32)
    return fine[:, low] * (1 - fraction) + fine[:, high] * fraction


def apply_spread(*, cases: pl.DataFrame, mixture: np.ndarray) -> np.ndarray:
    """Return the quantiles after the spread factor, fitted month by month on earlier cases.

    Follows experiment 33's wind rule: one factor for lead days 0 to 2 and one for the later days,
    each fitted on the non-trivial cases whose valid time precedes the month's first run by the
    telemetry delay, and applied from the month's first run.

    Args:
        cases: The `cases_table` rows in the order of `mixture`.
        mixture: Shape (cases, 11) mixture quantiles.

    Returns:
        Shape (cases, 11) quantiles; months with too few earlier cases keep the mixture.
    """
    actual = cases["actual_norm"].to_numpy().astype(np.float64)
    valid = cases["valid_time"].to_numpy()
    init = cases["power_fcst_init_time"].to_numpy()
    month = cases["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    short = cases["lead_hours"].to_numpy() < SHORT_BAND_HOURS
    scorable = mixture[:, UPPER_INDEX] > TRIVIAL_LEVEL
    out = mixture.copy()
    for month_label in sorted(set(month)):
        in_month = month == month_label
        cutoff = init[in_month].min()
        for group in (short, ~short):
            target = in_month & group & scorable
            fit_index = np.flatnonzero((valid <= cutoff - TELEMETRY_DELAY) & group & scorable)
            if not target.any() or fit_index.size < MIN_FIT_CASES:
                continue
            factor = fit_spread_factor(quantiles=mixture[fit_index], actual=actual[fit_index])
            out[target] = scale_about_median(quantiles=mixture[target], factor=factor)
    return out
