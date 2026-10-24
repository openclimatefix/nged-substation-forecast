# ruff: noqa: E501
"""Shared constants, group labels, and scoring helpers for experiment 335 (tail calibration of default 3)."""

import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 335
SLUG: Final[str] = "primary-demand-default-3-tail-calibration-at-0-5-to-240-hours"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
QUANTILE_NAMES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
BAND_EDGES: Final[np.ndarray] = np.arange(0.0, 241.0, 24.0)
BAND_LABELS: Final[tuple[str, ...]] = tuple(f"{int(a)}-{int(a) + 24} h" for a in BAND_EDGES[:-1])
TAILS: Final[tuple[tuple[str, int, float], ...]] = (
    ("q01", 0, 0.01), ("q05", 1, 0.05), ("q95", 9, 0.95), ("q99", 10, 0.99),
)  # fmt: skip
"""(name, column position, nominal level) of the four calibrated quantiles."""
MEDIAN: Final[int] = 5
WINTER_MONTHS: Final[tuple[int, ...]] = (11, 12, 1, 2, 3)
MIN_CELL_ROWS: Final[int] = 500
FACTOR_BOUNDS: Final[tuple[float, float]] = (0.8, 1.6)
GROUP_NAMES: Final[tuple[str, ...]] = (
    "Apr-Oct day", "Apr-Oct night", "Nov-Mar day", "Nov-Mar night", "Nov-Mar forecast cold night",
)  # fmt: skip


def band_index(*, lead_hours: np.ndarray) -> np.ndarray:
    """Return the 24-hour lead band, 0 for 0.5 to 24 hours up to 9 for 216 to 240 hours."""
    return np.clip(np.ceil(lead_hours / 24.0).astype(int) - 1, 0, len(BAND_LABELS) - 1)


def group_index(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the calibration group known at forecast time: season, day or night, forecast cold night.

    A forecast cold night is a November to March half-hour after dark where at least half of the 51
    ensemble members have a 24-hour mean temperature below 3 C (experiment 322's member share). The
    member share is missing below 168 hours, where the group falls back to "Nov-Mar night".
    """
    local = frame["valid_time"].dt.convert_time_zone("Europe/London")
    winter = np.isin(local.dt.month().to_numpy(), WINTER_MONTHS)
    day = frame["daylight"].fill_null(True).to_numpy()
    share = np.nan_to_num(frame["p_share"].to_numpy(), nan=0.0)
    cold = winter & ~day & (share >= 0.5)
    return np.where(cold, 4, np.where(winter, 2, 0) + np.where(day, 0, 1))


def pinball(*, actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Return the pinball loss of each of the 11 quantiles, shape (rows, 11)."""
    diff = actual[:, None] - quantiles
    return np.maximum(LEVELS * diff, (LEVELS - 1.0) * diff)


def quantile_score(*, actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Return twice the mean pinball loss over the 11 quantiles, per row (points of effective capacity)."""
    return 2.0 * pinball(actual=actual, quantiles=quantiles).mean(axis=1)


def tail_ratio(*, actual: np.ndarray, quantiles: np.ndarray, column: int) -> np.ndarray:
    """Return the standardised exceedance of one tail quantile: the error divided by the quantile's distance from the median.

    For an upper quantile this is (actual - median) / (q - median); for a lower quantile it is
    (median - actual) / (median - q). A factor f gives a calibrated quantile at the level where the
    share of ratios above f equals the tail probability.
    """
    median = quantiles[:, MEDIAN]
    distance = np.abs(quantiles[:, column] - median)
    sign = 1.0 if column > MEDIAN else -1.0
    return sign * (actual - median) / np.maximum(distance, 1e-6)


def apply_factor(*, quantiles: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """Scale the distance of q01, q05, q95, q99 from the median by one factor per row and tail, then sort.

    `factors` has shape (rows, 4) in the order of `TAILS`. Sorting along the quantile axis removes any
    crossing a narrowing factor causes.
    """
    out = quantiles.copy()
    median = quantiles[:, MEDIAN]
    for j, (_, column, _) in enumerate(TAILS):
        out[:, column] = median + factors[:, j] * (quantiles[:, column] - median)
    return np.sort(out, axis=1)


def write_json_atomic(*, path: Path, content: dict) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, default=float))
    temporary.rename(path)
