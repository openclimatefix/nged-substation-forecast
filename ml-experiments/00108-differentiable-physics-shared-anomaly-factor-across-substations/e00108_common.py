"""Constants and helpers shared by the scripts of experiment 108 (a shared anomaly factor).

The experiment corrects the frozen convex model A0 of experiment 106 (all years, no drift, pinned
capacity) with the per-site Kalman filter of `studies.dp_assimilation` and with a filter on a
factor built from the other substations' residuals. Quantile scores are twice the mean pinball loss
over experiment 87's 11 quantiles, in points of effective capacity.

**Lead.** A frozen fit on past weather does not depend on an initialisation, so each valid time is
written against a day-ahead initialisation at 09:00 UTC on the previous day (leads of 15.5 to 39
hours). Filter arms also write `days_ahead` 1 to 7.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.dp_assimilation import day_ahead_initialisation
from studies.dp_features import FeatureCache, default_input_paths, load_or_build_feature_cache
from studies.dp_gam import QUANTILE_COLUMNS, QUANTILE_LEVELS
from studies.experiment_store import HOLDOUT_START, write_predictions
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 108
SLUG: Final[str] = "differentiable-physics-shared-anomaly-factor-across-substations"
ROW_SET_ID: Final[str] = "dp_mw_v1"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
"""Small result files (fits, tuning, decisions, reports); private and never committed."""
PART4_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
)
"""Experiment 106's private folder: the feature cache, `work/tuned.json`, and the stored sets."""
CACHE_DIR: Final[Path] = PART4_DIR / "cache"

SCORED_SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
"""The six MW long-history substations: the contributors of the decision factor."""
DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
"""The sites counted for the decision; D9 is reported beside them."""

CUT_YEARS: Final[tuple[int, ...]] = (2021, 2022, 2023, 2024, 2025)
"""Every contributor is fitted at 1 July of each of these years."""
TUNING_YEAR: Final[int] = 2023
REPLICATE_YEAR: Final[int] = 2024
SCORED_YEAR: Final[int] = 2025
SCORED_YEARS: Final[dict[str, int]] = {
    "tuning": TUNING_YEAR,
    "replicate": REPLICATE_YEAR,
    "scored": SCORED_YEAR,
}
SEASON_YEARS: Final[range] = range(2019, 2026)
MIN_TRAINING_MONTHS: Final[int] = 6
"""A site contributes from a cut once it has this many months of readings before the cut."""
MIN_CONTRIBUTORS: Final[int] = 3
"""The factor is missing where fewer contributors than this have a reading."""
HALF_HOURS_PER_WEEK: Final[int] = 7 * 48
DELAY_HOURS: Final[float] = 3.5
SEED: Final[int] = 108


def cut_of(*, year: int) -> datetime:
    """Return the cut of the fit made at 1 July of `year`."""
    return datetime(year, 7, 1, tzinfo=UTC)


def year_end(*, cut: datetime) -> datetime:
    """Return the end of the scored year that starts at `cut`, never past the held-out start."""
    return min(cut.replace(year=cut.year + 1), HOLDOUT_START)


def load_cache() -> FeatureCache:
    """Return experiment 106's feature cache (built once there, read here)."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)


def to_64(*, moment: datetime) -> np.datetime64:
    """Return a timezone-aware datetime as a naive UTC `datetime64[us]`."""
    return np.datetime64(moment.replace(tzinfo=None), "us")


def grid_position_of_cut(*, cache: FeatureCache, cut: datetime) -> int:
    """Return the grid position of the last half-hour that ends at or before `cut`."""
    return int(np.searchsorted(cache.end_times, to_64(moment=cut), side="right")) - 1


def scoring_index(
    *, cache: FeatureCache, site: str, after: datetime, through: datetime
) -> np.ndarray:
    """Return the grid rows that have a reading and end in (`after`, `through`].

    Never at or after the held-out period.
    """
    features = cache.sites[site]
    end = cache.end_times
    keep = (
        features.present
        & np.isfinite(features.y)
        & (end > to_64(moment=after))
        & (end <= to_64(moment=min(through, HOLDOUT_START - timedelta(minutes=30))))
    )
    return np.flatnonzero(keep)


def prediction_frame(
    *,
    cache: FeatureCache,
    site: str,
    index: np.ndarray,
    quantiles: np.ndarray,
    mean: np.ndarray,
    arm: str,
    days_ahead: int = 1,
    columns: dict[str, np.ndarray] | None = None,
    init: np.ndarray | None = None,
) -> pl.DataFrame:
    """Return one substation's predictions as rows with the columns the scorers read.

    Args:
        cache: The cache.
        site: The substation's label.
        index: Grid rows predicted.
        quantiles: Shape (len(index), 11), the 11 quantiles in points.
        mean: The mean forecast the quantiles are built around, in points.
        arm: The arm's name.
        days_ahead: How many days before the valid day the forecast is made; 0 for the same-day
            row, which needs `init`.
        columns: Extra per-row columns, such as the filter states.
        init: The initialisation time of each row as `datetime64[us]`; the day-ahead initialisation
            if not given.

    Returns:
        Columns `time_series_id`, `valid_time`, `power_fcst_init_time`, `lead_hours`, `seed`,
        `arm`, `days_ahead`, `q01` to `q99`, `actual`, `mean`, `switching_mask`, and `columns`.

    Raises:
        ValueError: If the quantiles' shape is wrong.
    """
    features = cache.sites[site]
    if quantiles.shape != (len(index), len(QUANTILE_LEVELS)):
        expected = (len(index), len(QUANTILE_LEVELS))
        msg = f"expected quantiles of shape {expected}, got {quantiles.shape}"
        raise ValueError(msg)
    ordered = np.sort(quantiles, axis=1)
    valid = cache.end_times[index]
    if init is None:
        init = day_ahead_initialisation(end_times=valid, days_ahead=days_ahead)
    frame: dict[str, object] = {
        "time_series_id": np.full(len(index), features.table.time_series_id, dtype=np.int32),
        "valid_time": pl.Series(valid).dt.replace_time_zone("UTC"),
        "power_fcst_init_time": pl.Series(init).dt.replace_time_zone("UTC"),
        "lead_hours": ((valid - init) / np.timedelta64(1, "h")).astype(np.float64),
        "seed": np.zeros(len(index), dtype=np.int8),
        "arm": np.full(len(index), arm),
        "days_ahead": np.full(len(index), days_ahead, dtype=np.int8),
        **{
            column: ordered[:, position].astype(np.float32)
            for position, column in enumerate(QUANTILE_COLUMNS)
        },
        "actual": features.y[index].astype(np.float32),
        "mean": mean.astype(np.float32),
        "switching_mask": features.switching_mask[index],
    }
    for name, values in (columns or {}).items():
        frame[name] = values.astype(np.float32)
    return pl.DataFrame(frame)


def store_arm(*, name: str, frame: pl.DataFrame) -> Path:
    """Write an arm's predictions once; a second call returns the existing folder."""
    folder = DATA_DIR / "predictions" / name
    if folder.exists():
        return folder
    return write_predictions(
        experiment_id=EXPERIMENT_ID, name=name, frame=frame, kind="quantiles", row_set_id=ROW_SET_ID
    )


def quantile_score(*, actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Return twice the mean pinball loss over the 11 levels, per row, in points."""
    levels = np.array(QUANTILE_LEVELS)[None, :]
    error = actual[:, None] - quantiles
    return 2.0 * np.mean(np.maximum(levels * error, (levels - 1.0) * error), axis=1)
