"""Constants and helpers shared by the scripts of experiment 109 (partially pooled convex model).

The experiment fits the joint convex model of `studies.dp_gam_pooled` at four cuts: 2023-07-01
(tuning), 2024-07-01 (replicate), 2025-07-01 (scored), and 2025-02-01 (the winter onboarding cut).
It scores the 11 quantiles of experiment 87 as the quantile score (twice the mean pinball loss) in
points of effective capacity, on rows outside the hand-filtered switching events.

**Lead.** A frozen fit on past weather does not depend on an initialisation, so each valid time is
written once, against a day-ahead initialisation at 09:00 UTC on the previous day (leads of 15.5 to
39 hours).
"""

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.dp_features import FeatureCache, default_input_paths, load_or_build_feature_cache
from studies.dp_gam import QUANTILE_COLUMNS, QUANTILE_LEVELS, TrainingWindow
from studies.experiment_store import HOLDOUT_START, write_predictions
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 109
SLUG: Final[str] = "differentiable-physics-n0-partially-pooled-convex-model"
ROW_SET_ID: Final[str] = "dp_mw_v1"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
CACHE_DIR: Final[Path] = DATA_DIR / "cache"
WORK_DIR: Final[Path] = DATA_DIR / "work"
"""Small result files (tuning, gate, reports); private and never committed."""

POOL_SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16", "D18")
"""The MW Primaries in the joint fit; a site enters a cut only with the rows it holds."""
DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
"""The sites counted for the decision (4 of 5 needed)."""
HELD_OUT_SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
"""The sites that take turns as the short-history site; D9 is reported beside the decision sites."""
COASTAL_SITES: Final[tuple[str, ...]] = ("D5", "D9")

TUNING_CUT: Final[datetime] = datetime(2023, 7, 1, tzinfo=UTC)
REPLICATE_CUT: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
SCORED_CUT: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
WINTER_CUT: Final[datetime] = datetime(2025, 2, 1, tzinfo=UTC)
CUTS: Final[dict[str, datetime]] = {
    "tuning": TUNING_CUT,
    "replicate": REPLICATE_CUT,
    "scored": SCORED_CUT,
    "winter": WINTER_CUT,
}
SEASON_YEARS: Final[tuple[int, ...]] = tuple(range(2019, 2026))
FULL_WINDOW: Final[TrainingWindow] = TrainingWindow(months=24)
"""Two years before each cut, no drift, for every arm."""
BASE_RIDGE: Final[float] = 0.1
"""The ridge of every full-history substation: experiment 104's tuned value (flat to 0.003)."""
MIN_TRAINING_ROWS: Final[int] = 1344
"""A site enters a cut only with at least four weeks of usable rows."""
LAMBDA_GRID: Final[tuple[float, ...]] = (
    0.0,
    100.0,
    300.0,
    1_000.0,
    3_000.0,
    10_000.0,
    30_000.0,
    100_000.0,
    1_000_000.0,
)
"""The deviation penalty weights tried, in rows of evidence."""
G_SHORT_RIDGE_GRID: Final[tuple[float, ...]] = (0.1, 1.0, 10.0, 100.0)
"""The ridges tried for the per-site comparator on a short history."""
WEEK_DAYS: Final[int] = 7


@dataclass(frozen=True)
class History:
    """How much history the held-out site keeps.

    Attributes:
        label: A short name for stored sets: `4w`, `3m`, `2w`, and so on.
        weeks: The weeks kept, or `None` if `months` is given.
        months: The months kept, or `None` if `weeks` is given.
    """

    label: str
    weeks: int | None = None
    months: int | None = None

    def window(self, *, cut: datetime) -> TrainingWindow:
        """Return the training window of the held-out site at `cut`.

        Args:
            cut: The cut.

        Returns:
            The weeks or months before `cut`.
        """
        if self.months is not None:
            return TrainingWindow(months=self.months)
        return TrainingWindow(start=cut - timedelta(days=WEEK_DAYS * (self.weeks or 0)))

    @property
    def weeks_of_history(self) -> int:
        """The history in weeks, rounding a month up (3 months is 14 weeks, a window of 91 days)."""
        if self.weeks is not None:
            return self.weeks
        return math.ceil((self.months or 0) * 365.25 / 12 / WEEK_DAYS)


FOUR_WEEKS: Final[History] = History(label="4w", weeks=4)
THREE_MONTHS: Final[History] = History(label="3m", months=3)
ONBOARDING_CURVE: Final[tuple[History, ...]] = tuple(
    History(label=f"{weeks}w", weeks=weeks) for weeks in (2, 8, 26)
)


def year_end(*, cut: datetime) -> datetime:
    """Return the end of the scored year that starts at `cut`, never past the held-out start."""
    return min(cut.replace(year=cut.year + 1), HOLDOUT_START)


def load_cache() -> FeatureCache:
    """Return the feature cache, building and saving it on the first call."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)


def to_64(*, moment: datetime) -> np.datetime64:
    """Return a timezone-aware datetime as a naive UTC `datetime64[us]`."""
    return np.datetime64(moment.replace(tzinfo=None), "us")


def scoring_index(
    *, cache: FeatureCache, site: str, after: datetime, through: datetime
) -> np.ndarray:
    """Return the grid rows that have a reading and end in (`after`, `through`].

    Args:
        cache: The cache.
        site: The substation's label.
        after: Rows must end after this instant.
        through: Rows must end no later than this instant.

    Returns:
        Sorted grid positions, never at or after the held-out period.
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


def day_ahead_initialisation(*, end_times: np.ndarray) -> np.ndarray:
    """Return 09:00 UTC on the day before each half-hour's start day, as `datetime64[us]`."""
    day = (end_times - np.timedelta64(30, "m")).astype("datetime64[D]")
    return (day - np.timedelta64(1, "D")).astype("datetime64[us]") + np.timedelta64(9, "h")


def prediction_frame(
    *,
    cache: FeatureCache,
    site: str,
    index: np.ndarray,
    quantiles: np.ndarray,
    mean: np.ndarray,
    arm: str,
    held_out: str = "",
    weeks_of_history: int = 0,
) -> pl.DataFrame:
    """Return one substation's predictions as rows with the columns the scorers read.

    Args:
        cache: The cache (in original points of effective capacity).
        site: The substation's label.
        index: Grid rows predicted.
        quantiles: Shape (len(index), 11), the 11 quantiles in original points.
        mean: The mean forecast the quantiles are built around, in original points.
        arm: The arm's name.
        held_out: The short-history site's label for a short-history arm, else empty.
        weeks_of_history: The held-out site's weeks of history for a short-history arm, else 0.

    Returns:
        Columns `time_series_id`, `valid_time`, `power_fcst_init_time`, `lead_hours`, `seed`,
        `arm`, `held_out`, `weeks_of_history`, `q01` to `q99`, `actual`, `mean`, and
        `switching_mask`.

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
    init = day_ahead_initialisation(end_times=valid)
    return pl.DataFrame(
        {
            "time_series_id": np.full(len(index), features.table.time_series_id, dtype=np.int32),
            "valid_time": pl.Series(valid).dt.replace_time_zone("UTC"),
            "power_fcst_init_time": pl.Series(init).dt.replace_time_zone("UTC"),
            "lead_hours": ((valid - init) / np.timedelta64(1, "h")).astype(np.float64),
            "seed": np.zeros(len(index), dtype=np.int8),
            "arm": np.full(len(index), arm),
            "held_out": np.full(len(index), held_out),
            "weeks_of_history": np.full(len(index), weeks_of_history, dtype=np.int16),
            **{
                column: ordered[:, position].astype(np.float32)
                for position, column in enumerate(QUANTILE_COLUMNS)
            },
            "actual": features.y[index].astype(np.float32),
            "mean": mean.astype(np.float32),
            "switching_mask": features.switching_mask[index],
        }
    )


def store_arm(*, name: str, frame: pl.DataFrame) -> Path:
    """Write an arm's predictions once; a second call returns the existing folder.

    Args:
        name: The set's name.
        frame: The predictions of every scored substation.

    Returns:
        The set's folder.
    """
    folder = DATA_DIR / "predictions" / name
    if folder.exists():
        return folder
    return write_predictions(
        experiment_id=EXPERIMENT_ID, name=name, frame=frame, kind="quantiles", row_set_id=ROW_SET_ID
    )


def quantile_score(*, actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Return twice the mean pinball loss over the 11 levels, per row, in points.

    Args:
        actual: Measured values, shape (n,).
        quantiles: Predicted quantiles, shape (n, 11).

    Returns:
        The quantile score of each row.
    """
    levels = np.array(QUANTILE_LEVELS)[None, :]
    error = actual[:, None] - quantiles
    return 2.0 * np.mean(np.maximum(levels * error, (levels - 1.0) * error), axis=1)
