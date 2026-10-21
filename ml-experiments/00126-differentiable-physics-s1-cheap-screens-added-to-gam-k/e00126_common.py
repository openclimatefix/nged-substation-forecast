"""Constants and helpers shared by the scripts of experiment 126 (differentiable physics S1).

The experiment asks whether interaction terms (N3) or a per-site heating threshold (N4) improve
GAM-K's convex model after its recent-level Kalman filter. GAM-K is `studies.dp_gam`'s model with a
24-month window, no drift, the capacity pinned to experiment 45's real-time vintage, and a ridge of
0.1. Every contrast is on past weather (ERA5 temperature and CAMS irradiance), a ceiling for a live
forecast.

**Four cuts, each with one role.** A fit at cut `c` is frozen for the year after `c`. The fit at
2022-07-01 is the replica that supplies the tuning year's spread, the fit at 2023-07-01 scores the
tuning year, the fit at 2024-07-01 scores the replicate year, and the fit at 2025-07-01 scores the
verdict year. The replica of each scored cut is the fit 12 months earlier.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, cast

import numpy as np
import polars as pl
from studies.dp_assimilation import day_ahead_initialisation
from studies.dp_features import FeatureCache, default_input_paths, load_or_build_feature_cache
from studies.dp_gam import QUANTILE_COLUMNS, QUANTILE_LEVELS, GamSpec, TrainingWindow
from studies.experiment_store import HOLDOUT_START, write_predictions
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 126
SLUG: Final[str] = "differentiable-physics-s1-cheap-screens-added-to-gam-k"
ROW_SET_ID: Final[str] = "dp_mw_v1"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
"""Small result files (fits, decisions, gates, reports); private and never committed."""
CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
"""The feature cache that experiment 106 built; the same inputs hash to the same file."""

DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
"""The five substations that decide the verdict (4 of 5 needed)."""
SCORED_SITES: Final[tuple[str, ...]] = (*DECISION_SITES, "D9")
"""D9 is scored beside them in the verdict year only: it holds 9 months at the replicate cut."""
COASTAL_SITES: Final[tuple[str, ...]] = ("D5", "D9")

REPLICA_TUNING_CUT: Final[datetime] = datetime(2022, 7, 1, tzinfo=UTC)
TUNING_CUT: Final[datetime] = datetime(2023, 7, 1, tzinfo=UTC)
REPLICATE_CUT: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
SCORED_CUT: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
FIT_CUTS: Final[dict[str, datetime]] = {
    "replica_tuning": REPLICA_TUNING_CUT,
    "tuning": TUNING_CUT,
    "replicate": REPLICATE_CUT,
    "scored": SCORED_CUT,
}
"""Every fit's cut, by the year it scores: `replica_tuning` scores 2022-23 and exists only to
supply the tuning year's spread."""
CHAIN: Final[dict[str, tuple[str, ...]]] = {
    "tuning": ("tuning",),
    "replicate": ("tuning", "replicate"),
    "scored": ("tuning", "replicate", "scored"),
}
"""The cuts whose out-of-sample residuals a filter carries into each scored year."""
YEAR_MONTHS: Final[int] = 12

RIDGE: Final[float] = 0.1
"""Experiment 104's tuned ridge penalty."""
WINDOW_MONTHS: Final[int] = 24
SEASON_YEARS: Final[range] = range(2019, 2026)
MIN_HINGE_C: Final[float] = 12.0
MAX_HINGE_C: Final[float] = 19.0
HINGE_STEP_C: Final[float] = 0.5
HINGE_GRID_C: Final[tuple[float, ...]] = tuple(
    float(v) for v in np.arange(MIN_HINGE_C, MAX_HINGE_C + 1e-9, HINGE_STEP_C)
)
"""The 15 candidate heating thresholds, from 12 to 19 degrees C in 0.5 degree steps."""
BASE_THRESHOLD_C: Final[float] = 15.5
LAMBDA_GRID: Final[tuple[float, ...]] = (1.0, 100.0, 10000.0)
"""The interaction penalty weights, each multiplied by the training rows over `SITE_YEAR_ROWS`."""
THRESHOLD_FLAT_RANGE_C: Final[tuple[float, float]] = (13.5, 17.5)
"""A site's threshold is printed only if its interval lies wholly outside this range."""
N_BOOTSTRAP: Final[int] = 50
"""Month-block resamples per profile or capacity interval."""
SCORED_LEVEL: Final[float] = 100.0 - 5.0 / 2.0
"""The 97.5% interval that two Bonferroni-corrected planned contrasts need."""
N_RESAMPLES: Final[int] = 20000


def configure_threads(*, threads: int = 1) -> None:
    """Limit the torch threads of this process (the BLAS libraries read their own variables).

    Args:
        threads: The number of threads.
    """
    import os

    import torch

    torch.set_num_threads(threads)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(threads)


def load_cache() -> FeatureCache:
    """Return the feature cache of experiments 104 and 106."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)


def base_spec() -> GamSpec:
    """Return B0's penalties: GAM-K's ridge, no drift, and the capacity pinned to the vintage."""
    return GamSpec(ridge=RIDGE, drift="none", capacity="pinned", final_knot="free")


def fitted_capacity_spec() -> GamSpec:
    """Return B0's penalties with the PV capacity fitted instead of pinned."""
    return GamSpec(ridge=RIDGE, drift="none", capacity="fitted", final_knot="free")


def base_window() -> TrainingWindow:
    """Return GAM-K's training window: the last 24 months before the cut."""
    return TrainingWindow(months=WINDOW_MONTHS)


def year_end(*, cut: datetime) -> datetime:
    """Return the end of the scored year that starts at `cut`, never past the held-out start."""
    return min(cut.replace(year=cut.year + 1), HOLDOUT_START)


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


def prediction_frame(
    *,
    cache: FeatureCache,
    site: str,
    index: np.ndarray,
    quantiles: np.ndarray,
    mean: np.ndarray,
    arm: str,
    level: np.ndarray | None = None,
    anomaly: np.ndarray | None = None,
) -> pl.DataFrame:
    """Return one substation's day-ahead predictions as rows with the columns the scorers read.

    Args:
        cache: The cache.
        site: The substation's label.
        index: Grid rows predicted.
        quantiles: Shape (len(index), 11), the 11 quantiles in points.
        mean: The mean forecast the quantiles are built around, in points.
        arm: The arm's name.
        level: The filter's persistent part of the correction, for a filtered arm.
        anomaly: The filter's transient part of the correction, for a filtered arm.

    Returns:
        Columns `time_series_id`, `valid_time`, `power_fcst_init_time`, `lead_hours`, `seed`,
        `arm`, `q01` to `q99`, `actual`, `mean`, `switching_mask`, and, for a filtered arm,
        `level` and `anomaly`.

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
    init = day_ahead_initialisation(end_times=valid, days_ahead=1)
    columns: dict[str, object] = {
        "time_series_id": np.full(len(index), features.table.time_series_id, dtype=np.int32),
        "valid_time": pl.Series(valid).dt.replace_time_zone("UTC"),
        "power_fcst_init_time": pl.Series(init).dt.replace_time_zone("UTC"),
        "lead_hours": ((valid - init) / np.timedelta64(1, "h")).astype(np.float64),
        "seed": np.zeros(len(index), dtype=np.int8),
        "arm": np.full(len(index), arm),
        **{
            column: ordered[:, position].astype(np.float32)
            for position, column in enumerate(QUANTILE_COLUMNS)
        },
        "actual": features.y[index].astype(np.float32),
        "mean": mean.astype(np.float32),
        "switching_mask": features.switching_mask[index],
    }
    if level is not None and anomaly is not None:
        columns["level"] = level.astype(np.float32)
        columns["anomaly"] = anomaly.astype(np.float32)
    return pl.DataFrame(columns)


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


def as_float(value: object) -> float:
    """Return a polars scalar (typed as a wide union) as a float."""
    return float(cast("float", value))
