"""Constants and helpers shared by the scripts of experiment 110 (differentiable physics, part 5).

The experiment fits the convex model of `studies.dp_gam` per substation at three cuts (the tuning
cut 2023-07-01, the replicate cut 2024-07-01, and the scored cut 2025-07-01) with the
generation-side options of experiment 110 (an MVA meter, an unmetered generator node, an export
clip, and capacity-identification options), corrects each arm with the recent-level filter of
`studies.dp_assimilation`, and scores the 11 quantiles of experiment 87 as the quantile score
(twice the mean pinball loss) in points of effective capacity.

**Lead.** A frozen fit on past weather does not depend on an initialisation, so each valid time is
written against a day-ahead initialisation at 09:00 UTC on the previous day (leads of 15.5 to 39
hours).

**Privacy.** Substations appear by D-label only. The injected farms (labels A to F in experiment
39's frame) are metered generators, so only ratios of recovered to injected capacity leave the
private data folder. The real generator case is plotted under the neutral label G-c.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.dp_assimilation import day_ahead_initialisation
from studies.dp_features import (
    FeatureCache,
    default_input_paths,
    load_or_build_feature_cache,
)
from studies.dp_gam import QUANTILE_COLUMNS, QUANTILE_LEVELS
from studies.experiment_store import HOLDOUT_START, write_predictions
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 110
SLUG: Final[str] = "differentiable-physics-5-embedded-generation-and-capacity"
ROW_SET_ID: Final[str] = "dp_mw_v1"
MVA_ROW_SET_ID: Final[str] = "dp_mva_sunlit_v1"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
"""Small result files (tuning, decisions, gate, capacity tables); private and never committed."""
PART_4_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
)
"""Part 4's private folder: the feature cache, its tuned settings, and its stored predictions."""
CACHE_DIR: Final[Path] = PART_4_DIR / "cache"
"""The feature cache, built from the same input files as Part 4's and read from there."""
INJECTION_FRAME_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00039-pv-disaggregation-cams-injection" / "work" / "frame.parquet"
)
"""Experiment 39's frame, which holds the solar farms' power beside the CAMS-driven unit shape."""

DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
"""The MW sites counted for the decision; D9 holds 9 months at 2024-07-01."""
MW_SCORED_SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
"""The MW sites fitted: the decision sites and D9 beside them. D18 has no usable rows before the
replica cut, so it has no replica spread and is not fitted."""
MVA_SITES: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
COASTAL_SITES: Final[tuple[str, ...]] = ("D5", "D9")
"""Possible sunshine-tracking loads (PVDIS), reported separately."""
HELD_VINTAGE_SITE: Final[str] = "D14"
"""D14's vintage stops in 2023, so its reference is the capacity held after the cut, flagged."""
FLAGGED_CLIP_SITE: Final[str] = "D5"
"""The one decision site where Part 1 found clipping (4 of 7 summers)."""
INJECTION_SITES: Final[tuple[str, ...]] = ("D10", "D11", "D14", "D16")
"""Where the gate injects farms, generators, and clips."""
TUNING_INJECTION_SITES: Final[tuple[str, ...]] = ("D5", "D9")
"""Where the node's penalties are tuned, on an injection that gate test 4 never uses."""

TUNING_CUT: Final[datetime] = datetime(2023, 7, 1, tzinfo=UTC)
REPLICATE_CUT: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
SCORED_CUT: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
CUTS: Final[dict[str, datetime]] = {
    "tuning": TUNING_CUT,
    "replicate": REPLICATE_CUT,
    "scored": SCORED_CUT,
}
CAPACITY_CUTS: Final[tuple[str, ...]] = ("replicate", "scored")
"""The two cuts at which the capacity labels are read."""
SEASON_YEARS: Final[range] = range(2019, 2026)
RIDGE: Final[float] = 0.1
"""Experiment 104's tuned ridge penalty, as Part 4 used it."""
MARGIN_POINTS: Final[float] = 0.10
N_BOOTSTRAP_WEIGHTINGS: Final[int] = 20
BOOTSTRAP_SEED: Final[int] = 110
PRECISION_FLOOR_POINTS: Final[float] = 3.0
PRECISION_FLOOR_SHARE: Final[float] = 0.10
NEAR_VINTAGE_SHARE: Final[float] = 0.20
K_III_RATIO_RANGE: Final[tuple[float, float]] = (0.65, 0.87)
"""The slope with which CAMS tracks a real farm's anomaly (experiment 63)."""
POWER_FACTOR_PRIOR_BAND: Final[tuple[float, float]] = (0.95, 0.99)


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


def prediction_frame(
    *,
    cache: FeatureCache,
    site: str,
    index: np.ndarray,
    quantiles: np.ndarray,
    mean: np.ndarray,
    arm: str,
    filtered: bool,
    days_ahead: int = 1,
    extra: dict[str, np.ndarray] | None = None,
) -> pl.DataFrame:
    """Return one substation's predictions as rows with the columns the scorers read.

    Args:
        cache: The cache.
        site: The substation's label.
        index: Grid rows predicted.
        quantiles: Shape (len(index), 11), the 11 quantiles in points.
        mean: The mean forecast the quantiles are built around, in points.
        arm: The arm's name.
        filtered: True when the recent-level filter's correction is in the forecast.
        days_ahead: How many days before the valid day the forecast is made.
        extra: Further columns, such as `p_mw` and `q_mvar` for a meter arm, each of length
            `len(index)`.

    Returns:
        Columns `time_series_id`, `valid_time`, `power_fcst_init_time`, `lead_hours`, `seed`,
        `arm`, `filtered`, `days_ahead`, `q01` to `q99`, `actual`, `mean`, `switching_mask`, and the
        extra columns.

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
    init = day_ahead_initialisation(end_times=valid, days_ahead=days_ahead)
    columns: dict[str, object] = {
        "time_series_id": np.full(len(index), features.table.time_series_id, dtype=np.int32),
        "valid_time": pl.Series(valid).dt.replace_time_zone("UTC"),
        "power_fcst_init_time": pl.Series(init).dt.replace_time_zone("UTC"),
        "lead_hours": ((valid - init) / np.timedelta64(1, "h")).astype(np.float64),
        "seed": np.zeros(len(index), dtype=np.int8),
        "arm": np.full(len(index), arm),
        "filtered": np.full(len(index), filtered),
        "days_ahead": np.full(len(index), days_ahead, dtype=np.int8),
        **{
            column: ordered[:, position].astype(np.float32)
            for position, column in enumerate(QUANTILE_COLUMNS)
        },
        "actual": features.y[index].astype(np.float32),
        "mean": mean.astype(np.float32),
        "switching_mask": features.switching_mask[index],
    }
    for name, values in (extra or {}).items():
        columns[name] = values.astype(np.float32)
    return pl.DataFrame(columns)


def store_arm(*, name: str, frame: pl.DataFrame, row_set_id: str = ROW_SET_ID) -> Path:
    """Write an arm's predictions once; a second call returns the existing folder.

    Args:
        name: The set's name.
        frame: The predictions of every scored substation.
        row_set_id: The row set the predictions cover.

    Returns:
        The set's folder.
    """
    folder = DATA_DIR / "predictions" / name
    if folder.exists():
        return folder
    return write_predictions(
        experiment_id=EXPERIMENT_ID,
        name=name,
        frame=frame,
        kind="quantiles",
        row_set_id=row_set_id,
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
