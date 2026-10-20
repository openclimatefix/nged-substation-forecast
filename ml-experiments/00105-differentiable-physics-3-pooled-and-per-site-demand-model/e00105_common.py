"""Constants and helpers shared by the scripts of experiment 105 (differentiable physics, part 3).

The experiment fits the convex model of `studies.dp_gam` and lagless XGBoost per substation at
three cuts: 2024-07-01 (tuning), 2025-07-01 (the scored year), and 2022-11-01 (the cold block). It
writes each arm's quantiles as a prediction set and scores them with the quantile score, twice the
mean pinball loss over the 11 levels of experiment 87, in points of effective capacity.

**Lead.** A frozen fit on past weather does not depend on an initialisation, so each valid time is
written once, against a day-ahead initialisation at 09:00 UTC on the previous day (leads of 15.5 to
39 hours). The lagged reference arms use their own initialisations (`SCHEMES`), because lags need
one.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, Literal

import numpy as np
import polars as pl
from studies.dp_features import (
    PART_1_DIR,
    FeatureCache,
    default_input_paths,
    load_or_build_feature_cache,
)
from studies.dp_gam import QUANTILE_COLUMNS, QUANTILE_LEVELS
from studies.dp_pooled import PooledBatch, build_pooled_batch
from studies.experiment_store import HOLDOUT_START, write_predictions
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 105
SLUG: Final[str] = "differentiable-physics-3-pooled-and-per-site-demand-model"
ROW_SET_ID: Final[str] = "dp_mw_v1"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
PART_2_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00104-differentiable-physics-2-simple-structural-model"
)
CACHE_DIR: Final[Path] = PART_2_DIR / "cache"
"""Part 2's feature cache, reused: the inputs and the cut dates are the same."""
WORK_DIR: Final[Path] = DATA_DIR / "work"
"""Small result files (tuning, capacities, weights, reports); private and never committed."""

PART_1_ROWS: Final[Path] = PART_1_DIR / "rows.parquet"
"""Experiment 103's row table (scarce-data kinds, masks, day types)."""

SCORED_SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
"""The six MW long-history substations that are scored."""
ELIGIBLE_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
"""The sites counted for the decision; D9 holds 9.0 months at 2024-07-01."""
COASTAL_SITES: Final[tuple[str, ...]] = ("D5", "D9")
"""Possible sunshine-tracking loads (PVDIS), reported separately."""
COLD_SITES: Final[tuple[str, ...]] = ELIGIBLE_SITES

TUNING_CUT: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
FINAL_CUT: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
COLD_CUT: Final[datetime] = datetime(2022, 11, 1, tzinfo=UTC)
TUNING_END: Final[datetime] = FINAL_CUT
SCORING_END: Final[datetime] = HOLDOUT_START
COLD_END: Final[datetime] = datetime(2023, 3, 1, tzinfo=UTC)
SEASON_YEARS: Final[range] = range(2019, 2026)

SchemeType = Literal["day_ahead", "d1", "d6"]
"""How a row's initialisation time is set: 09:00 UTC the day before; 00:00 UTC one day before the
valid day (leads of 24 to 48 hours); 00:00 UTC six days before (leads of 144 to 168 hours)."""

SCHEMES: Final[dict[str, tuple[int, int]]] = {
    "day_ahead": (1, 9),
    "d1": (1, 0),
    "d6": (6, 0),
}
"""Days before the valid day and the UTC hour of the initialisation, for each scheme."""

LAGGED_SCHEMES: Final[tuple[SchemeType, ...]] = ("d1", "d6")
TELEMETRY_DELAY_STEPS: Final[int] = 7
"""A reading is usable once it is 3.5 hours old: 7 half-hours."""
STEPS_PER_DAY: Final[int] = 48
STEPS_PER_WEEK: Final[int] = 7 * STEPS_PER_DAY


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


def initialisation_times(*, end_times: np.ndarray, scheme: SchemeType) -> np.ndarray:
    """Return each valid time's initialisation time under a scheme.

    Args:
        end_times: Half-hour end stamps as `datetime64[us]`.
        scheme: How the initialisation is set.

    Returns:
        `datetime64[us]` initialisation times: the UTC day of the half-hour's start, minus the
        scheme's days, at the scheme's hour.
    """
    days_before, hour = SCHEMES[scheme]
    day = (end_times - np.timedelta64(30, "m")).astype("datetime64[D]")
    return (day - np.timedelta64(days_before, "D")).astype("datetime64[us]") + np.timedelta64(
        hour, "h"
    )


def prediction_frame(
    *,
    cache: FeatureCache,
    site: str,
    index: np.ndarray,
    quantiles: np.ndarray,
    scheme: SchemeType = "day_ahead",
) -> pl.DataFrame:
    """Return one substation's predictions as rows with the columns the scorers read.

    Args:
        cache: The cache.
        site: The substation's label.
        index: Grid rows predicted.
        quantiles: Shape (len(index), 11), the 11 quantiles in points.
        scheme: How the initialisation time is set.

    Returns:
        Columns `time_series_id`, `valid_time`, `power_fcst_init_time`, `lead_hours`, `seed`,
        `q01` to `q99`, `actual`, and `switching_mask`.

    Raises:
        ValueError: If the quantiles' shape is wrong or they cross.
    """
    features = cache.sites[site]
    if quantiles.shape != (len(index), len(QUANTILE_LEVELS)):
        expected = (len(index), len(QUANTILE_LEVELS))
        msg = f"expected quantiles of shape {expected}, got {quantiles.shape}"
        raise ValueError(msg)
    ordered = np.sort(quantiles, axis=1)
    valid = cache.end_times[index]
    init = initialisation_times(end_times=valid, scheme=scheme)
    return pl.DataFrame(
        {
            "time_series_id": np.full(len(index), features.table.time_series_id, dtype=np.int32),
            "valid_time": pl.Series(valid).dt.replace_time_zone("UTC"),
            "power_fcst_init_time": pl.Series(init).dt.replace_time_zone("UTC"),
            "lead_hours": ((valid - init) / np.timedelta64(1, "h")).astype(np.float64),
            "seed": np.zeros(len(index), dtype=np.int8),
            **{
                column: ordered[:, position].astype(np.float32)
                for position, column in enumerate(QUANTILE_COLUMNS)
            },
            "actual": features.y[index].astype(np.float32),
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


# ---- Part 3 additions: the pooled set, the fit settings, and small helpers -----------------------

MW_POOL: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16", "D18")
"""The MW Primaries in the pooled training set (D15 carries a generator and is out)."""
MVA_POOL: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
"""The MVA Primaries, on dark rows only."""
POOL: Final[tuple[str, ...]] = (*MW_POOL, *MVA_POOL)
REPLICA_MONTHS: Final[int] = 12
"""The replica fit is made this many months before the cut."""
WINDOW_MONTHS: Final[int] = 24
"""The 2-year window of every main arm."""
FIT_STEPS: int = 1200
"""Optimiser steps of a pooled fit: the training loss is still falling at 600 steps."""
FIT_RATE: float = 0.15
FINAL_RATE_FRACTION: Final[float] = 0.01
"""The cosine learning-rate schedule ends at this fraction of the starting rate."""
INTERACTION: Final[tuple[int, ...]] = (64, 64)
TIMEOUT_MINUTES: Final[float] = 10.0

SERIES_TABLE: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR.parent / "studies/per_study/stage1_pilot/demand/inputs/series.parquet"
)
VINTAGES: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00045-pvdis-4-weekly-capacity-asymmetric-l1"
    / "capacity_weekly_vintages.parquet"
)
FARMS_FILE: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00045-pvdis-4-weekly-capacity-asymmetric-l1"
    / "farm_half_hourly.parquet"
)


def months_before(*, moment: datetime, months: int) -> datetime:
    """Return `moment` shifted back by whole calendar months (first of the month)."""
    total = moment.year * 12 + moment.month - 1 - months
    return moment.replace(year=total // 12, month=total % 12 + 1, day=1)


def pooled_batch_for(
    *,
    cache: FeatureCache,
    sites: tuple[str, ...],
    cut: datetime,
    months: int | None = WINDOW_MONTHS,
    device: str = "cuda",
) -> PooledBatch:
    """Return the pooled training rows of the `months` months before `cut` (all years if None)."""
    return build_pooled_batch(
        cache=cache,
        sites=sites,
        cut_end=to_64(moment=cut),
        after=None if months is None else to_64(moment=months_before(moment=cut, months=months)),
        device=device,
    )


def scoring_batch_for(
    *,
    cache: FeatureCache,
    sites: tuple[str, ...],
    after: datetime,
    through: datetime,
    device: str = "cuda",
    all_present_rows: bool = False,
) -> PooledBatch:
    """Return the rows in (`after`, `through`] for the sites, at most the last valid time.

    With `all_present_rows` the rows inside switching masks are kept too, for scoring beside the
    default rows.
    """
    return build_pooled_batch(
        cache=cache,
        sites=sites,
        cut_end=to_64(moment=min(through, HOLDOUT_START - timedelta(minutes=30))),
        after=to_64(moment=after),
        device=device,
        all_present_rows=all_present_rows,
    )


def effective_capacity_mw(*, cache: FeatureCache, site: str) -> float:
    """Return a substation's effective capacity in MW."""
    return float(cache.sites[site].table.effective_capacity_mw)


def write_json(*, name: str, payload: object) -> Path:
    """Write a small result file under the experiment's work folder."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    path = WORK_DIR / name
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    return path


REPORT_SITES: Final[tuple[str, ...]] = (*SCORED_SITES, "D18")
"""The scored sites plus D18, the one short-history site, which is reported beside them."""
