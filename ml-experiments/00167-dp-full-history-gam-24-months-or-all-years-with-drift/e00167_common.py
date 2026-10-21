"""Constants and helpers shared by the scripts of experiment 167 (full-history GAM, T-all).

The experiment refits the convex model with the heating split by day type (T-all) at 16 quarterly
cuts (four burn-in cuts whose scored windows only feed the spread pools, then 12 scored origins from
2023-07-01 to 2026-04-01), on past weather only (ERA5 and CAMS), and scores each fit on the 3 months
after its cut. The helpers are copies of experiment 107's. Each job (one substation at one cut)
writes its own parquet file, atomically, so a crashed worker is rerun without touching the others.
"""

import calendar
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.dp_features import FeatureCache, default_input_paths, load_or_build_feature_cache
from studies.dp_gam import QUANTILE_LEVELS
from studies.experiment_store import HOLDOUT_START
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 167
SLUG: Final[str] = "dp-full-history-gam-24-months-or-all-years-with-drift"
ROW_SET_ID: Final[str] = "dp_mw_v1"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
"""Small result files (per-job fits, tuning, reports); private and never committed."""
CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00104-differentiable-physics-2-simple-structural-model" / "cache"
)
"""Experiment 104's feature cache: it is content-addressed by the inputs' hash, so it is shared."""
TUNED_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00104-differentiable-physics-2-simple-structural-model"
    / "work"
    / "tuned.json"
)
"""Part 2's tuned penalties (ridge 0.1, drift step 3,000), used unchanged."""
BASE_THRESHOLD_C: Final[float] = 15.5
"""The heating hinge's threshold, as in experiments 114 and 136."""
T_ALL_LAMBDA: Final[float] = 1.0
"""T-all's difference-penalty weight, chosen by experiment 114."""
MIN_HISTORY_MONTHS: Final[int] = 24
"""A site is scored at an origin only if it has at least this many months of usable rows before the
origin, so the 24-month arm trains on a full window and the arms differ in what they train on."""

DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
"""The five sites that carry the verdicts."""
DESCRIPTIVE_SITES: Final[tuple[str, ...]] = ("D9",)
"""D9's data start on 2023-08-23, so under the history rule it is scored at three origins only
(2025-10-01 onwards) and is reported descriptively. D18's data start on 2024-12-11, too late."""
ALL_SITES: Final[tuple[str, ...]] = DECISION_SITES + DESCRIPTIVE_SITES
MIN_FIT_MONTHS: Final[int] = 12
"""A site is fitted at a cut once it has this many months of usable rows; a fit before the scored
history rule is met only feeds a spread pool."""

SEASON_YEARS: Final[range] = range(2019, 2026)
SCORED_MONTHS: Final[int] = 3
POOL_MONTHS: Final[int] = 12
"""A spread pool is the arm's own out-of-sample residuals over the 12 months before the origin."""

D10_JUMP_WINDOWS: Final[tuple[tuple[datetime, datetime], ...]] = (
    (datetime(2024, 11, 8, tzinfo=UTC), datetime(2024, 12, 2, tzinfo=UTC)),
    (datetime(2025, 11, 17, tzinfo=UTC), datetime(2025, 12, 15, tzinfo=UTC)),
)
"""D10's two unlabelled jumps, 2024-11-08 to 2024-12-01 and 2025-11-17 to 2025-12-14, inclusive."""


def add_months(*, moment: datetime, months: int) -> datetime:
    """Return `moment` shifted by whole calendar months, clamping the day of month."""
    total = moment.year * 12 + (moment.month - 1) + months
    year, month = total // 12, total % 12 + 1
    return moment.replace(
        year=year, month=month, day=min(moment.day, calendar.monthrange(year, month)[1])
    )


CUTS: Final[tuple[datetime, ...]] = tuple(
    add_months(moment=datetime(2022, 7, 1, tzinfo=UTC), months=3 * step) for step in range(16)
)
"""All 16 cuts: 2022-07-01 to 2026-04-01 in steps of 3 months."""
BURN_IN_CUTS: Final[tuple[datetime, ...]] = CUTS[:4]
"""Their scored windows (2022-07 to 2023-06) only feed the first scored origins' spread pools."""
SCORED_ORIGINS: Final[tuple[datetime, ...]] = CUTS[4:]
"""The 12 scored origins, 2023-07-01 to 2026-04-01."""
WINTER_QUARTER_ORIGINS: Final[tuple[datetime, ...]] = tuple(
    cut for cut in SCORED_ORIGINS if cut.month in (10, 1)
)
"""Origins whose scored window is October to March (cuts on 10-01 and 01-01)."""
POST_TUNING_ORIGINS: Final[tuple[datetime, ...]] = SCORED_ORIGINS[-4:]
OUT_OF_TUNING_ORIGINS: Final[tuple[datetime, ...]] = SCORED_ORIGINS[:4] + SCORED_ORIGINS[-4:]


def origin_year(*, cut: datetime) -> str:
    """Return the origin year of a scored origin, July to June: `2023-24`, `2024-25`, `2025-26`."""
    start = cut.year if cut.month >= 7 else cut.year - 1
    return f"{start}-{str(start + 1)[2:]}"


def causality_group(*, cut: datetime) -> str:
    """Return Part 2's penalty causality for a scored origin: pre-, in-, or post-tuning."""
    return {"2023-24": "pre-tuning", "2024-25": "in-tuning", "2025-26": "post-tuning"}[
        origin_year(cut=cut)
    ]


def quarter_type(*, cut: datetime) -> str:
    """Return `winter` for a window of October to March, `summer` for April to September."""
    return "winter" if cut.month in (10, 1) else "summer"


def cut_label(*, cut: datetime) -> str:
    """Return a cut's date as `YYYY-MM-DD`."""
    return cut.date().isoformat()


def history_months(*, cache: FeatureCache, site: str, cut: datetime) -> float:
    """Return the months from a site's first usable row to the cut (30.4375-day months)."""
    features = cache.sites[site]
    usable = np.flatnonzero(features.present & np.isfinite(features.y))
    return float((to_64(moment=cut) - cache.end_times[usable[0]]) / np.timedelta64(1, "D")) / 30.4375


def sites_for_cut(*, cache: FeatureCache, cut: datetime) -> tuple[str, ...]:
    """Return the sites fitted at a cut: those with at least `MIN_FIT_MONTHS` of usable history."""
    return tuple(
        site
        for site in ALL_SITES
        if history_months(cache=cache, site=site, cut=cut) >= MIN_FIT_MONTHS
    )


def is_scored(*, cache: FeatureCache, site: str, cut: datetime) -> bool:
    """Return True if the site has `MIN_HISTORY_MONTHS` of usable history at the cut."""
    return history_months(cache=cache, site=site, cut=cut) >= MIN_HISTORY_MONTHS


def load_cache() -> FeatureCache:
    """Return the feature cache, building and saving it on the first call."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)


def to_64(*, moment: datetime) -> np.datetime64:
    """Return a timezone-aware datetime as a naive UTC `datetime64[us]`."""
    return np.datetime64(moment.replace(tzinfo=None), "us")


def window_end(*, cut: datetime) -> datetime:
    """Return the end of the window a fit at `cut` is scored on."""
    return add_months(moment=cut, months=SCORED_MONTHS)


def rows_between(
    *, cache: FeatureCache, site: str, after: datetime, through: datetime
) -> np.ndarray:
    """Return the grid rows that have a reading and end in (`after`, `through`].

    Args:
        cache: The cache.
        site: The substation's label.
        after: Rows must end after this instant.
        through: Rows must end no later than this instant, nor in the held-out period.

    Returns:
        Sorted grid positions.
    """
    features = cache.sites[site]
    end = cache.end_times
    last = min(through, HOLDOUT_START - timedelta(minutes=30))
    keep = (
        features.present
        & np.isfinite(features.y)
        & (end > to_64(moment=after))
        & (end <= to_64(moment=last))
    )
    return np.flatnonzero(keep)


def scoring_index(*, cache: FeatureCache, site: str, cut: datetime) -> np.ndarray:
    """Return the rows a fit at `cut` is scored on: the 3 months after it."""
    return rows_between(cache=cache, site=site, after=cut, through=window_end(cut=cut))


def jump_window_mask(*, cache: FeatureCache, index: np.ndarray) -> np.ndarray:
    """Return True where a row ends inside one of D10's two unlabelled-jump windows.

    Each window runs to the end of its last day, inclusive.
    """
    end = cache.end_times[index]
    inside = np.zeros(len(index), dtype=bool)
    for start, stop in D10_JUMP_WINDOWS:
        inside |= (end > to_64(moment=start)) & (end <= to_64(moment=stop))
    return inside


def d10_masked_cache(*, cache: FeatureCache) -> FeatureCache:
    """Return the cache with D10's two jump windows removed from its training and scoring rows."""
    from dataclasses import replace

    features = cache.sites["D10"]
    inside = jump_window_mask(cache=cache, index=np.arange(len(cache.end_times)))
    masked = replace(
        features,
        default_rows=features.default_rows & ~inside,
        switching_mask=features.switching_mask | inside,
    )
    return replace(cache, sites={**cache.sites, "D10": masked})


def base_frame(
    *,
    cache: FeatureCache,
    site: str,
    cut: datetime,
    index: np.ndarray,
    extra: dict[str, np.ndarray],
) -> pl.DataFrame:
    """Return the identifying columns of one job's rows, plus the arm columns in `extra`.

    Args:
        cache: The cache.
        site: The substation's label.
        cut: The fit's cut, which is the forecast's initialisation time.
        index: Grid rows predicted.
        extra: Further columns, each of length `len(index)`.

    Returns:
        Columns `site`, `time_series_id`, `valid_time`, `power_fcst_init_time`, `origin`,
        `lead_hours`, `actual`, `switching_mask`, `jump_window`, then `extra`.
    """
    features = cache.sites[site]
    valid = cache.end_times[index]
    lead = ((valid - to_64(moment=cut)) / np.timedelta64(1, "h")).astype(np.float64)
    return pl.DataFrame(
        {
            "site": np.full(len(index), site),
            "time_series_id": np.full(len(index), features.table.time_series_id, dtype=np.int32),
            "valid_time": pl.Series(valid).dt.replace_time_zone("UTC"),
            "power_fcst_init_time": np.full(len(index), to_64(moment=cut)),
            "origin": np.full(len(index), cut_label(cut=cut)),
            "lead_hours": lead,
            "actual": features.y[index].astype(np.float64),
            "switching_mask": features.switching_mask[index],
            "jump_window": jump_window_mask(cache=cache, index=index)
            if site == "D10"
            else np.zeros(len(index), dtype=bool),
            **extra,
        }
    ).with_columns(power_fcst_init_time=pl.col("power_fcst_init_time").dt.replace_time_zone("UTC"))


def write_atomic(*, frame: pl.DataFrame, path: Path) -> None:
    """Write a parquet file through a temporary name, so a crashed worker leaves no partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.write_parquet(temporary)
    temporary.replace(path)


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
