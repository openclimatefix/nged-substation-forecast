"""Paths, the 16 Primaries, the unmasked GAM_ONLY monthly forecast, and daily residual helpers.

Copies the patterns of experiments 274 and 290: GAM_ONLY (no lagged power, 24-month window,
ridge 0.1) is refitted at the first of each month and predicts the whole following month with the
realised weather. Unlike those experiments, the fit is unmasked: the hand-filtered switching rows
train too, because a masked fit flatters the residual.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
from studies.dp_features import (
    FeatureCache,
    default_input_paths,
    load_or_build_feature_cache,
    naive_utc,
)
from studies.dp_gam import GamSpec, TrainingWindow, fit_gam, predict_gam
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00298-lsc-long-deep-detector-and-refit-cleaning"
)
PANEL_PATH: Final[Path] = EXPERIMENT_DIR / "clean_panel.parquet"
WORLDS_DIR: Final[Path] = EXPERIMENT_DIR / "worlds"
CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
SITES: Final[tuple[str, ...]] = (
    "D2",
    "D3",
    "D4",
    "D5",
    "D8",
    "D9",
    "D10",
    "D11",
    "D12",
    "D14",
    "D15",
    "D16",
    "D18",
    "D19",
    "D21",
    "D22",
)
INJECT_SITES: Final[tuple[str, ...]] = (
    "D2",
    "D3",
    "D4",
    "D5",
    "D10",
    "D11",
    "D12",
    "D19",
    "D21",
    "D22",
)
"""The 10 Primaries of experiments 274 and 290, which have long histories and no generator."""
SERIES_TO_SITE: Final[dict[int, str]] = {
    3: "D3",
    7: "D9",
    8: "D11",
    9: "D19",
    10: "D10",
    11: "D4",
    12: "D15",
    13: "D18",
    14: "D12",
    15: "D22",
    16: "D14",
    17: "D5",
    18: "D16",
    24: "D21",
    25: "D2",
    26: "D8",
}
ADJACENCY_SERIES: Final[dict[int, tuple[int, ...]]] = {
    3: (15, 16, 26),
    7: (24,),
    8: (10, 14, 24),
    9: (11, 24),
    10: (8, 12, 13, 14, 24, 25),
    11: (9, 25),
    12: (10, 13, 14, 16, 25),
    13: (10, 12, 14, 16, 25),
    14: (8, 10, 12, 13, 15, 16),
    15: (3, 14, 17, 18),
    16: (3, 12, 13, 14, 26),
    17: (15, 18, 26),
    18: (15, 17),
    24: (7, 8, 9, 10, 25),
    25: (10, 11, 12, 13, 24),
    26: (3, 16, 17),
}
"""Experiment 43's adjacency of the 16 metered series, as series ids."""
NEIGHBOURS: Final[dict[str, tuple[str, ...]]] = {
    SERIES_TO_SITE[a]: tuple(SERIES_TO_SITE[b] for b in bs) for a, bs in ADJACENCY_SERIES.items()
}
SEASON_YEARS: Final[range] = range(2019, 2027)
RIDGE: Final[float] = 0.1
WINDOW: Final[TrainingWindow] = TrainingWindow(months=24)
FIRST_CUT: Final[datetime] = datetime(2020, 1, 1, tzinfo=UTC)
LAST_CUT: Final[datetime] = datetime(2026, 6, 1, tzinfo=UTC)
MIN_ROWS_PER_DAY: Final[int] = 36
"""A day needs 36 of 48 half-hours to have a daily mean."""


def load_cache() -> FeatureCache:
    """Return the feature cache with every site's training mask unmasked."""
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)
    sites = {}
    for name, features in cache.sites.items():
        unmasked = features.default_rows | (
            features.in_pool & ~features.lockdown & features.switching_mask
        )
        sites[name] = replace(features, default_rows=unmasked)
    return replace(cache, sites=sites)


def add_months(*, moment: datetime, months: int) -> datetime:
    """Return the first of the month `months` after `moment`'s month."""
    index = moment.year * 12 + moment.month - 1 + months
    return datetime(index // 12, index % 12 + 1, 1, tzinfo=UTC)


def month_starts(*, first: datetime, last: datetime) -> list[datetime]:
    """Return the first of each month from `first` to `last` inclusive."""
    out, cut = [], first
    while cut <= last:
        out.append(cut)
        cut = add_months(moment=cut, months=1)
    return out


def target_rows(*, cache: FeatureCache, site: str, cut: datetime, stop: datetime) -> np.ndarray:
    """Return the scored grid rows from `cut` to `stop`: a reading and finite model inputs."""
    features = cache.sites[site]
    first, last = naive_utc(moment=cut), naive_utc(moment=stop)
    keep = (
        (cache.end_times > first)
        & (cache.end_times <= last)
        & features.present
        & np.isfinite(features.y)
        & np.isfinite(features.pv_unit_shape)
        & np.isfinite(features.smoothed_temperature_c).all(axis=1)
    )
    return np.flatnonzero(keep)


def forecast_month(
    *, cache: FeatureCache, site: str, cut: datetime, index: np.ndarray
) -> np.ndarray | None:
    """Fit GAM_ONLY at `cut` and predict rows `index` in points; `None` if no row can train."""
    try:
        fit = fit_gam(
            cache=cache,
            site=site,
            specs=[GamSpec(ridge=RIDGE, drift="none", capacity="fitted", final_knot="free")],
            cut=cut,
            window=WINDOW,
            season_years=SEASON_YEARS,
        )
    except ValueError:
        return None
    return predict_gam(cache=cache, fit=fit, index=index)


def utc_day_of(*, cache: FeatureCache, index: np.ndarray) -> np.ndarray:
    """Return the UTC date (as `datetime64[D]`) of the start of each half-hour in `index`."""
    start = cache.end_times[index] - np.timedelta64(30, "m")
    return start.astype("datetime64[D]")


def daily_means(*, days: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (unique days, mean of `values`) for days with at least `MIN_ROWS_PER_DAY` rows."""
    unique, inverse, counts = np.unique(days, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=values)
    keep = counts >= MIN_ROWS_PER_DAY
    return unique[keep], (sums / counts)[keep]


def forecast_many(
    *, cache: FeatureCache, site: str, cut: datetime, indexes: list[np.ndarray]
) -> list[np.ndarray] | None:
    """Fit GAM_ONLY once at `cut` and predict several row sets; `None` if no row can train."""
    try:
        fit = fit_gam(
            cache=cache,
            site=site,
            specs=[GamSpec(ridge=RIDGE, drift="none", capacity="fitted", final_knot="free")],
            cut=cut,
            window=WINDOW,
            season_years=SEASON_YEARS,
        )
    except ValueError:
        return None
    return [predict_gam(cache=cache, fit=fit, index=index) for index in indexes]
