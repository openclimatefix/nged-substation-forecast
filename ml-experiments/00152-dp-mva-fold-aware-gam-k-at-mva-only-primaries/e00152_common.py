"""Constants shared by the scripts of experiment 152 (the fold-aware GAM-K at the MVA Primaries).

Substations appear by D-label only. Experiment 110's cache and stored predictions are read from
their private folders and never rewritten.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
from studies.dp_features import FeatureCache, default_input_paths, load_or_build_feature_cache
from studies.dp_gam import QUANTILE_LEVELS
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

EXPERIMENT_ID: Final[int] = 152
SLUG: Final[str] = "dp-mva-fold-aware-gam-k-at-mva-only-primaries"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
EXPERIMENT_110_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00110-differentiable-physics-5-embedded-generation-and-capacity"
)
CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
"""Part 4's feature cache, shared by experiments 110, 124, and 136 and read only here."""
QUANTILE_LEVELS_ARRAY: Final[tuple[float, ...]] = QUANTILE_LEVELS

MVA_ONLY_PRIMARIES: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
"""The 8 Primaries whose series is metered only in MVA."""
SUNNY_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)
"""April to September, as experiment 56."""
MIDDAY_FIRST_HALF_HOUR_UTC: Final[float] = 10.0
MIDDAY_LAST_HALF_HOUR_UTC: Final[float] = 13.5
"""The half-hours that start between 10:00 and 13:30 UTC, as experiment 56."""
CLEAR_SKY_INDEX_MINIMUM: Final[float] = 0.7
"""A sunny half-hour has CAMS irradiance at least this share of the Haurwitz clear-sky value."""


def load_cache() -> FeatureCache:
    """Return the shared feature cache, building and saving it on the first call."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)


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


# --- The ensemble-driven engine (copied from experiments 111 and 136) -----------------------------

ENS_SOURCE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00111-differentiable-physics-6-ens-forecast-engine-against-bg2"
    / "work"
)
"""Experiment 111's private folder: the MW sites' ensemble stores and calibrations, read only."""
OWN_STORE_DIR: Final[Path] = WORK_DIR / "ens_store"
OWN_CALIBRATION_DIR: Final[Path] = WORK_DIR / "calibration"
FRAME_FOLDERS: Final[tuple[Path, ...]] = (
    ENS_SOURCE_DIR / "frames_demand",
    ML_EXPERIMENTS_DATA_DIR
    / "00048-weekly-retraining-confirmation-and-extension-demand-generators"
    / "work"
    / "frames_demand",
    REPO_DATA_DIR / "studies/per_study/stage1_pilot/demand/ens_every5d/val",
)
POOL_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00133-differentiable-physics-vk-learned-noise-and-observation-variance-in-gam-k"
    / "work"
    / "select"
    / "pool.json"
)
"""Experiment 133's shared filter cell (POOL): log q, logit phi, log sigma z, log sigma eps."""

DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
"""The five MW decision Primaries, where the signed truth is known."""
FIRST_RUN: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
LAST_RUN: Final[datetime] = datetime(2026, 6, 30, tzinfo=UTC)
PUBLICATION_DELAY: Final[timedelta] = timedelta(hours=9)
SCORED_FIRST_RUN: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
EVERY5D_STRIDE_DAYS: Final[int] = 5
TUNING_MONTHS: Final[frozenset[tuple[int, int]]] = frozenset(
    {(2024, 7), (2024, 9), (2024, 11), (2025, 1), (2025, 3), (2025, 5)}
)
REPLICATE_MONTHS: Final[frozenset[tuple[int, int]]] = frozenset(
    {(2024, 8), (2024, 10), (2024, 12), (2025, 2), (2025, 4), (2025, 6)}
)
MIN_TRAINING_DAYS: Final[int] = 90
SEASON_YEARS: Final[range] = range(2019, 2027)
RIDGE: Final[float] = 0.1
WINDOW_MONTHS: Final[int] = 24
D8_FIRST_DAY: Final[datetime] = datetime(2025, 4, 1, tzinfo=UTC)
"""Marsh Lane (D8) enters training and scoring on this day, when its series changed meaning."""


def frame_path(*, run: datetime) -> Path:
    """Return the path of a run's ensemble frame in the first folder that holds it."""
    name = f"init={run:%Y-%m-%d}.parquet"
    for folder in FRAME_FOLDERS:
        if (folder / name).exists():
            return folder / name
    msg = f"no frame for the run of {run:%Y-%m-%d}"
    raise FileNotFoundError(msg)


def run_dates() -> list[datetime]:
    """Return every daily 00:00 UTC run from 2024-04-01 to 2026-06-30."""
    return [FIRST_RUN + timedelta(days=i) for i in range((LAST_RUN - FIRST_RUN).days + 1)]


def monthly_cuts(*, last: datetime = datetime(2026, 6, 1, tzinfo=UTC)) -> list[datetime]:
    """Return the first-of-the-month 00:00 UTC cuts from 2024-04-01 to `last`."""
    cuts = []
    year, month = FIRST_RUN.year, FIRST_RUN.month
    while (cut := datetime(year, month, 1, tzinfo=UTC)) <= last:
        cuts.append(cut)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return cuts


def origin_of(*, run: datetime) -> datetime:
    """Return a run's `power_fcst_init_time`: its initialisation plus the publication delay."""
    return run + PUBLICATION_DELAY


def period_of_run(*, run: datetime) -> str:
    """Return `tuning`, `replicate`, `scored`, `stride`, or `warm_up` for a run.

    A run belongs to the month of its `power_fcst_init_time`. `stride` is a run of the scored year
    on the every-fifth-day stride, never scored.
    """
    origin = origin_of(run=run)
    key = (origin.year, origin.month)
    if key in TUNING_MONTHS:
        return "tuning"
    if key in REPLICATE_MONTHS:
        return "replicate"
    if run >= SCORED_FIRST_RUN and run <= LAST_RUN:
        return "scored" if (run - SCORED_FIRST_RUN).days % EVERY5D_STRIDE_DAYS != 0 else "stride"
    return "warm_up"
