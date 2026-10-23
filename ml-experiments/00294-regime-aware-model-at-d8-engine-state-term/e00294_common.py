"""Paths and constants shared by the scripts of experiment 294 (copied from experiment 191)."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

from studies.dp_features import FeatureCache, default_input_paths, load_or_build_feature_cache
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 294
SLUG: Final[str] = "regime-aware-model-at-d8-engine-state-term"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"

CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
"""Part 4's feature cache, shared because it is keyed by a hash of its inputs, read only."""
E111_WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00111-differentiable-physics-6-ens-forecast-engine-against-bg2"
    / "work"
)
E152_WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00152-dp-mva-fold-aware-gam-k-at-mva-only-primaries" / "work"
)
POOL_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00133-differentiable-physics-vk-learned-noise-and-observation-variance-in-gam-k"
    / "work"
    / "select"
    / "pool.json"
)
"""Experiment 133's shared filter cell (POOL): log q, logit phi, log sigma z, log sigma eps."""
PART_4_TUNED: Final[Path] = CACHE_DIR.parent / "work" / "tuned.json"
PREDS_DIR: Final[Path] = WORK_DIR / "preds"
FITS_DIR: Final[Path] = WORK_DIR / "fits"

ENGINE_SITE: Final[str] = "D8"
CONTROL_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11")
ALL_SITES: Final[tuple[str, ...]] = (ENGINE_SITE, *CONTROL_SITES)
DONORS: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D19", "D21", "D22")
BREAK: Final[datetime] = datetime(2023, 8, 15, tzinfo=UTC)
"""The day NGED stopped adding the engine's output to D8's reading (experiment 194)."""
D8_FIRST_DAY: Final[datetime] = datetime(2025, 4, 1, tzinfo=UTC)
"""Experiment 191's first D8 day (the series changed meaning after 8.91 MW of solar connected)."""
STATE_GRID_START: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
UNIT_MW: Final[float] = 1.97
"""One engine unit, from the engine's recorded output levels (experiment 194)."""

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
N_MEMBERS: Final[int] = 51
MIN_WINDOW_HALF_HOURS: Final[int] = 48


def load_cache() -> FeatureCache:
    """Return the shared feature cache, building and saving it on the first call."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)


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
    if SCORED_FIRST_RUN <= run <= LAST_RUN:
        return "scored" if (run - SCORED_FIRST_RUN).days % EVERY5D_STRIDE_DAYS != 0 else "stride"
    return "warm_up"
