"""Paths and constants shared by the scripts of experiment 191."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

from studies.dp_features import FeatureCache, default_input_paths, load_or_build_feature_cache
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 191
SLUG: Final[str] = "dp-price-sensitive-assets-in-the-history"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"

PUBLIC_185: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00185-dp-price-dfs-national-signals-past-demand" / "public"
)
PUBLIC_186: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00186-switching-d8-gas-peaker-spark-spread-and-triads" / "public"
)
PUBLIC_177: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00177-dp-worst-days-other-data-sources" / "public"
)
PRICES_PATH: Final[Path] = WORK_DIR / "prices.parquet"
EFFICIENCY: Final[float] = 0.40
"""Electrical efficiency of the central spark spread, as in experiment 186."""
VARIABLE_COST_GBP_PER_MWH: Final[float] = 5.0
MAX_PLAUSIBLE_GAS_P_PER_KWH: Final[float] = 15.0
MIN_PLAUSIBLE_GAS_P_PER_KWH: Final[float] = 0.5
PRICE_PERCENTILES: Final[tuple[float, ...]] = (0.5, 0.75, 0.9)
FEATURE_SETS: Final[dict[str, tuple[str, ...]]] = {
    "halfhourly": ("price_p50", "price_p75", "price_p90", "spread_pos"),
    "daily": ("day_p50", "day_p75", "day_p90", "day_spread_pos"),
}
"""The pre-registered half-hourly indicators, and the post hoc settlement-day indicators."""
FEATURE_NAMES: Final[tuple[str, ...]] = FEATURE_SETS["halfhourly"]

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

ASSET_SITES: Final[tuple[str, ...]] = ("D8", "D14")
NO_HARM_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D16")
ALL_SITES: Final[tuple[str, ...]] = (*ASSET_SITES, *NO_HARM_SITES)
DONORS: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D19", "D21", "D22")
CONNECTED: Final[dict[str, datetime]] = {
    "D8": datetime(2023, 6, 13, tzinfo=UTC),
    "D14": datetime(2023, 6, 13, tzinfo=UTC),
    "D5": datetime(2021, 3, 11, tzinfo=UTC),
}
"""Connection date of the registered gas engine under each Primary (August 2026 ECR). D8 is gated
on from the engine's date because the engine may feed D8 through the network arrangement."""
D8_FIRST_DAY: Final[datetime] = datetime(2025, 4, 1, tzinfo=UTC)

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
MASK_SHARE_OF_MAXIMUM: Final[float] = 0.25
"""Arm T drops a training half-hour when H's fitted price output there exceeds this share of the
largest fitted price output in the fit, provided that largest output is at least
`MASK_MINIMUM_POINTS`."""
MASK_MINIMUM_POINTS: Final[float] = 1.0
"""One point is 1% of the site's effective capacity (0.1 MW at D8). A smaller fitted price effect
is not material, and masking on it would drop half the training rows."""
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
