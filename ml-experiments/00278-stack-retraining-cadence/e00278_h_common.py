"""Constants and helpers shared by the scripts of experiment 136 (best-guess stage 1 assembly).

The constants, periods, and run dates are experiment 111's, copied so this folder imports only
itself.
The stored fits, ensemble arrays, and feature cache are read from experiment 111's private data
folder; this experiment writes only under its own `WORK_DIR`.

**Three periods, each with one role.** The tuning period is the odd months of 2024-07 to 2025-06
(calibration, solar shrink, filter hyperparameters, spread factors, the XGBoost bars' start
choice), the replicate period is the even months of the same year (the sign of the contrasts), and
the scored year is 2025-07-01 to 2026-06-30 (the 292 runs of `fold_a_daily_gap_v1`). A run belongs
to the month of its `power_fcst_init_time`.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

from studies.dp_features import (
    FeatureCache,
    default_input_paths,
    load_or_build_feature_cache,
)
from studies.dp_gam import GamSpec, TrainingWindow
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 278
SLUG: Final[str] = "stack-retraining-cadence"
SOURCE_SLUG: Final[str] = "differentiable-physics-6-ens-forecast-engine-against-bg2"
ROW_SET_ID: Final[str] = "fold_a_daily_gap_v1"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
E201_FEATURES: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00201-dp-gam-residual-mlp" / "work" / "features"
)
"""Experiment 201's stored per-substation feature frames, read and never written."""
SOURCE_WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00111-{SOURCE_SLUG}" / "work"
STORE_DIR: Final[Path] = SOURCE_WORK_DIR / "ens_store"
FITS_DIR: Final[Path] = SOURCE_WORK_DIR / "fits"
CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
"""Part 4's feature cache, shared because it is keyed by a hash of its inputs."""
PART_4_TUNED: Final[Path] = CACHE_DIR.parent / "work" / "tuned.json"

DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
"""The sites counted for the decision: Phase A needs 4 of 5."""
SCORED_SITES: Final[tuple[str, ...]] = (*DECISION_SITES, "D9")
DESCRIPTIVE_SITES: Final[tuple[str, ...]] = ("D18",)
ALL_SITES: Final[tuple[str, ...]] = (*SCORED_SITES, *DESCRIPTIVE_SITES)
COASTAL_SITES: Final[tuple[str, ...]] = ("D5", "D9")

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
REPLICATE_SIGN_MONTHS: Final[frozenset[tuple[int, int]]] = frozenset(
    {(2025, 2), (2025, 4), (2025, 6)}
)
"""The replicate months with 10 or more months of ENS rows for XGBoost; the sign is read here."""
START_CHOICE_MONTHS: Final[frozenset[tuple[int, int]]] = frozenset(
    {(2025, 1), (2025, 3), (2025, 5)}
)
"""The tuning months on which the bars' start choice is read."""

MIN_TRAINING_DAYS: Final[int] = 90
"""A site's first fit has at least this many days of usable rows."""
SEASON_YEARS: Final[range] = range(2019, 2027)
HORIZON_HALF_HOURS: Final[int] = 336
PROXY_WARM_UP_RUNS: Final[int] = 9
"""Runs before this index lack a week of analysis proxy before their origin."""


def load_cache() -> FeatureCache:
    """Return the feature cache, building and saving it on the first call."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)


def monthly_cuts(*, last: datetime = datetime(2026, 6, 1, tzinfo=UTC)) -> list[datetime]:
    """Return the first-of-the-month 00:00 UTC cuts from 2024-04-01 to `last`."""
    cuts = []
    year, month = FIRST_RUN.year, FIRST_RUN.month
    while (cut := datetime(year, month, 1, tzinfo=UTC)) <= last:
        cuts.append(cut)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return cuts


def run_dates() -> list[datetime]:
    """Return every daily 00:00 UTC run from 2024-04-01 to 2026-06-30."""
    return [FIRST_RUN + timedelta(days=i) for i in range((LAST_RUN - FIRST_RUN).days + 1)]


def origin_of(*, run: datetime) -> datetime:
    """Return a run's `power_fcst_init_time`: its initialisation plus the publication delay."""
    return run + PUBLICATION_DELAY


def month_of_origin(*, run: datetime) -> tuple[int, int]:
    """Return the (year, month) of a run's `power_fcst_init_time`."""
    origin = origin_of(run=run)
    return origin.year, origin.month


def is_scored_run(*, run: datetime) -> bool:
    """Return True for the 292 scored runs: daily runs of the scored year off the 5-day stride."""
    if not SCORED_FIRST_RUN <= run <= LAST_RUN:
        return False
    return (run - SCORED_FIRST_RUN).days % EVERY5D_STRIDE_DAYS != 0


def period_of_run(*, run: datetime) -> str:
    """Return `tuning`, `replicate`, `scored`, `stride`, or `warm_up` for a run.

    `stride` is a run of the scored year on the every-fifth-day stride, which chose BG2's blend and
    offset and is never scored; `warm_up` is a run before 2024-07 or in no named period.
    """
    key = month_of_origin(run=run)
    if key in TUNING_MONTHS:
        return "tuning"
    if key in REPLICATE_MONTHS:
        return "replicate"
    if run >= SCORED_FIRST_RUN:
        return "scored" if is_scored_run(run=run) else "stride"
    return "warm_up"


def write_json_atomic(*, path: Path, content: dict) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True))
    temporary.rename(path)


RIDGE: Final[float] = 0.1
"""Experiment 104's tuned ridge penalty, as in experiments 111 and 114."""
WINDOW_MONTHS: Final[int] = 24
BASE_THRESHOLD_C: Final[float] = 15.5
T_ALL_LAMBDA: Final[float] = 1.0
"""T-all's difference-penalty weight, chosen by experiment 114 (`decisions.json`)."""
POOL_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00133-differentiable-physics-vk-learned-noise-and-observation-variance-in-gam-k"
    / "work"
    / "select"
    / "pool.json"
)
"""Experiment 133's single shared filter cell (POOL)."""


def base_spec() -> GamSpec:
    """Return B0's penalties: GAM-K's ridge, no drift, and the capacity pinned to the vintage."""
    return GamSpec(ridge=RIDGE, drift="none", capacity="pinned", final_knot="free")


def base_window() -> TrainingWindow:
    """Return GAM-K's training window: the last 24 months before the cut."""
    return TrainingWindow(months=WINDOW_MONTHS)


E155_DP_PREDS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00155-dp-bg-stage2-assembly-night-offset-and-error-correction"
    / "work"
    / "dp_preds"
)
"""Experiment 155's stored arm D, which this experiment reuses unchanged for the ENS era."""
QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
WINTER_MONTHS: Final[frozenset[int]] = frozenset({11, 12, 1, 2, 3})
N_NIGHT_SLOTS: Final[int] = 24
NIGHT_START_MINUTE: Final[int] = 21 * 60 + 30
OFFSET_RIDGE: Final[float] = 100.0
R_KAPPA: Final[float] = 0.75
R_PROFILE_DAYS: Final[int] = 14
R_MIN_DAYS: Final[int] = 3
R_FIRST_SOURCE_BACK: Final[int] = 2
MIN_DAY_READINGS: Final[int] = 40
D1_FIRST_LEAD: Final[float] = 15.5
D1_LAST_LEAD: Final[float] = 39.0
SHORT_LEAD_LIMIT_HOURS: Final[float] = 48.0
EARLY_LAST_ORIGIN: Final[datetime] = datetime(2024, 7, 1, 9, tzinfo=UTC)
"""Past-weather (`era` = `past_weather`) rows end before this origin; stored arm D starts after."""
EARLY_TUNING_MONTHS_PER_YEAR: Final[int] = 6
