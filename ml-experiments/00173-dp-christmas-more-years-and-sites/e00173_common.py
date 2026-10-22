"""Constants and helpers shared by the scripts of experiment 173 (Christmas, more years).

**One forecast season per Christmas.** A convex-model fit on a cut (1 November; 16 December for
Christmas 2019-20, the first date with 90 days of data) forecasts daily 09:00 UTC runs of 168 hours
up to 31 January, driven by observed weather. The Kalman filter's noise setting is experiment 133's
shared one, and recipe R is experiment 155's.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

from studies.dp_features import FeatureCache, default_input_paths, load_or_build_feature_cache
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 173
SLUG: Final[str] = "dp-christmas-more-years-and-sites"
ROW_SET_ID: Final[str] = "dp_christmas_daily_v1"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
ARMS: Final[tuple[str, ...]] = ("m12", "all")
"""`m12`: the pre-registered arm (12 months of history). `all`: post hoc, all years of history."""
E155_PREDS_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00155-dp-bg-stage2-assembly-night-offset-and-error-correction"
    / "work"
    / "dp_preds"
)


def forecast_dir(*, arm: str) -> Path:
    """Return the folder that holds an arm's forecast files."""
    return WORK_DIR / "forecasts" / arm


CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
"""Part 4's feature cache, shared because it is keyed by a hash of its inputs."""
TUNED_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00104-differentiable-physics-2-simple-structural-model"
    / "work"
    / "tuned.json"
)
"""Part 2's tuned penalties (ridge 0.1, drift step), used unchanged."""
POOL_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00133-differentiable-physics-vk-learned-noise-and-observation-variance-in-gam-k"
    / "work"
    / "select"
    / "pool.json"
)
"""Experiment 133's single shared filter cell (POOL)."""

DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
ALL_SITES: Final[tuple[str, ...]] = (
    "D2", "D3", "D4", "D5", "D8", "D9", "D10", "D11", "D12", "D14", "D15", "D16", "D18", "D19",
    "D21", "D22",
)  # fmt: skip
"""The 16 Primary substations in the feature cache."""

SEASONS: Final[tuple[int, ...]] = tuple(range(2019, 2026))
"""A season is named by the year of its 24 December: 2019 is the Christmas of 2019-20."""
FIRST_FULL_CUT_DATE: Final[datetime] = datetime(2019, 12, 16, tzinfo=UTC)
"""The first date with 90 days of data, the cut of the 2019 season."""
FIRST_SEASON_WITH_SPREAD: Final[int] = 2020
PUBLICATION_DELAY: Final[timedelta] = timedelta(hours=9)
HORIZON_HALF_HOURS: Final[int] = 336
SEASON_YEARS: Final[range] = range(2019, 2027)
TRAINING_MONTHS: Final[int] = 12
"""The pre-registered arm's window; the `all` arm uses every year before the cut."""
REFERENCE_FIRST_DAY: Final[tuple[int, int]] = (12, 8)
"""Ordinary weekdays are counted from 8 December (month, day) to the end of the season's runs."""
SPREAD_FIRST_VALID: Final[tuple[int, int]] = (11, 16)
SPREAD_LAST_VALID: Final[tuple[int, int]] = (12, 22)

R_KAPPA: Final[float] = 0.75
R_PROFILE_DAYS: Final[int] = 14
R_MIN_DAYS: Final[int] = 3
R_FIRST_SOURCE_BACK: Final[int] = 2
MIN_DAY_READINGS: Final[int] = 40
D1_FIRST_LEAD: Final[float] = 15.5
D1_LAST_LEAD: Final[float] = 39.0
SHORT_LEAD_LIMIT_HOURS: Final[float] = 48.0
QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99)


def cut_of_season(*, season: int) -> datetime:
    """Return the cut: 1 November of the season's year, or 16 December for the first season."""
    if season == SEASONS[0]:
        return FIRST_FULL_CUT_DATE
    return datetime(season, 11, 1, tzinfo=UTC)


def last_run_of_season(*, season: int) -> datetime:
    """Return the last daily run: 31 January after the season's Christmas."""
    return datetime(season + 1, 1, 31, tzinfo=UTC)


def load_cache() -> FeatureCache:
    """Return the feature cache, building and saving it on the first call."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)


def write_json_atomic(*, path: Path, content: dict) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, default=float))
    temporary.rename(path)


ADEQUATE_TRAINING_DAYS: Final[tuple[int, ...]] = (90, 180, 270)
"""Post hoc thresholds (days of usable training rows) for a fit that has seen enough of a year.

A fit's `n_train_rows` counts half-hours after the lockdown window and the other row decisions of
experiment 103 are applied, so a substation can have far fewer rows than the calendar suggests.
"""
