"""Constants and loaders shared by the scripts of experiment 155 (best-guess stage 2 assembly).

Arm A is experiment 136's A2, read from that experiment's private `dp_preds` folder. Every other
arm is arm A's forecast plus a correction, then a spread refitted on the tuning months. This
experiment writes only under its own `WORK_DIR`.

**Three periods, each with one role** (as in experiment 136). The tuning period is the odd months of
2024-07 to 2025-06 (the spread's fit), the replicate period is the even months of the same year (the
sign of the contrasts), and the scored year is the 292 runs of `fold_a_daily_gap_v1` from 2025-07-02
to 2026-06-29. A run belongs to the month of its `power_fcst_init_time`.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.dp_features import FeatureCache, default_input_paths, load_or_build_feature_cache
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 197
SLUG: Final[str] = "dp-kalman-level-adaptation-retest"
ROW_SET_ID: Final[str] = "fold_a_daily_gap_v1"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
OUT_DIR: Final[Path] = WORK_DIR / "dp_preds"
E136_WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00136-dp-bg-stage1-assembly-on-the-ensemble-mean" / "work"
)
E111_WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00111-differentiable-physics-6-ens-forecast-engine-against-bg2"
    / "work"
)
E111_STORE_DIR: Final[Path] = E111_WORK_DIR / "ens_store"
CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
"""Part 4's feature cache, shared because it is keyed by a hash of its inputs."""

DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
SCORED_SITES: Final[tuple[str, ...]] = (*DECISION_SITES, "D9")
PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")
QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])

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

WINTER_MONTHS: Final[frozenset[int]] = frozenset({11, 12, 1, 2, 3})
N_NIGHT_SLOTS: Final[int] = 24
NIGHT_START_MINUTE: Final[int] = 21 * 60 + 30
"""The first night slot is the half-hour that ends at 21:30 UTC; the last ends at 09:00 UTC."""
OFFSET_RIDGE: Final[float] = 100.0
"""Experiment 137's ridge penalty, chosen on its odd tuning months."""
R_KAPPA: Final[float] = 0.75
R_PROFILE_DAYS: Final[int] = 14
R_MIN_DAYS: Final[int] = 3
R_FIRST_SOURCE_BACK: Final[int] = 2
"""The newest source day is `T - 2`: the target day `T - 1` is the origin's own day."""
MIN_DAY_READINGS: Final[int] = 40
D1_FIRST_LEAD: Final[float] = 15.5
D1_LAST_LEAD: Final[float] = 39.0
SHORT_LEAD_LIMIT_HOURS: Final[float] = 48.0
HORIZON_HOURS: Final[float] = 168.0

BASE_COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id", "run", "power_fcst_init_time", "valid_time", "lead_hours", "seed",
    *QUANTILES, "actual", "mean", "hour_band", "quarter", "switching_mask",
)  # fmt: skip


def load_cache() -> FeatureCache:
    """Return the feature cache, building and saving it on the first call."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)


def run_dates() -> list[datetime]:
    """Return every daily 00:00 UTC run from 2024-04-01 to 2026-06-30."""
    return [FIRST_RUN + timedelta(days=i) for i in range((LAST_RUN - FIRST_RUN).days + 1)]


def period_of_run(*, run: datetime) -> str:
    """Return `tuning`, `replicate`, `scored`, `stride`, or `warm_up` for a run."""
    origin = run + PUBLICATION_DELAY
    key = (origin.year, origin.month)
    if key in TUNING_MONTHS:
        return "tuning"
    if key in REPLICATE_MONTHS:
        return "replicate"
    if run >= SCORED_FIRST_RUN:
        stride = (run - SCORED_FIRST_RUN).days % EVERY5D_STRIDE_DAYS == 0
        return "stride" if stride else "scored"
    return "warm_up"


def period_origins(*, period: str) -> list[datetime]:
    """Return the `power_fcst_init_time` of every run in a period."""
    return [r + PUBLICATION_DELAY for r in run_dates() if period_of_run(run=r) == period]


def read_base(*, site: str) -> pl.DataFrame:
    """Return arm A (experiment 136's A2) at a site: all three periods, ascending by origin."""
    pieces = [
        pl.read_parquet(E136_WORK_DIR / "dp_preds" / f"a2_{period}_{site}.parquet")
        .select(list(BASE_COLUMNS))
        .with_columns(period=pl.lit(period))
        for period in PERIODS
    ]
    return pl.concat(pieces).sort(["power_fcst_init_time", "valid_time"])


def out_path(*, arm: str, period: str, site: str) -> Path:
    """Return the file that holds an arm's forecasts for one period at one site."""
    return OUT_DIR / f"{arm}_{period}_{site}.parquet"


def write_json_atomic(*, path: Path, content: dict) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, default=float))
    temporary.rename(path)
