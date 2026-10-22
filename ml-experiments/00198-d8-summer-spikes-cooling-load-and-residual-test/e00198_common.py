# ruff: noqa: E501
"""Constants and loaders shared by the scripts of experiment 198 (D8 summer spikes, cooling term).

Arm D is experiment 155's arm D, read from that experiment's private `dp_preds` folder. The
ensemble temperatures are experiment 111's stored ensemble arrays. This experiment writes only
under its own `WORK_DIR`.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.dp_features import FeatureCache, default_input_paths, load_or_build_feature_cache
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 198
SLUG: Final[str] = "d8-summer-spikes-cooling-load-and-residual-test"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
FIGURE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / (f"{EXPERIMENT_ID:05d}-{SLUG}")
)
E155_PREDS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00155-dp-bg-stage2-assembly-night-offset-and-error-correction"
    / "work"
    / "dp_preds"
)
E111_WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00111-differentiable-physics-6-ens-forecast-engine-against-bg2"
    / "work"
)
CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
PANEL_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "briefs" / "sw_false_alarms" / "work" / "arrays.pkl"
)
PANEL_START: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
D8_SERIES: Final[int] = 26
RUN_ZERO_UTC: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)

DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
STORED_SITES: Final[tuple[str, ...]] = (*DECISION_SITES, "D9")
QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
SUMMER_MONTHS: Final[frozenset[int]] = frozenset({6, 7, 8, 9})
FIT_MONTHS: Final[tuple[tuple[int, int], ...]] = ((2024, 7), (2024, 8), (2024, 9), (2025, 6))
"""The four non-scored summer months the hinge is fitted on (valid-time month of the run's origin)."""
TAUS: Final[tuple[float, ...]] = (16.0, 18.0, 20.0, 22.0, 24.0)
MIN_DAYS_ABOVE_TAU: Final[int] = 15
MIN_HALF_HOURS_ABOVE_TAU: Final[int] = 100
HOT_DAY_C: Final[float] = 25.0
N_BANDS: Final[int] = 6
RIDGE: Final[float] = 1.0
HORIZON_HALF_HOURS: Final[int] = 336


def load_cache() -> FeatureCache:
    """Return the shared feature cache (built by an earlier experiment)."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)


def read_arm_d(*, site: str, period: str) -> pl.DataFrame:
    """Return arm D's forecasts at a site for one period (`tuning`, `replicate`, or `scored`)."""
    return pl.read_parquet(E155_PREDS / f"d_{period}_{site}.parquet")


def read_ens_store(*, site: str) -> dict[str, np.ndarray]:
    """Return experiment 111's stored ensemble arrays at a site."""
    with np.load(E111_WORK / "ens_store" / f"{site}.npz") as stored:
        return {name: stored[name] for name in stored.files}


def write_json_atomic(*, path: Path, content: dict) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, default=float))
    temporary.rename(path)


def half_hours_after(*, start: datetime, count: int) -> list[datetime]:
    """Return `count` half-hour timestamps from `start`."""
    return [start + timedelta(minutes=30 * i) for i in range(count)]
