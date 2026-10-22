"""Constants and loaders shared by the scripts of experiment 176 (a band width smooth in lead).

The reference is arm A of experiment 169 (experiment 160's interpolated spread table plus
experiment 164's fix O). Every arm keeps that arm's mean forecast and fix-O multipliers and changes
only the base offsets from the mean. This experiment writes only under its own `WORK_DIR`.
"""

import json
from pathlib import Path
from typing import Final

import numpy as np
from studies.dp_features import FeatureCache, default_input_paths, load_or_build_feature_cache
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 176
SLUG: Final[str] = "calibration-band-shape-smooth-in-lead"
ROW_SET_ID: Final[str] = "fold_a_daily_gap_v1"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
OUT_DIR: Final[Path] = WORK_DIR / "dp_preds"
E169_PREDS_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00169-calibration-combined-fixes-and-ensemble-spread"
    / "work"
    / "dp_preds"
)
E160_WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00160-calibration-band-width-fix-on-stage-2-winner" / "work"
)
CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)

DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
SCORED_SITES: Final[tuple[str, ...]] = (*DECISION_SITES, "D9")
PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")
QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
FIT_LEVEL_POSITIONS: Final[tuple[int, ...]] = (0, 1, 2, 3, 7, 8, 9, 10)
"""Positions in `QUANTILES` of the eight levels refitted; q40, q50, and q60 keep their values."""
UPPER_FIX_POSITIONS: Final[tuple[int, ...]] = (7, 8, 9, 10)
"""The levels experiment 164's fix O multiplies on forecast-cold days."""
COLD_GATE_C: Final[float] = 2.0
HORIZON_HOURS: Final[float] = 168.0
N_LEAD_DAYS: Final[int] = 7
"""Lead days 0 to 6; day = ceil(lead_hours / 24) - 1, so a lead of exactly 168 hours is in day 6."""
N_HOUR_BANDS: Final[int] = 4
N_QUARTERS: Final[int] = 4
N_LEAD_BANDS: Final[int] = 4
LEAD_BAND_CENTRES_HOURS: Final[tuple[float, ...]] = (12.0, 36.0, 96.0, 156.0)
LEAD_BAND_UPPER_HOURS: Final[tuple[float, ...]] = (24.0, 48.0, 144.0, 168.0)
KNOTS_HOURS: Final[np.ndarray] = np.arange(0.0, 169.0, 24.0)
"""Knots of the piecewise-linear growth of fix P and fix U: every 24 hours from 0 to 168."""

ARMS: Final[tuple[str, ...]] = ("a", "pb", "p", "u", "s", "ps")
ARM_LABELS: Final[dict[str, str]] = {
    "a": "A: reference (experiment 169)",
    "pb": "PB: monotone growth per hour band (first attempt)",
    "p": "P: monotone growth per hour band and quarter, pooled across substations",
    "u": "U: monotone growth per substation, hour band, and quarter",
    "s": "S: cells shrunk towards the pooled table",
    "ps": "PS: P with its cell scales shrunk",
}
BASE_COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id", "run", "power_fcst_init_time", "valid_time", "lead_hours", "seed",
    *QUANTILES, "actual", "mean", "hour_band", "quarter", "switching_mask", "ens_mean_c",
)  # fmt: skip


def load_cache() -> FeatureCache:
    """Return the feature cache that experiments 106 and 155 built."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)


def stored_path(*, arm: str, period: str, site: str) -> Path:
    """Return the file that holds experiment 169's forecasts of one arm, period, and site."""
    return E169_PREDS_DIR / f"{arm}_{period}_{site}.parquet"


def out_path(*, arm: str, period: str, site: str) -> Path:
    """Return the file that holds an arm's forecasts for one period at one site."""
    return OUT_DIR / f"{arm}_{period}_{site}.parquet"


def table_path(*, site: str) -> Path:
    """Return experiment 160's stored spread table: hour band, quarter, lead band, level."""
    return E160_WORK_DIR / f"table_{site}.npy"


def lead_day(*, lead_hours: np.ndarray) -> np.ndarray:
    """Return the lead day, 0 to 6, of leads in hours."""
    return np.clip(np.ceil(lead_hours / 24.0) - 1, 0, N_LEAD_DAYS - 1).astype(np.int64)


def write_json_atomic(*, path: Path, content: dict) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, default=float))
    temporary.rename(path)
