"""Constants and loaders shared by the scripts of experiment 164 (the cold-night tail).

The reference is arm I of experiment 160: experiment 155's arm D with arm A's spread table read by
interpolation between the lead-band centres. Every fix keeps the reference's mean and rescales the
quantile offsets on cold days. This experiment writes only under its own `WORK_DIR`.
"""

import json
from pathlib import Path
from typing import Final

import numpy as np
from studies.dp_features import FeatureCache, default_input_paths, load_or_build_feature_cache
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 164
SLUG: Final[str] = "calibration-cold-night-tail-under-coverage"
ROW_SET_ID: Final[str] = "fold_a_daily_gap_v1"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
OUT_DIR: Final[Path] = WORK_DIR / "dp_preds"
E160_PREDS_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00160-calibration-band-width-fix-on-stage-2-winner"
    / "work"
    / "dp_preds"
)
REFERENCE_ARM_FILE: Final[str] = "i"
"""Experiment 160's arm I (the interpolated spread table)."""
E111_STORE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00111-differentiable-physics-6-ens-forecast-engine-against-bg2"
    / "work"
    / "ens_store"
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
MEDIAN_POSITION: Final[int] = 5

HORIZON_HOURS: Final[float] = 168.0
N_LEAD_DAYS: Final[int] = 7
"""Lead days 0 to 6, where day 6 holds leads over 144 and up to 168 hours."""
WINDOW_HALF_HOURS: Final[int] = 24
"""The temperature window is 24 half-hours either side of the valid time (a 24-hour mean)."""
COLD_GATE_C: Final[float] = 2.0
"""A row is cold-forecast when the ensemble-mean 24-hour temperature is below this."""
TEMPERATURE_CLASS_EDGES_C: Final[tuple[float, ...]] = (2.0, 6.0, 10.0)
"""Edges of the four temperature classes of fix T: below 2, 2 to 6, 6 to 10, and 10 or more."""
MIN_CLASS_ROWS: Final[int] = 500
UPPER_FIX_LEVELS: Final[tuple[float, ...]] = (0.75, 0.9, 0.95, 0.99)

ARMS: Final[tuple[str, ...]] = ("ref", "t", "s", "o")
ARM_LABELS: Final[dict[str, str]] = {
    "ref": "Reference (experiment 160, arm I)",
    "t": "T: temperature-class factor",
    "s": "S: ensemble-spread factor",
    "o": "O: one-sided factor",
}
BASE_COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id", "run", "power_fcst_init_time", "valid_time", "lead_hours", "seed",
    *QUANTILES, "actual", "mean", "hour_band", "quarter", "switching_mask",
)  # fmt: skip
ENS_COLUMNS: Final[tuple[str, ...]] = ("ens_mean_c", "ens_std_c")
OUTPUT_COLUMNS: Final[tuple[str, ...]] = (*BASE_COLUMNS, *ENS_COLUMNS, "arm")


def load_cache() -> FeatureCache:
    """Return the feature cache that experiments 106 and 155 built."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)


def reference_path(*, period: str, site: str) -> Path:
    """Return the file that holds the reference forecasts of one period at one site."""
    return E160_PREDS_DIR / f"{REFERENCE_ARM_FILE}_{period}_{site}.parquet"


def out_path(*, arm: str, period: str, site: str) -> Path:
    """Return the file that holds an arm's forecasts for one period at one site."""
    return OUT_DIR / f"{arm}_{period}_{site}.parquet"


def write_json_atomic(*, path: Path, content: dict) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, default=float))
    temporary.rename(path)
