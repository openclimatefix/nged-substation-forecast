"""Constants and loaders shared by the scripts of experiment 160 (the band-width fix).

The reference is arm D of experiment 155 with arm A's spread table (`d_spread_a` in 155's private
`dp_preds` folder). Every variant keeps the reference's mean and changes only how the stored spread
table is read along the lead. This experiment writes only under its own `WORK_DIR`.
"""

import json
from pathlib import Path
from typing import Final

import numpy as np
from studies.dp_features import FeatureCache, default_input_paths, load_or_build_feature_cache
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 160
SLUG: Final[str] = "calibration-band-width-fix-on-stage-2-winner"
ROW_SET_ID: Final[str] = "fold_a_daily_gap_v1"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
OUT_DIR: Final[Path] = WORK_DIR / "dp_preds"
E155_PREDS_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00155-dp-bg-stage2-assembly-night-offset-and-error-correction"
    / "work"
    / "dp_preds"
)
REFERENCE_ARM_FILE: Final[str] = "d_spread_a"
"""Experiment 155's arm D with arm A's spread table."""
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

LEAD_BAND_UPPER_HOURS: Final[tuple[float, ...]] = (24.0, 48.0, 144.0, 168.0)
"""Upper edges of the stored spread table's lead bands, as `studies.dp_ens_weather` defines them."""
LEAD_BAND_CENTRES_HOURS: Final[tuple[float, ...]] = (12.0, 36.0, 96.0, 156.0)
"""The centre of each lead band: 0 to 24, 24 to 48, 48 to 144, and 144 to 168 hours."""
N_LEAD_BANDS: Final[int] = 4
N_HOUR_BANDS: Final[int] = 4
N_QUARTERS: Final[int] = 4
HORIZON_HOURS: Final[float] = 168.0
N_LEAD_DAYS: Final[int] = 7
"""Lead days 0 to 6, where day 6 holds leads over 144 and up to 168 hours."""

ARMS: Final[tuple[str, ...]] = ("ref", "i", "m", "im")
ARM_LABELS: Final[dict[str, str]] = {
    "ref": "Reference (stored)",
    "i": "I: interpolated",
    "m": "M: non-decreasing",
    "im": "IM: both",
}
BASE_COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id", "run", "power_fcst_init_time", "valid_time", "lead_hours", "seed",
    *QUANTILES, "actual", "mean", "hour_band", "quarter", "switching_mask",
)  # fmt: skip


def load_cache() -> FeatureCache:
    """Return the feature cache that experiments 106 and 155 built."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)


def reference_path(*, period: str, site: str) -> Path:
    """Return the file that holds the reference forecasts of one period at one site."""
    return E155_PREDS_DIR / f"{REFERENCE_ARM_FILE}_{period}_{site}.parquet"


def out_path(*, arm: str, period: str, site: str) -> Path:
    """Return the file that holds an arm's forecasts for one period at one site."""
    return OUT_DIR / f"{arm}_{period}_{site}.parquet"


def write_json_atomic(*, path: Path, content: dict) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, default=float))
    temporary.rename(path)
