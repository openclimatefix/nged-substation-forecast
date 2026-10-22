"""Constants and loaders shared by the scripts of experiment 213 (online band adaptation).

The reference is arm A of experiment 169, read from the forecasts that experiment stored. Every
arm here keeps arm A's mean and rescales its quantile offsets. This experiment writes only under
its own `WORK_DIR`.
"""

import json
import os
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.dp_features import FeatureCache, default_input_paths, load_or_build_feature_cache
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 213
SLUG: Final[str] = "calibration-online-band-adaptation-from-coverage-misses"
ROW_SET_ID: Final[str] = "fold_a_daily_gap_v1"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
OUT_DIR: Final[Path] = WORK_DIR / "dp_preds"
REPORT_DIR: Final[Path] = WORK_DIR / "report"
E169_PREDS_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00169-calibration-combined-fixes-and-ensemble-spread"
    / "work"
    / "dp_preds"
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
ADAPTED_POSITIONS: Final[tuple[int, ...]] = (0, 1, 2, 3, 7, 8, 9, 10)
"""Positions in `QUANTILES` of the levels whose scale is adapted; q40, q50, and q60 are not."""
N_BANDS: Final[int] = 4
HORIZON_HOURS: Final[float] = 168.0

GAMMAS: Final[tuple[float, ...]] = (0.01, 0.03, 0.1)
SCALE_FLOOR: Final[float] = 0.7
SCALE_CAP: Final[float] = 2.0
DELAY_HOURS: Final[float] = 3.5
"""A reading enters the state used by a forecast only when it is at least this old at issue time."""
BLOCK_GAP_HOURS: Final[float] = 36.0
"""A gap between readings longer than this restarts the state (alternate-month tuning blocks)."""
CW_MULTIPLIERS: Final[dict[int, float]] = {8: 1.06, 9: 1.06, 10: 1.11}
"""Experiment 169's audit's intercept-only widening of q90, q95, and q99, by position."""

BASE_COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id", "run", "power_fcst_init_time", "valid_time", "lead_hours", "seed",
    *QUANTILES, "actual", "mean", "hour_band", "quarter", "switching_mask",
)  # fmt: skip
OUTPUT_COLUMNS: Final[tuple[str, ...]] = (*BASE_COLUMNS, "arm")


def gamma_arm(*, gamma: float) -> str:
    """Return the arm name of an adaptation run, such as `aci_g0.03`."""
    return f"aci_g{gamma:g}"


ARMS: Final[tuple[str, ...]] = ("a", "cw", *(gamma_arm(gamma=g) for g in GAMMAS))


def load_cache() -> FeatureCache:
    """Return the feature cache that experiments 106 and 155 built."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)


def stored_path(*, period: str, site: str) -> Path:
    """Return the file that holds experiment 169's arm A forecasts of one period and site."""
    return E169_PREDS_DIR / f"a_{period}_{site}.parquet"


def out_path(*, arm: str, period: str, site: str) -> Path:
    """Return the file that holds an arm's forecasts for one period at one site."""
    return OUT_DIR / f"{arm}_{period}_{site}.parquet"


def write_parquet_atomic(*, frame: pl.DataFrame, target: Path) -> None:
    """Write a parquet file through a temporary name so an interrupted write leaves no file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    frame.write_parquet(temporary)
    temporary.rename(target)


def write_json_atomic(*, path: Path, content: dict) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, default=float))
    temporary.rename(path)
