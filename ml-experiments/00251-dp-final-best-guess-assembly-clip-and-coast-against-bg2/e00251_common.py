# ruff: noqa: E501
"""Constants and helpers shared by the scripts of experiment 251 (final best-guess assembly)."""

import datetime as dt
import json
from pathlib import Path
from typing import Final

import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 251
SLUG: Final[str] = "dp-final-best-guess-assembly-clip-and-coast-against-bg2"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
ARM_DIR: Final[Path] = WORK_DIR / "arms"
REPORT_DIR: Final[Path] = WORK_DIR / "report"

E87_PREDICTIONS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00087-bg2-best-guess-assembled-with-drop-one-ablations" / "predictions"
)  # fmt: skip
E228_SCORED: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00228-dp-bg-rolling-origin-stack-against-bg2" / "work" / "arms" / "scored"
)  # fmt: skip
E178_ROWS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00178-dp-sunny-daylight-gap-diagnosis" / "work" / "report" / "joined_rows.parquet"
)  # fmt: skip
E180_ARMS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00180-dp-sunny-daylight-fix-d5-solar-shape" / "work" / "arms"
)  # fmt: skip
E205_REPORT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00205-dp-gamk-gap-robustness-coast-or-frozen-gam" / "work" / "report" / "report.json"
)  # fmt: skip
E169_PREDS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00169-calibration-combined-fixes-and-ensemble-spread" / "work" / "dp_preds"
)  # fmt: skip
E220_ARMS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00220-dp-bg-stage4-assembly-with-drop-one" / "work" / "arms"
)  # fmt: skip
STORED_87: Final[dict[str, str]] = {
    "bg2": "p_s6_bg2",
    "climatology": "ref_climatology",
    "xgboost": "ref_xgboost_baseline",
    "manual_heuristic": "ref_manual_heuristic",
}

DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
SCORED_SITES: Final[tuple[str, ...]] = (*DECISION_SITES, "D9")
QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
SCORED_START: Final[dt.datetime] = dt.datetime(2025, 7, 2, tzinfo=dt.UTC)
GATE_ARM_A: Final[float] = 2.7519
GATE_ARM_1: Final[float] = 2.617
GATE_BG2: Final[float] = 2.780
GATE_TOLERANCE: Final[float] = 0.005


def write_json_atomic(*, path: Path, content: dict) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, default=float))
    temporary.rename(path)


def write_npy_atomic(*, path: Path, array: np.ndarray) -> None:
    """Write a NumPy array through a temporary name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.npy")
    np.save(temporary, array)
    temporary.rename(path)
