# ruff: noqa: E501
"""Constants and helpers shared by the scripts of experiment 228 (rolling-origin comparison)."""

import json
from pathlib import Path
from typing import Final

import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 228
SLUG: Final[str] = "dp-bg-rolling-origin-stack-against-bg2"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
ARM_DIR: Final[Path] = WORK_DIR / "arms"
REPORT_DIR: Final[Path] = WORK_DIR / "report"

E87_PREDICTIONS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00087-bg2-best-guess-assembled-with-drop-one-ablations" / "predictions"
)  # fmt: skip
E220_ARMS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00220-dp-bg-stage4-assembly-with-drop-one" / "work" / "arms"
)  # fmt: skip
STORED_87: Final[dict[str, str]] = {
    "bg2": "p_s6_bg2",
    "climatology": "ref_climatology",
    "manual_heuristic": "ref_manual_heuristic",
}

DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
SCORED_SITES: Final[tuple[str, ...]] = (*DECISION_SITES, "D9")
QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
GATE_ARM_A: Final[float] = 2.7519
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
