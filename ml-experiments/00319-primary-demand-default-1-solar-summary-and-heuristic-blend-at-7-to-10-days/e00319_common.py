# ruff: noqa: E501
"""Constants shared by the scripts of experiment 319 (solar summary and heuristic blend at 7 to 10 days).

Every input is a stored file of an earlier experiment, read and never written. This experiment
writes only under its own `WORK_DIR`.
"""

import json
from pathlib import Path
from typing import Final

import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 319
SLUG: Final[str] = "primary-demand-default-1-solar-summary-and-heuristic-blend-at-7-to-10-days"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
MEMBERS_DIR: Final[Path] = WORK_DIR / "members"
ARMS_DIR: Final[Path] = WORK_DIR / "arms"
REPORT_DIR: Final[Path] = WORK_DIR / "report"

E317_WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00317-primary-demand-default-1-at-7-to-10-days" / "work"
)
CHAIN_DIR: Final[Path] = E317_WORK / "chain"
E184_ARMS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00184-dp-ens-daylight-calibration" / "work" / "arms"
)
E87_PREDICTIONS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00087-bg2-best-guess-assembled-with-drop-one-ablations"
    / "predictions"
)
E251_ARMS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00251-dp-final-best-guess-assembly-clip-and-coast-against-bg2"
    / "work"
    / "arms"
    / "scored"
)
E178_ROWS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00178-dp-sunny-daylight-gap-diagnosis"
    / "work"
    / "report"
    / "joined_rows.parquet"
)
E220_ARMS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00220-dp-bg-stage4-assembly-with-drop-one" / "work" / "arms"
)
# fmt: off
DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
SCORED_SITES: Final[tuple[str, ...]] = (*DECISION_SITES, "D9")
PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")
QUANTILES: Final[tuple[str, ...]] = ("q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99")
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
# fmt: on
HORIZON_HOURS: Final[float] = 240.0
HORIZON_HALF_HOURS: Final[int] = 480
STORED_HORIZON_HOURS: Final[float] = 168.0
LONG_BAND_EDGES_HOURS: Final[tuple[float, ...]] = (168.0, 192.0, 216.0, 240.0)
"""Edges of the three 24-hour lead bands past 168 hours."""
WINDOW_DAYS: Final[int] = 365
WINTER_MONTHS: Final[frozenset[int]] = frozenset({11, 12, 1, 2, 3})


def long_band(*, lead_hours: np.ndarray) -> np.ndarray:
    """Return the long lead band (0, 1, 2) of leads above 168 hours, and -1 at 168 hours or less."""
    index = np.searchsorted(np.array(LONG_BAND_EDGES_HOURS), lead_hours, side="left") - 1
    return np.where(lead_hours > STORED_HORIZON_HOURS, np.clip(index, 0, 2), -1)


def write_json_atomic(*, path: Path, content: dict) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, default=float))
    temporary.rename(path)
