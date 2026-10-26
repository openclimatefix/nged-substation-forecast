"""Constants shared by the scripts of experiment 352.

Experiment 352 scales the pinned solar capacity of the first stage of Primary demand research
default 3 and forecasts with and without the Kalman filter. It imports the engine of experiment
349's folder (run these scripts with that folder on `PYTHONPATH`) and writes only under its own
private data folder.
"""

from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 352
SLUG: Final[str] = "primary-demand-default-3-capacity-response-surface"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
PREDS_DIR: Final[Path] = WORK_DIR / "preds"
SCALES: Final[tuple[float, ...]] = (0.6, 0.8, 1.0, 1.2, 1.4)
"""Multipliers on the pinned weekly capacity path."""
FILTERS: Final[tuple[str, ...]] = ("kf", "nokf")
"""`kf` runs the Kalman filter's correction; `nokf` forecasts the fitted mean alone."""
PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")
STORED_ARM_A: Final[str] = "pin"
"""Experiment 349's arm A, which scale 1.0 with the filter must reproduce."""


def arm_name(*, scale: float, filter_name: str) -> str:
    """Return the arm label, for example `s1.2_kf`."""
    return f"s{scale:.1f}_{filter_name}"


def preds_path(*, scale: float, filter_name: str, period: str, site: str) -> Path:
    """Return the file that holds an arm's forecasts for one period at one site."""
    return PREDS_DIR / f"{arm_name(scale=scale, filter_name=filter_name)}_{period}_{site}.parquet"
