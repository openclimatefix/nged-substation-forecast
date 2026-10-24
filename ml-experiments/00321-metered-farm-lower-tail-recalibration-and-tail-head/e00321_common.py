"""Shared constants for experiment 321 (lower-tail recalibration and a tail-targeted head).

Farms appear only as labels A to F. Every table written here lives under `data/ml-experiments/`.
"""

from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00321-metered-farm-lower-tail-recalibration-and-tail-head"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
PRED_DIR: Final[Path] = DATA_DIR / "predictions"
E310_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00310-metered-farm-long-lead-scoring"
E318_PRED_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00318-metered-farm-mdn-head-more-features" / "predictions"
)
OWN_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
ZERO_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
CONTROL_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
ZERO_OUTPUT: Final[float] = 0.002
"""An half-hour with output at or below this (in the farm's scale units) counts as an exact zero."""
ZI_INPUTS: Final[tuple[str, ...]] = (
    "mu",
    "plant_spread",
    "avail_ratio",
    "lag_power",
    "lag_missing",
    "csi",
    "lead_hours",
    "hour_sin",
    "hour_cos",
)
"""Experiment 310's head inputs plus the hour as a sine and cosine pair."""
TAIL_INPUTS: Final[tuple[str, ...]] = (
    "mu",
    "plant_spread",
    "avail_ratio",
    "lag_power",
    "lag_missing",
    "csi",
    "lead_hours",
    "lead_day",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "ghi_std",
)
"""Experiment 318's `full` inputs, copied so a change there is noticed."""
TAIL_TRAIN_LEVELS: Final[tuple[float, ...]] = (0.01, 0.02, 0.05, 0.95, 0.98, 0.99)
TAIL_STORED_LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.95, 0.99)
NON_CROSSING_WEIGHT: Final[float] = 10.0
PURE_LEAD_INPUT_EXTRA: Final[str] = "lead_day"
SHUFFLED_COLUMNS: Final[tuple[str, ...]] = ("lead_hours", "lead_day", "csi")
