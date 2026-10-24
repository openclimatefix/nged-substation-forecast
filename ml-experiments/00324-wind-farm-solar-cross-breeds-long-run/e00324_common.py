"""Shared paths for experiment 324 (solar-farm findings crossed with the three metered wind farms).

Wind farms appear only as W1 to W3. Every table written here lives under `data/ml-experiments/`.
"""

from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00324-wind-farm-solar-cross-breeds-long-run"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
FINAL_DIR: Final[Path] = DATA_DIR / "final"
E303_WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00303-wind-adapter-in-production-setting" / "work"
)
E313_WIND: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00313-metered-farm-pure-xgboost-and-mdn-vs-plant" / "wind"
)
FARMS: Final[tuple[str, ...]] = ("W1", "W2", "W3")
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
