"""Paths and constants shared by the scripts of experiment 00339.

Run the scripts with `PYTHONPATH` set to the folders of experiments 303, 310, 321, 324, 326, and
336 (see `e00339_run.py`). Everything read from other experiments' folders is read only.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 339
SLUG: Final[str] = "wind-research-default-1-adapter-plus-mixture-head"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
PARTS_DIR: Final[Path] = DATA_DIR / "parts"
"""One file per task: `<arm>_b<base seed>_h<head seed>.npy`, written atomically."""
FINAL_DIR: Final[Path] = DATA_DIR / "final"
E324_FINAL: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00324-wind-farm-solar-cross-breeds-long-run" / "final"
)
E336_WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00336-wind-research-default-1-below-cut-in-zeroing" / "work"
)
FARMS: Final[tuple[str, ...]] = ("W1", "W2", "W3")
BASE_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
"""Seeds of the stored base quantiles (experiment 303's P2). The head seed equals the base seed."""
EXTRA_HEAD_SEEDS: Final[tuple[int, ...]] = (1, 2, 3, 4)
"""Seed check: heads with these seeds on base seed 0, beside the main seed 0 head."""
MAIN_CUTOFF: Final[datetime] = datetime(2025, 4, 16, tzinfo=UTC)
REPLICATE_CUTOFF: Final[datetime] = datetime(2024, 12, 1, tzinfo=UTC)
REPLICATE_START: Final[datetime] = datetime(2025, 1, 1, tzinfo=UTC)
