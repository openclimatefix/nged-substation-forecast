"""Shared paths and constants of experiment 00357 (longer retest of the monthly retrained head).

Run every script with `PYTHONPATH` set to the folders in `PYTHONPATH_FOLDERS`. The case table, the
default 1 and default 2 forecasts come from experiment 00356's data folder.
"""

from pathlib import Path
from typing import Final

import e00356_common as c356  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 357
SLUG: Final[str] = "wind-default-2-monthly-head-longer-retest"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
PARTS_DIR: Final[Path] = DATA_DIR / "parts"
"""One file per monthly head and seed: `u3_s<seed>_<YYYY-MM>.npy`, written atomically."""
SOURCE_DIR: Final[Path] = c356.DATA_DIR
SEEDS: Final[tuple[int, ...]] = c356.SEEDS
PYTHONPATH_FOLDERS: Final[tuple[str, ...]] = (
    "00357-wind-default-2-monthly-head-longer-retest",
    *c356.PYTHONPATH_FOLDERS,
)
