"""Paths and constants shared by the scripts of experiment 00338.

The scripts import experiment 00326's and 00303's modules, so run them with
`PYTHONPATH=ml-experiments/00303-wind-adapter-in-production-setting:ml-experiments/00326-wind-research-default-1-change-arms`.
Everything read from other experiments' folders is read only.
"""

from pathlib import Path
from typing import Final

import e00326_common as c326  # isort: skip  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

c303 = c326.c303

EXPERIMENT_ID: Final[int] = 338
SLUG: Final[str] = "wind-adapter-column-without-neural-network"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables; private and never committed."""
MIX_DIR: Final[Path] = WORK_DIR / "mix"
FINAL_DIR: Final[Path] = WORK_DIR / "final"
TABLE_DIR: Final[Path] = WORK_DIR / "tables"
SEEDS: Final[tuple[int, ...]] = c326.SEEDS
FARMS: Final[tuple[str, ...]] = c326.FARMS
SHUFFLE_SEED: Final[int] = 338
DIRECTION_SECTORS: Final[int] = 8
RATIO_BINS: Final[int] = 6
TEMPERATURE_BINS: Final[int] = 4
KNOWN_ANSWER_TOLERANCE: Final[float] = 0.03

FIT_ARMS: Final[tuple[str, ...]] = (
    "x1",
    "x2",
    "x3",
    "x1_no_direction",
    "x1_no_ratio",
    "x1_pl",
    "x2_pl",
)
"""Arms fitted here. `x4` is experiment 326's arm (c), read from storage; `default` is
experiment 326's refit default, read from storage; `stored_default` is experiment 285's `xgb_B0`."""
STORED_ARMS: Final[dict[str, Path]] = {
    "default": c326.MIX_DIR,
    "x4": c326.STORED_303_MIX_DIR,
    "stored_default": c326.STORED_285_MIX_DIR,
}
STORED_PREFIX: Final[dict[str, str]] = {
    "default": "default",
    "x4": "P8",
    "stored_default": "xgb_B0",
}
IMPLIED_CASES_PATH: Final[Path] = c326.IMPLIED_CASES_PATH
"""Experiment 307's hindcast cases with the ensemble-mean 100 m wind (for the above-rated flag)."""
STORED_303_FINAL_DIR: Final[Path] = c326.STORED_303_FINAL_DIR
REPLICATE_START: Final = c326.REPLICATE_START
