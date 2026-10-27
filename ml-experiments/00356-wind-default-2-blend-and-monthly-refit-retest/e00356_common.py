"""Shared paths and constants of experiment 00356 (blend and monthly refit of Wind farms default 2).

Run every script with `PYTHONPATH` set to the folders of experiments 00356, 00348, 00347, 00346,
00336, 00326, 00310, 00303, 00259, and 00230 (see `PYTHONPATH_FOLDERS`).
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 356
SLUG: Final[str] = "wind-default-2-blend-and-monthly-refit-retest"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
INPUTS_DIR: Final[Path] = DATA_DIR / "inputs"
"""Plant summaries of the daily runs after the hindcast, one pair of files per farm."""
PARTS_DIR: Final[Path] = DATA_DIR / "parts"
"""One file per arm and seed: `<name>.npy`, written atomically."""
E346_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00346-wind-research-default-1-offset-table-refit-cadence"
)
E346_MEMBERS: Final[Path] = E346_DIR / "work" / "members"
FARMS: Final[tuple[str, ...]] = ("W1", "W2", "W3")
SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
FIRST_NEW_RUN: Final[datetime] = datetime(2025, 6, 17, tzinfo=UTC)
LAST_NEW_RUN: Final[datetime] = datetime(2026, 10, 23, tzinfo=UTC)
WINDOW_START: Final[datetime] = datetime(2025, 4, 16, tzinfo=UTC)
"""The first day of the finished runs that default 2's head did not train on (its cutoff)."""
HEAD_WINDOW_START: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
FIRST_SCORED_MONTH: Final[tuple[int, int]] = (2025, 7)
LAST_SCORED_MONTH: Final[tuple[int, int]] = (2026, 10)
OUT_OF_TIME_FIRST_RUN: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
MASK_END: Final[datetime] = datetime(2026, 9, 13, 18, tzinfo=UTC)
DELAY: Final[timedelta] = timedelta(minutes=210)
MIN_WINDOW_ROWS: Final[int] = 500
LEAD_EDGES: Final[tuple[float, ...]] = (24.0, 72.0, 240.0)
MAX_LEAD_DAY: Final[int] = 15
PYTHONPATH_FOLDERS: Final[tuple[str, ...]] = (
    "00356-wind-default-2-blend-and-monthly-refit-retest",
    "00348-wind-plant-mdn-upper-tail",
    "00347-wind-physics-hybrid-plant-then-mdn",
    "00346-wind-research-default-1-offset-table-refit-cadence",
    "00336-wind-research-default-1-below-cut-in-zeroing",
    "00326-wind-research-default-1-change-arms",
    "00310-metered-farm-long-lead-scoring",
    "00303-wind-adapter-in-production-setting",
    "00259-wind-plant-plus-quantile-offset-head",
    "00230-wind-physics-core-part-1",
)
