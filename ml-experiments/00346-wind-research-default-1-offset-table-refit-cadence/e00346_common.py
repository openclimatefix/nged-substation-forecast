"""Paths and constants shared by the scripts of experiment 00346.

The scripts import experiments 00342, 00340, 00336, 00326, 00303, 00285, 00259, and 00230, so run
them with
`PYTHONPATH=ml-experiments/00259-wind-plant-plus-quantile-offset-head:ml-experiments/00303-wind-adapter-in-production-setting:ml-experiments/00326-wind-research-default-1-change-arms:ml-experiments/00336-wind-research-default-1-below-cut-in-zeroing`.
Everything read from other experiments' folders is read only.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00336_common as c336  # isort: skip  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

c303 = c336.c303
EXPERIMENT_ID: Final[int] = 346
SLUG: Final[str] = "wind-research-default-1-offset-table-refit-cadence"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables; private and never committed."""
MEMBERS_DIR: Final[Path] = WORK_DIR / "members"
MIX_DIR: Final[Path] = WORK_DIR / "mix"
FINAL_DIR: Final[Path] = WORK_DIR / "final"
FARMS: Final[tuple[str, ...]] = c336.FARMS
SEEDS: Final[tuple[int, ...]] = c336.SEEDS
FIRST_NEW_RUN: Final[datetime] = datetime(2025, 6, 17, tzinfo=UTC)
LAST_NEW_RUN: Final[datetime] = datetime(2026, 10, 23, tzinfo=UTC)
"""Daily 00 UTC runs after the hindcast of experiment 259, which ends on 2025-06-16."""
SCORED_FIRST_MONTH: Final[str] = "2025-07"
OUT_OF_TIME_FIRST_RUN: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
TELEMETRY_DELAY_MINUTES: Final[int] = 210
MAX_LEAD_DAY: Final[int] = 15
