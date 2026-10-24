# ruff: noqa: E501
"""Paths and constants shared by the scripts of experiment 337.

The network, the input matrix layout, and the scoring helpers are those of experiments 311 and 330.
Run every script with the folders of experiments 201, 278, 311, 314, 330, and 337 on `PYTHONPATH`:

    export PYTHONPATH=$(ls -d ml-experiments/00201-* ml-experiments/00278-* ml-experiments/00311-* ml-experiments/00314-* ml-experiments/00330-* ml-experiments/00337-* | paste -sd:)
    uv run --no-sync python <script>
"""

from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 337
SLUG: Final[str] = "primary-demand-standard-architecture-168-to-240-hours"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
PREP_DIR: Final[Path] = WORK_DIR / "prep"
FIT_DIR: Final[Path] = WORK_DIR / "fits"
REPORT_DIR: Final[Path] = WORK_DIR / "report"
CHAIN_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00317-primary-demand-default-1-at-7-to-10-days" / "work" / "chain"
)
"""Experiment 317's convex chain rows out to 240 hours (arm D's mean and the default 1 quantiles)."""
E335_SCORED: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00335-primary-demand-default-3-tail-calibration-at-0-5-to-240-hours"
    / "work"
    / "default3_scored.parquet"
)
"""Primary demand research default 3, the manual heuristic, and climatology over 0.5 to 240 hours."""
STORED_HORIZON_HOURS: Final[float] = 168.0
MAX_LEAD_HOURS: Final[int] = 240
"""A training row's valid time is at most this many hours after its origin."""
LEAD_BANDS: Final[tuple[tuple[float, float], ...]] = ((168.0, 192.0), (192.0, 216.0), (216.0, 240.0))  # fmt: skip
