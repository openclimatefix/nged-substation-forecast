# ruff: noqa: E501
"""Paths shared by the scripts of experiment 314; every other constant comes from experiment 311.

Run every script with experiments 278 and 311 on `PYTHONPATH`:

    PYTHONPATH=ml-experiments/00278-stack-retraining-cadence:ml-experiments/00311-*:ml-experiments/00314-* \
        uv run --no-sync python <script>
"""

from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 314
SLUG: Final[str] = "primary-demand-gated-or-pinball-band-head"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
FIT_DIR: Final[Path] = WORK_DIR / "fits"
REPORT_DIR: Final[Path] = WORK_DIR / "report"
E311_DATA_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00311-primary-demand-mixture-density-band-head" / "work"
)
"""Experiment 311's prepared arrays, mixture fits, and scored arms (read only)."""
