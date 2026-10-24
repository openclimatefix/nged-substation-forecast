# ruff: noqa: E501
"""Paths and constants shared by the scripts of experiment 330; the rest comes from experiment 311.

Run every script with experiments 278, 311, and 314 on `PYTHONPATH`:

    export PYTHONPATH=ml-experiments/00278-stack-retraining-cadence:ml-experiments/00311-*:ml-experiments/00314-*:ml-experiments/00330-*
    uv run --no-sync python <script>
"""

from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 330
SLUG: Final[str] = "primary-demand-standard-architecture-arm"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
FIT_DIR: Final[Path] = WORK_DIR / "fits"
REPORT_DIR: Final[Path] = WORK_DIR / "report"
E311_DATA_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00311-primary-demand-mixture-density-band-head" / "work"
)
"""Experiment 311's prepared arrays (read only)."""
HEADS: Final[tuple[str, ...]] = ("std", "stda")
"""`std`: the adapter plus a mixture output. `stda`: the adapter plus a median-only output."""
SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4, 5)
"""Six seeds averaged, because network seed noise is 0.01 to 0.02 points (experiment 325)."""
ADAPTER_COLUMNS: Final[tuple[int, ...]] = (3, 5, 6, 7, 28)
"""Columns of experiment 311's input matrix that the adapter reads: the stored 5% to 95% width, the
cold and warm temperature hinges, the solar term, and the log of the band scale."""
