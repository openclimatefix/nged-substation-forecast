"""Constants shared by the scripts of experiment 350 (capacity-free irradiance curve).

Every input is a stored file of an earlier experiment, read and never written. This experiment
writes only under its own `WORK_DIR`.
"""

import json
from pathlib import Path
from typing import Final

import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 350
SLUG: Final[str] = "primary-demand-capacity-free-irradiance-curve"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
FITS_DIR: Final[Path] = WORK_DIR / "fits"
CHAIN_DIR: Final[Path] = WORK_DIR / "chain"
REPORT_DIR: Final[Path] = WORK_DIR / "report"

ARMS: Final[tuple[str, ...]] = ("A", "B", "C")
SCORED_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16", "D9")

IRRADIANCE_KNOTS_W_M2: Final[tuple[float, ...]] = (
    0.0,
    75.0,
    150.0,
    250.0,
    350.0,
    450.0,
    550.0,
    700.0,
)
"""Left edges of the eight saturating ramps; ramp `j` rises from 0 to 1 between knot `j` and
the next."""
IRRADIANCE_TOP_W_M2: Final[float] = 1000.0
"""The right edge of the last ramp."""
PV_RIDGE: Final[float] = 1e-6
"""A ridge on each curve coefficient, only so that the normal equations are positive definite."""
SHAPE_ITERATIONS: Final[int] = 8
"""Alternating rounds of the shared-shape fit (arm B)."""

LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip


def write_json_atomic(*, path: Path, content: dict) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, default=float))
    temporary.rename(path)
