"""Constants shared by the scripts of experiment 323 (flexible sunshine response in the convex fit).

Every input is a stored file of an earlier experiment, read and never written. This experiment
writes only under its own `WORK_DIR`.
"""

import json
from pathlib import Path
from typing import Final

import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 323
SLUG: Final[str] = "primary-demand-default-3-flexible-sunshine-response-in-the-convex-fit"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
FITS_DIR: Final[Path] = WORK_DIR / "fits"
MEMBERS_DIR: Final[Path] = WORK_DIR / "members"
ARMS_DIR: Final[Path] = WORK_DIR / "arms"
REPORT_DIR: Final[Path] = WORK_DIR / "report"

E317_CHAIN: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00317-primary-demand-default-1-at-7-to-10-days" / "work" / "chain"
)
E322_ARMS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00322-primary-demand-default-2-daylight-only-heuristic-blend-and-sunshine-graft-replication"
    / "work"
    / "arms"
)
E202_ARMS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00202-dp-ens-two-number-rolling-replay" / "work" / "arms"
)

PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")
VARIANTS: Final[tuple[str, ...]] = ("f1", "f2")
HORIZON_HALF_HOURS: Final[int] = 480
N_MEMBERS: Final[int] = 51

# Pre-registered smoothing, set before any score was read.
LAMBDA_DIFFERENCE: Final[float] = 100.0
LAMBDA_RIDGE: Final[float] = 10.0
CSI_KNOTS: Final[np.ndarray] = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.2])
ELEVATION_KNOTS_DEG: Final[np.ndarray] = np.array([0.0, 5.0, 15.0, 30.0, 45.0, 65.0])
"""The sun-elevation knots; the response is zero at the first knot, which has no column."""
F2_ITERATIONS: Final[int] = 6


def write_json_atomic(*, path: Path, content: dict) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, default=float))
    temporary.rename(path)
