"""Constants shared by the scripts of experiment 325 (sunshine response: refit and restriction).

Every earlier experiment's files are read and never written. This experiment writes only under its
own `WORK_DIR`.
"""

import json
from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 325
SLUG: Final[str] = "primary-demand-default-3-sunshine-response-refit-and-restriction"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
FITS_DIR: Final[Path] = WORK_DIR / "fits"
MEMBERS_DIR: Final[Path] = WORK_DIR / "members"
CHAIN_DIR: Final[Path] = WORK_DIR / "chain"
ARM_D_DIR: Final[Path] = WORK_DIR / "arm_d"
NET_DIR: Final[Path] = WORK_DIR / "net"
ARMS_DIR: Final[Path] = WORK_DIR / "arms"
REPORT_DIR: Final[Path] = WORK_DIR / "report"
E323_DATA: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00323-primary-demand-default-3-flexible-sunshine-response-in-the-convex-fit"
    / "work"
)

SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16", "D9")
PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")
HORIZON_HALF_HOURS: Final[int] = 480
N_MEMBERS: Final[int] = 51
RESPONSE_VARIANTS: Final[tuple[str, ...]] = ("f1", "f1s")
"""Variants with a sunshine response; `v0r` (no response, rebuilt) is the control."""
VARIANTS: Final[tuple[str, ...]] = ("v0r", *RESPONSE_VARIANTS)

# The restriction, fixed before any score was read. The response weight is
# clip((index - 0.6) / 0.1, 0, 1) * clip((elevation - 15) / 5, 0, 1).
RESTRICT_INDEX_START: Final[float] = 0.6
RESTRICT_INDEX_RAMP: Final[float] = 0.1
RESTRICT_ELEVATION_START_DEG: Final[float] = 15.0
RESTRICT_ELEVATION_RAMP_DEG: Final[float] = 5.0


def write_json_atomic(*, path: Path, content: dict) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, default=float))
    temporary.rename(path)
