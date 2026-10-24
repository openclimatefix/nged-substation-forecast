# ruff: noqa: E501
"""Constants shared by the scripts of experiment 322 (daylight-only heuristic blend, sunshine graft).

Every input is a stored file of an earlier experiment, read and never written. This experiment
writes only under its own `WORK_DIR`.
"""

import json
from pathlib import Path
from typing import Final

import e00320_common as c320  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 322
SLUG: Final[str] = "primary-demand-default-2-daylight-only-heuristic-blend-and-sunshine-graft-replication"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
ARMS_DIR: Final[Path] = WORK_DIR / "arms"
REPORT_DIR: Final[Path] = WORK_DIR / "report"

E320_ARMS = c320.ARMS_DIR
E111_STORE: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00111-differentiable-physics-6-ens-forecast-engine-against-bg2"
    / "work"
    / "ens_store"
)
COLD_MEMBER_C: Final[float] = 3.0
"""A member counts as cold when its 24-hour mean temperature at the half-hour is below this."""
FLAG_SHARE: Final[float] = 0.5
"""The share rule flags a half-hour when at least this share of the members is cold."""
FIT_ROWS: Final[int] = 60_000


def write_json_atomic(*, path: Path, content: dict) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, default=float))
    temporary.rename(path)
