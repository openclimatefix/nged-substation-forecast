# ruff: noqa: E501
"""Paths and constants shared by the scripts of experiment 333 (zero-shot foundation model).

Run every script with experiments 317, 319, 320, and 322 on `PYTHONPATH`:

    export PYTHONPATH=$(ls -d ml-experiments/00317-* ml-experiments/00319-* ml-experiments/00320-* ml-experiments/00322-* ml-experiments/00333-* | paste -sd:)
"""

import json
from pathlib import Path
from typing import Final

import e00319_common as c319  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 333
SLUG: Final[str] = "primary-demand-foundation-model-zero-shot"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
FORECAST_DIR: Final[Path] = WORK_DIR / "forecasts"
REPORT_DIR: Final[Path] = WORK_DIR / "report"

CHAIN_DIR: Final[Path] = c319.CHAIN_DIR
SCORED_SITES: Final[tuple[str, ...]] = c319.SCORED_SITES
QUANTILES: Final[tuple[str, ...]] = c319.QUANTILES
LEVELS: Final[tuple[float, ...]] = tuple(float(x) for x in c319.LEVELS)
MODEL_ID: Final[str] = "amazon/chronos-2"
CONTEXT_HALF_HOURS: Final[int] = 2688
"""Eight weeks of half-hourly history."""
TELEMETRY_DELAY_HALF_HOURS: Final[int] = 7
"""The newest reading at the 09:00 UTC origin ended 3.5 hours earlier (experiment 320)."""
HORIZON_HALF_HOURS: Final[int] = 480
PREDICT_HALF_HOURS: Final[int] = HORIZON_HALF_HOURS + TELEMETRY_DELAY_HALF_HOURS
ARMS: Final[tuple[str, ...]] = ("z0", "z1")


def write_json_atomic(*, path: Path, content: dict) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, default=float))
    temporary.rename(path)
