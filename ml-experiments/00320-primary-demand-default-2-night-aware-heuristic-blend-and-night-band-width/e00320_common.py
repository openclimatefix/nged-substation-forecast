# ruff: noqa: E501
"""Constants shared by the scripts of experiment 320 (night-aware blend and night band width).

Every input is a stored file of an earlier experiment, read and never written. This experiment
writes only under its own `WORK_DIR`.
"""

import json
from pathlib import Path
from typing import Final

import e00319_common as c319  # ty: ignore[unresolved-import]
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 320
SLUG: Final[str] = "primary-demand-default-2-night-aware-heuristic-blend-and-night-band-width"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
ARMS_DIR: Final[Path] = WORK_DIR / "arms"
REPORT_DIR: Final[Path] = WORK_DIR / "report"

E319_ARMS: Final[Path] = c319.ARMS_DIR
CHAIN_DIR: Final[Path] = c319.CHAIN_DIR
QUANTILES: Final[tuple[str, ...]] = c319.QUANTILES
LEVELS: Final[np.ndarray] = c319.LEVELS
SCORED_SITES: Final[tuple[str, ...]] = c319.SCORED_SITES
PERIODS: Final[tuple[str, ...]] = c319.PERIODS
WINTER_MONTHS: Final[frozenset[int]] = c319.WINTER_MONTHS
STORED_HORIZON_HOURS: Final[float] = c319.STORED_HORIZON_HOURS
HORIZON_HOURS: Final[float] = c319.HORIZON_HOURS

COLD_FORECAST_C: Final[float] = 2.0
"""Ensemble-mean 24-hour temperature below which a night counts as forecast cold (experiment 317)."""
WIDTH_WINDOW_DAYS: Final[int] = 90
WIDTH_GRID: Final[np.ndarray] = np.round(np.arange(0.6, 1.2001, 0.02), 2)
NIGHT_START_MINUTE: Final[int] = 21 * 60
NIGHT_END_MINUTE: Final[int] = 7 * 60 + 30
TEMPERATURE_CENTRE_C: Final[float] = 10.0
TEMPERATURE_SCALE_C: Final[float] = 5.0
TELEMETRY_DELAY_MINUTES: Final[int] = 210


def write_json_atomic(*, path: Path, content: dict) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, default=float))
    temporary.rename(path)
