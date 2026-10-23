# ruff: noqa: E501
"""Shared constants for experiment 310 (the farm forecast at leads of 1 to 10 days).

Farms appear only as labels A to F. Every table written here lives under `data/ml-experiments/`.
Day `d` takes the 00 UTC ensemble run whose lead for a half-hour is `12 + 24(d-1)` to
`36 + 24(d-1)` hours. A day band pools one or more days and is fitted on its own.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
from studies.ens_members import MIXTURE_LEVELS
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00310-metered-farm-long-lead-scoring"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
MEMBERS_DIR: Final[Path] = DATA_DIR / "members"
PRED_DIR: Final[Path] = DATA_DIR / "predictions"
E270_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00270-metered-farm-xgboost-vs-plant-diagnostic"
FARM_ROWS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00218-metered-solar-farm-plant-fit" / "farm_rows_cams.parquet"
)
FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F")
SETTINGS: Final[tuple[str, ...]] = ("own_history", "zero_history")
LEVELS: Final[np.ndarray] = np.array(MIXTURE_LEVELS)
N_MEMBERS: Final[int] = 51
ENS_START: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
GRID_END: Final[str] = "2026-09-01"
SCORED_START: Final[datetime] = datetime(2025, 9, 1, tzinfo=UTC)
SCORED_END: Final[datetime] = datetime(2026, 9, 1, tzinfo=UTC)
DAYS: Final[tuple[int, ...]] = tuple(range(1, 11))
BANDS: Final[dict[str, tuple[int, ...]]] = {
    "d1": (1,),
    "d2": (2,),
    "d3": (3,),
    "d4_5": (4, 5),
    "d6_7": (6, 7),
    "d8_10": (8, 9, 10),
}
BAND_ORDER: Final[tuple[str, ...]] = tuple(BANDS)
THREE_HOURLY_UNTIL_HOURS: Final[int] = 144
"""The ensemble store's stamps are 3-hourly up to this lead and 6-hourly after it."""
CEILING_QUANTILE: Final[float] = 0.995
MIN_BIN_ROWS: Final[int] = 50
MIN_FORECAST_POWER: Final[float] = 0.02
N_BINS: Final[int] = 4
HEAT_PER_KELVIN: Final[float] = 0.004
FLEET_TILT_DEG: Final[float] = 35.0
FLEET_AZIMUTH_DEG: Final[float] = 180.0
CELL_TEMPERATURE_RISE_K: Final[float] = 25.0
REFERENCE_CELL_TEMPERATURE_C: Final[float] = 25.0


def table_path(*, site: str, day: int) -> Path:
    """Return the parquet path of one (farm, day) table."""
    return MEMBERS_DIR / f"{site}_d{day}.parquet"


def unit_path(*, site: str, day: int) -> Path:
    """Return the npy path of one (farm, day) member-response array."""
    return MEMBERS_DIR / f"{site}_d{day}_unit.npy"
