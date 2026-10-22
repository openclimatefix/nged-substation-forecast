"""Shared constants and the farm-output reader for experiment 224 (two-number day-ahead farm forecast).

Farms appear only as labels A to F. All tables written here live under `data/ml-experiments/`.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import polars as pl
from studies.ens_members import MIXTURE_LEVELS
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00224-metered-farm-dayahead-two-number"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
MEMBERS_DIR: Final[Path] = DATA_DIR / "members"
ARMS_DIR: Final[Path] = DATA_DIR / "predictions"
FARM_ROWS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00218-metered-solar-farm-plant-fit" / "farm_rows_cams.parquet"
)
FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F")
N_MEMBERS: Final[int] = 51
ENS_START: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
BAND_SHIFT_DAYS: Final[tuple[int, ...]] = (0, 2, 4)
"""Band `j` takes the run whose window holds the half-hour at leads 12 + 24j to 36 + 24j hours."""
HEADLINE_BAND: Final[int] = 0
FOLDS: Final[dict[str, tuple[datetime, datetime]]] = {
    "scored_year": (datetime(2025, 9, 1, tzinfo=UTC), datetime(2026, 9, 1, tzinfo=UTC)),
    "replicate": (datetime(2025, 3, 1, tzinfo=UTC), datetime(2025, 9, 1, tzinfo=UTC)),
}
"""Each fold scores `[start, end)` and tunes on every ensemble month before `start`."""
LEVELS: Final[tuple[float, ...]] = MIXTURE_LEVELS
CEILING_QUANTILE: Final[float] = 0.995
HEAT_PER_KELVIN: Final[float] = 0.004
FLEET_TILT_DEG: Final[float] = 35.0
FLEET_AZIMUTH_DEG: Final[float] = 180.0
CELL_TEMPERATURE_RISE_K: Final[float] = 25.0
REFERENCE_CELL_TEMPERATURE_C: Final[float] = 25.0


def farm_output() -> pl.DataFrame:
    """Return the metered output over capacity per daylight half-hour: `site`, `time`, `output`."""
    return pl.read_parquet(FARM_ROWS).select("site", "time", "output", "clear_sky")
