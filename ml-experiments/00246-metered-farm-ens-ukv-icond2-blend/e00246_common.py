"""Shared constants for experiment 246 (ECMWF ensemble, UKV, and ICON-D2 for a metered solar farm).

Farms appear only as labels A to F. Every table written here lives under `data/ml-experiments/`.
The ensemble member tables are copied from experiment 224's data folder, which this experiment reads
and does not change.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import polars as pl
from studies.ens_members import MIXTURE_LEVELS
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

SLUG: Final[str] = "00246-metered-farm-ens-ukv-icond2-blend"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
MEMBERS_DIR: Final[Path] = DATA_DIR / "members"
UKV_DIR: Final[Path] = DATA_DIR / "ukv"
ARMS_DIR: Final[Path] = DATA_DIR / "predictions"
NWP_ROOT: Final[Path] = REPO_DATA_DIR / "studies" / "downloads" / "NWP"
NWP_DIR: Final[Path] = NWP_ROOT
ICON_PATH: Final[Path] = (
    NWP_ROOT / "OPEN-METEO-PREVIOUS-RUNS" / "ICON-D2" / "previous_runs" / "combined.parquet"
)
FARM_ROWS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00218-metered-solar-farm-plant-fit" / "farm_rows_cams.parquet"
)
UKV_STORES: Final[tuple[str, ...]] = ("UKV-CEDA", "UKV-CEDA-part2", "UKV-CEDA-part3")
FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F")
N_MEMBERS: Final[int] = 51
BANDS: Final[tuple[int, ...]] = (0,)
"""Band 0 holds valid times 12 to 29 hours after the ensemble run's 00 UTC start.

The latest valid time with an ICON-D2 run available at 08 UTC is 29 hours out: the run is made 24
hours before the valid time and published 3 hours after it starts.
"""
BAND_LEAD_RANGE: Final[dict[int, tuple[float, float]]] = {0: (12.0, 29.0)}
FOLDS: Final[dict[str, tuple[datetime, datetime]]] = {
    "scored": (datetime(2025, 9, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC)),
    "replicate": (datetime(2025, 3, 1, tzinfo=UTC), datetime(2025, 9, 1, tzinfo=UTC)),
}
"""Each fold scores `[start, end)` and tunes on every ensemble month before `start`. Both lie in the
second UKV era (2020-01 to 2025-12), so no fit and score straddle a UKV physics change."""
ENS_PUBLICATION_DELAY_HOURS: Final[int] = 8
UKV_PUBLICATION_DELAY_HOURS: Final[int] = 4
LEVELS: Final[tuple[float, ...]] = MIXTURE_LEVELS
CEILING_QUANTILE: Final[float] = 0.995
UKV_SLOTS_PER_DAY: Final[int] = 4
UKV_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
UKV_VARIABLES: Final[tuple[str, ...]] = (
    "shortwave_down",
    "longwave_down",
    "cloud_total",
    "cloud_low",
    "cloud_medium",
    "cloud_high",
    "cloud_very_low",
    "cloud_base_height",
    "convective_cloud_top_height",
    "temperature_1p5m",
    "temperature_0m",
    "dew_point_1p5m",
    "relative_humidity_1p5m",
    "visibility_1p5m",
    "visibility_below_1km_probability",
    "precipitation_rate",
    "precipitation_amount",
    "wind_speed_10m",
    "wind_direction_10m",
    "gust_10m",
    "pressure_msl",
    "snow_depth",
)
UKV_IRRADIANCE_AND_CLOUD: Final[tuple[str, ...]] = (
    "shortwave_down",
    "cloud_total",
    "cloud_low",
    "cloud_medium",
    "cloud_high",
    "cloud_very_low",
)
NATIVE_STEPS: Final[tuple[int, ...]] = (*range(49), 51, 54)
"""The leads the UKV-CEDA files hold: hourly to 48 hours, then 3-hourly to 54."""
SUNNY_CLEAR_SKY_W_M2: Final[float] = 650.0


def members_table(*, site: str, band: int) -> Path:
    """Return the path of a farm and band's ensemble table."""
    return MEMBERS_DIR / f"{site}_b{band}.parquet"


def members_unit(*, site: str, band: int) -> Path:
    """Return the path of a farm and band's array of unit responses to the 51 members."""
    return MEMBERS_DIR / f"{site}_b{band}_unit.npy"


ENS_START: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
BAND_SHIFT_DAYS: Final[tuple[int, ...]] = (0,)
"""Band `j` takes the run whose window holds the half-hour at leads 12 + 24j to 36 + 24j hours."""
HEAT_PER_KELVIN: Final[float] = 0.004
FLEET_TILT_DEG: Final[float] = 35.0
FLEET_AZIMUTH_DEG: Final[float] = 180.0
CELL_TEMPERATURE_RISE_K: Final[float] = 25.0
REFERENCE_CELL_TEMPERATURE_C: Final[float] = 25.0


def farm_output() -> pl.DataFrame:
    """Return the metered output over capacity per daylight half-hour: `site`, `time`, `output`."""
    return pl.read_parquet(FARM_ROWS).select("site", "time", "output", "clear_sky")


ICON_DELAY_HOURS: Final[int] = 3
ICON_COLUMNS: Final[tuple[str, ...]] = (
    "shortwave_radiation_previous_day1",
    "direct_radiation_previous_day1",
    "temperature_2m_previous_day1",
    "wind_speed_10m_previous_day1",
)
