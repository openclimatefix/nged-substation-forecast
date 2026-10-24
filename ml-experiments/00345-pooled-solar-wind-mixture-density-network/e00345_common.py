# ruff: noqa: E501
"""Shared constants for experiment 345 (one pooled mixture-density network for solar and wind farms).

Solar farms appear only as labels A to F and wind farms as W1 to W3. Public wind farms are never
named. Every table written here lives under `data/ml-experiments/`.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00345-pooled-solar-wind-mixture-density-network"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
PRED_DIR: Final[Path] = DATA_DIR / "solar_predictions"
WIND_DIR: Final[Path] = DATA_DIR / "wind_predictions"
PREP_DIR: Final[Path] = DATA_DIR / "prep"
E310_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00310-metered-farm-long-lead-scoring"
E313_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00313-metered-farm-pure-xgboost-and-mdn-vs-plant"
E318_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00318-metered-farm-mdn-head-more-features"
E288_WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00288-pooled-head-on-public-wind-farms" / "work"

OWN_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
ZERO_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
FEW_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
"""Seeds of the controls and of the monthly-refit arm."""
WIND_FARMS: Final[tuple[str, ...]] = ("W1", "W2", "W3")
WIND_TRAIN_END: Final[datetime] = datetime(2025, 6, 17, tzinfo=UTC)
"""Wind training rows and the solar rows that train the wind-scoring networks end here."""
PUBLIC_END: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
VALIDATION_DAYS: Final[int] = 61
EPOCHS: Final[int] = 20
BATCH: Final[int] = 2048
LEARNING_RATE: Final[float] = 2e-3
WEIGHT_DECAY: Final[float] = 1e-4

SHARED: Final[tuple[str, ...]] = (
    "temperature",
    "wind10",
    "wind100",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
)
SOLAR_ONLY: Final[tuple[str, ...]] = (
    "ghi_mean",
    "ghi_std",
    "dni_mean",
    "dhi_mean",
    "elevation_deg",
    "azimuth_sin",
    "azimuth_cos",
    "relative_humidity",
    "longwave",
    "precipitation",
    "pressure_surface",
    "pressure_msl",
    "z500",
    "csi",
    "lead_hours",
)
WIND_ONLY: Final[tuple[str, ...]] = ("dir_sin10", "dir_cos10", "dir_sin100", "dir_cos100")
FLAGS: Final[tuple[str, ...]] = ("is_wind", "temp_missing")
"""Columns that are 0 or 1 and are never standardised."""
PLANT: Final[tuple[str, ...]] = ("mu",)


@dataclass(frozen=True)
class Arm:
    """One network design: which rows train it and which columns it reads."""

    name: str
    columns: tuple[str, ...]
    solar_rows: bool
    wind_rows: bool
    shuffle_fuel: bool = False
    shuffle_wind_target: bool = False


SOLAR_POOLED: Final = Arm("solar_pooled", (*SHARED, *SOLAR_ONLY, "temp_missing"), True, False)
WIND_POOLED: Final = Arm("wind_pooled", (*SHARED, *WIND_ONLY, "temp_missing"), False, True)
BOTH_COLUMNS: Final[tuple[str, ...]] = (*SHARED, *SOLAR_ONLY, *WIND_ONLY, *FLAGS)
BOTH: Final = Arm("both", BOTH_COLUMNS, True, True)
ARMS: Final[dict[str, Arm]] = {
    a.name: a
    for a in (
        SOLAR_POOLED,
        WIND_POOLED,
        BOTH,
        Arm("both_plant", (*BOTH_COLUMNS, *PLANT), True, True),
        Arm("both_shuf_fuel", BOTH_COLUMNS, True, True, shuffle_fuel=True),
        Arm("both_shuf_wind", BOTH_COLUMNS, True, True, shuffle_wind_target=True),
        Arm("both_once", BOTH_COLUMNS, True, True),
        Arm("both_monthly", BOTH_COLUMNS, True, True),
    )
}
SOLAR_ARM_NAMES: Final[tuple[str, ...]] = (
    "solar_pooled",
    "both",
    "both_plant",
    "both_shuf_fuel",
    "both_shuf_wind",
    "both_once",
    "both_monthly",
)
WIND_ARM_NAMES: Final[tuple[str, ...]] = ("wind_pooled", "both", "both_shuf_fuel", "both_shuf_wind")
CONTROL_ARMS: Final[frozenset[str]] = frozenset({"both_shuf_fuel", "both_shuf_wind"})
OWN_ONLY_SOLAR: Final[frozenset[str]] = frozenset(
    {"both_shuf_fuel", "both_shuf_wind", "both_monthly"}
)
"""Solar arms run in own history only; the cadence arm `both_once` runs in both settings."""
