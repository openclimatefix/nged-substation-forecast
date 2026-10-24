"""Shared constants for experiment 313 (pure XGBoost and mixture-density predictors vs the plant).

Solar farms appear only as labels A to F, wind farms as W1 to W3. Every table written here lives
under `data/ml-experiments/`.
"""

from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00313-metered-farm-pure-xgboost-and-mdn-vs-plant"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
SOLAR_PRED_DIR: Final[Path] = DATA_DIR / "solar_predictions"
WIND_DIR: Final[Path] = DATA_DIR / "wind"
E310_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00310-metered-farm-long-lead-scoring"
SOLAR_PURE_COLUMNS: Final[tuple[str, ...]] = (
    "ghi_mean",
    "ghi_std",
    "dni_mean",
    "dhi_mean",
    "temperature_mean",
    "elevation_deg",
    "azimuth_sin",
    "azimuth_cos",
    "hour",
    "doy",
    "relative_humidity",
    "longwave",
    "precipitation",
    "pressure_surface",
    "wind10",
    "wind100",
    "pressure_msl",
    "z500",
    "csi",
    "lead_hours",
)
"""Experiment 310's rung-six columns without the plant forecast, its spread, and the lag columns."""
PLANT_DERIVED: Final[frozenset[str]] = frozenset(
    {"mu", "plant_spread", "avail_ratio", "unit_mean", "lag_power", "lag_missing"}
)
