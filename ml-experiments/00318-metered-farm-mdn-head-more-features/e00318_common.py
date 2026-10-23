"""Shared constants for experiment 318 (the mixture-density head with more inputs).

Farms appear only as labels A to F. Every table written here lives under `data/ml-experiments/`.
"""

from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00318-metered-farm-mdn-head-more-features"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
PRED_DIR: Final[Path] = DATA_DIR / "predictions"
E310_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00310-metered-farm-long-lead-scoring"
E313_PRED_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00313-metered-farm-pure-xgboost-and-mdn-vs-plant" / "solar_predictions"
)
OWN_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
ZERO_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
LAG_COLUMNS: Final[tuple[str, ...]] = ("lag_power", "lag_missing", "avail_ratio")
ADDED_COLUMNS: Final[tuple[str, ...]] = ("hour_sin", "hour_cos", "doy_sin", "doy_cos", "ghi_std")
PURE_COLUMNS: Final[tuple[str, ...]] = (
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
"""Experiment 313's pure inputs: no plant forecast, no spread of it, no lagged power."""
DEFAULT_INPUTS: Final[tuple[str, ...]] = (
    "mu",
    "plant_spread",
    "avail_ratio",
    "lag_power",
    "lag_missing",
    "csi",
    "lead_hours",
)
"""Experiment 310's mixture-head inputs (`HEAD_INPUTS`), copied so a change there is noticed."""
LEAD_INPUTS: Final[tuple[str, ...]] = (*DEFAULT_INPUTS, "lead_day")
FULL_INPUTS: Final[tuple[str, ...]] = (*LEAD_INPUTS, *ADDED_COLUMNS)
FULL_NOLAG_INPUTS: Final[tuple[str, ...]] = tuple(
    c for c in FULL_INPUTS if c not in LAG_COLUMNS
)

# arm -> (input columns, whether the head's base is the plant forecast, shuffle the added columns)
PER_BAND_ARMS: Final[dict[str, tuple[tuple[str, ...], bool, bool]]] = {
    "default": (DEFAULT_INPUTS, True, False),
    "lead": (LEAD_INPUTS, True, False),
    "full": (FULL_INPUTS, True, False),
    "full_nolag": (FULL_NOLAG_INPUTS, True, False),
    "full_shuffled": (FULL_INPUTS, True, True),
    "pure_plant": ((*PURE_COLUMNS, "mu"), False, False),
}
POOLED_ARMS: Final[dict[str, tuple[tuple[str, ...], bool, bool]]] = {
    "default_pooled": (DEFAULT_INPUTS, True, False),
    "full_pooled": (FULL_INPUTS, True, False),
    "pure_pooled": (PURE_COLUMNS, False, False),
    "pure_lead_pooled": ((*PURE_COLUMNS, "lead_day"), False, False),
}
OWN_ONLY_ARMS: Final[frozenset[str]] = frozenset(
    {"lead", "full_nolag", "full_shuffled", "pure_plant"}
)
"""Per-band arms run in own history only; the pooled arms are also own history only."""
