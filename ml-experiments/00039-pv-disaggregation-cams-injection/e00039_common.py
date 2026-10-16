"""Paths, constants, and labels of experiment 00039.

The experiment asks whether a calendar-plus-temperature demand term minus one PV shape driven by
CAMS irradiance can separate embedded solar from the demand of the 8 directional Primary
substations
(`docs/ml-experiments/00039-pv-disaggregation-cams-injection/result.md`). Every script runs on
two cores:

    POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run --with holidays python ...
"""

from pathlib import Path
from typing import Final

from studies.pv_physics import PlantParameters
from studies.sources import (
    CAMS_SITE_POINTS_DIR,
    ML_EXPERIMENTS_DATA_DIR,
    REPO_DATA_DIR,
    study_dir_for,
)

EXPERIMENT_ID: Final[int] = 39
SLUG: Final[str] = "pv-disaggregation-cams-injection"

WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables; private and never committed."""
CHART_DIR: Final[Path] = WORK_DIR / "charts"
FRAME_PATH: Final[Path] = WORK_DIR / "frame.parquet"
SITES_PATH: Final[Path] = WORK_DIR / "sites.parquet"
INJECTION_PATH: Final[Path] = WORK_DIR / "injection.parquet"
INJECTION_HALF_HOURLY_PATH: Final[Path] = WORK_DIR / "injection_half_hourly.parquet"
REAL_PATH: Final[Path] = WORK_DIR / "real.parquet"
PLACEBO_PATH: Final[Path] = WORK_DIR / "placebo.parquet"
SHIFT_PATH: Final[Path] = WORK_DIR / "timestamp_scan.parquet"
REPORT_PATH: Final[Path] = WORK_DIR / "report.txt"

DEMAND_INPUTS_DIR: Final[Path] = study_dir_for(study="stage1_pilot") / "demand" / "inputs"
SERIES_PATH: Final[Path] = DEMAND_INPUTS_DIR / "series.parquet"
ERA5_PATH: Final[Path] = DEMAND_INPUTS_DIR / "era5_features.parquet"
CAMS_DEMAND_POINTS_PATH: Final[Path] = (
    CAMS_SITE_POINTS_DIR.parent / "demand_points" / "cams_demand_points.parquet"
)
CAMS_FARM_POINTS_PATH: Final[Path] = CAMS_SITE_POINTS_DIR / "beam_diffuse_cams.parquet"
HIERARCHY_DIR: Final[Path] = REPO_DATA_DIR / "NGED" / "substation_hierarchy_2026-10-15"
METADATA_PATH: Final[Path] = REPO_DATA_DIR / "NGED" / "metadata.parquet"

FLEET: Final[PlantParameters] = PlantParameters(
    tilt_deg=35.0, azimuth_deg=180.0, dc_capacity_mw=1.15, ac_capacity_mw=1.0
)
"""The generic fleet whose output per MW of AC capacity is the unit PV shape `u`."""
MAX_ZENITH_DEG: Final[float] = 75.0
MIN_RELIABILITY: Final[float] = 0.9
HDD_BASE_C: Final[float] = 15.5
MIN_WINDOW_ROWS: Final[int] = 3000
MIN_DAILY_ROWS: Final[int] = 10
DROPOUT_GHI_W_M2: Final[float] = 100.0
"""A farm half-hour at or below zero power while CAMS shows more than this is a dropout."""
SUMMER_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)

DIRECTIONAL_PRIMARIES: Final[tuple[str, ...]] = (
    "D5", "D9", "D10", "D11", "D14", "D15", "D16", "D18",
)  # fmt: skip
"""The 8 Primaries whose meter reads a signed MW flow, so PV above demand does not fold."""
PRIMARY_NAMES: Final[dict[str, str]] = {
    "D5": "Kirton",
    "D9": "Trusthorpe",
    "D10": "Spilsby",
    "D11": "Horncastle",
    "D14": "Mount Bridge Boston",
    "D15": "Wrangle T1",
    "D16": "Donington",
    "D18": "Wrangle T2",
    "D2": "Warth Lane Skegness",
    "D3": "Sleaford Road Boston",
    "D4": "Ingoldmells",
    "D8": "Marsh Lane",
    "D12": "Stickney",
    "D19": "Chapel St Leonards",
    "D21": "Alford",
    "D22": "Langrick",
    "D17": "Boston (BSP)",
    "D13": "Skegness (BSP)",
}
"""Substation names; substations may be named on a published chart."""
MVA_PRIMARIES: Final[tuple[str, ...]] = (
    "D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22",
)  # fmt: skip
BSP_SITES: Final[tuple[str, ...]] = ("D17", "D13")
"""The Boston and Skegness bulk supply points."""
BSP_FARMS: Final[dict[str, tuple[str, ...]]] = {"D17": ("B", "D", "F"), "D13": ("A", "C", "E")}
"""The anonymised solar farms whose output joins each bulk supply point's flow."""
FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F")
FARM_NAMES: Final[dict[str, str]] = {f: f"Solar farm {f}" for f in FARMS}

INJECTION_SIZES: Final[tuple[float, ...]] = (0.05, 0.10, 0.20, 0.40)
"""Injected P99 as a share of the Primary's effective capacity."""
HEADLINE_SIZE: Final[float] = 0.20
BOOTSTRAP_DRAWS: Final[int] = 2000
BOOTSTRAP_SEED: Final[int] = 39
PLACEBO_DRAWS: Final[int] = 20
PLACEBO_SEED: Final[int] = 390
SHIFTS: Final[tuple[int, ...]] = (-2, -1, 0, 1, 2)

RECOVERY_BAND: Final[tuple[float, float]] = (0.75, 1.25)
RECOVERY_INTERVAL_LIMITS: Final[tuple[float, float]] = (0.65, 1.35)
MAX_NMAE: Final[float] = 0.15
K2_TOLERANCE: Final[float] = 0.15
K3_TOLERANCE: Final[float] = 0.20
K4_TOLERANCE: Final[float] = 0.20
