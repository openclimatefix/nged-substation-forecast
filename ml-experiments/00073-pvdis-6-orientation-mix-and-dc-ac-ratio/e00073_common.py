"""Paths and constants of experiment 00073 (PVDIS-6).

The experiment asks whether the mix of east-, south-, and west-facing solar behind each Primary
substation, and the ratio of DC to AC capacity, can be recovered from the Primary's power. The plan
is `docs/ml-experiments/00073-pvdis-6-orientation-mix-and-dc-ac-ratio/result.md`. Every script runs
on six cores:

    POLARS_MAX_THREADS=6 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --with holidays python ...

Experiment 00039's frame is read and never written.
"""

from pathlib import Path
from typing import Final

from studies.sources import CAMS_SITE_POINTS_DIR, ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 73
SLUG: Final[str] = "pvdis-6-orientation-mix-and-dc-ac-ratio"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables; private and never committed."""
CHART_DIR: Final[Path] = WORK_DIR / "charts"
SHAPES_PATH: Final[Path] = WORK_DIR / "shapes.parquet"
KNOWN_ORIENTATION_PATH: Final[Path] = WORK_DIR / "known_orientation.parquet"
KNOWN_DCAC_PATH: Final[Path] = WORK_DIR / "known_dcac.parquet"
GATE_PATH: Final[Path] = WORK_DIR / "knee_gate.parquet"
REAL_PATH: Final[Path] = WORK_DIR / "real.parquet"
HELDOUT_PATH: Final[Path] = WORK_DIR / "heldout.parquet"
REPORT_PATH: Final[Path] = WORK_DIR / "report.txt"

INPUT_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00039-pv-disaggregation-cams-injection" / "work"
FRAME_PATH: Final[Path] = INPUT_DIR / "frame.parquet"
SITES_PATH: Final[Path] = INPUT_DIR / "sites.parquet"
CAMS_FARM_POINTS_PATH: Final[Path] = CAMS_SITE_POINTS_DIR / "beam_diffuse_cams.parquet"
CAMS_DEMAND_POINTS_PATH: Final[Path] = (
    CAMS_SITE_POINTS_DIR.parent / "demand_points" / "cams_demand_points.parquet"
)
FRAME_ROWS: Final[int] = 2_891_184

MAX_ZENITH_DEG: Final[float] = 75.0
MIN_RELIABILITY: Final[float] = 0.9
HDD_BASE_C: Final[float] = 15.5
DROPOUT_GHI_W_M2: Final[float] = 100.0
SUMMER_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)
MIN_WINDOW_ROWS: Final[int] = 1000
HOUR_BANDS: Final[tuple[tuple[int, int], ...]] = ((0, 20), (20, 28), (28, 48))
CLEAR_CLOUD_MAX: Final[float] = 0.15
MIN_CLEAR_DAYS: Final[int] = 20
FLEET_RATIO: Final[float] = 1.15

DIRECTIONAL_PRIMARIES: Final[tuple[str, ...]] = (
    "D5", "D9", "D10", "D11", "D14", "D15", "D16", "D18",
)  # fmt: skip
PRIMARY_NAMES: Final[dict[str, str]] = {
    "D5": "Kirton",
    "D9": "Trusthorpe",
    "D10": "Spilsby",
    "D11": "Horncastle",
    "D14": "Mount Bridge Boston",
    "D15": "Wrangle T1",
    "D16": "Donington",
    "D18": "Wrangle T2",
}
FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F")
TUNING_FARMS: Final[tuple[str, ...]] = ("A", "B", "C")
SCORING_FARMS: Final[tuple[str, ...]] = ("D", "E", "F")
FARM_DC_AC: Final[dict[str, float]] = {
    "A": 1.15, "B": 1.17, "C": 1.22, "D": 1.42, "E": 1.12, "F": 1.26,
}  # fmt: skip
"""Experiment 00029's PyTorch DC:AC ratios of the six metered farms."""

TILTS: Final[tuple[float, ...]] = (25.0, 15.0, 40.0)
MAIN_TILT: Final[float] = 25.0
AZIMUTHS: Final[dict[str, float]] = {"E": 90.0, "S": 180.0, "W": 270.0}
FLEET_MIX: Final[tuple[float, float, float]] = (0.2, 0.6, 0.2)
MIXES: Final[tuple[tuple[float, float, float], ...]] = (
    (0.0, 1.0, 0.0),
    (0.2, 0.6, 0.2),
    (0.5, 0.4, 0.1),
    (0.1, 0.4, 0.5),
    (0.5, 0.0, 0.5),
)
INJECTED_MW: Final[float] = 3.0
MIN_INCREMENT_MW: Final[float] = 0.3
NO_INCREMENT_ERROR: Final[float] = 0.5
RIDGE_GRID: Final[tuple[float, ...]] = (0.0, 0.1, 1.0, 10.0)
RATIO_GRID: Final[tuple[float, ...]] = tuple(round(1.0 + 0.05 * k, 2) for k in range(13))
DCAC_INJECTED_MW: Final[float] = 10.0
KNEE_GATE_HALF_HOURS: Final[int] = 100
BOOTSTRAP_DRAWS: Final[int] = 4000
BOOTSTRAP_SEED: Final[int] = 73
BLOCK_DAYS: Final[int] = 7
HELDOUT_BLOCK_DAYS: Final[int] = 28
MIN_CELL_ROWS: Final[int] = 5
LEVEL: Final[float] = 1 - 0.05 / 4
"""The Bonferroni level for the four planned contrasts."""
