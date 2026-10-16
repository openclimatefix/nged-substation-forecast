"""Paths, constants, and labels of experiment 00040.

The experiment asks whether the disagreement between six least-squares fits of embedded-solar
capacity detects a cloud-driven demand load, and so lets us attach an error bar to the capacity of
the directional Primary substations that show none
(`docs/ml-experiments/00040-pv-capacity-error-bars/result.md`). Every script runs on two cores:

    POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run --with holidays python ...

The input frame and the 00039 reports are read from experiment 00039's private work folder and
never written.
"""

from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 40
SLUG: Final[str] = "pv-capacity-error-bars"

WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables; private and never committed."""
CHART_DIR: Final[Path] = WORK_DIR / "charts"
UNIT_VECTORS_PATH: Final[Path] = WORK_DIR / "unit_vectors.parquet"
REAL_VECTORS_PATH: Final[Path] = WORK_DIR / "real_vectors.parquet"
BOOTSTRAP_PATH: Final[Path] = WORK_DIR / "bootstrap.parquet"
UNIT_SCORES_PATH: Final[Path] = WORK_DIR / "unit_scores.parquet"
SCORES_PATH: Final[Path] = WORK_DIR / "scores.parquet"
REAL_SCORES_PATH: Final[Path] = WORK_DIR / "real_scores.parquet"
REPORT_PATH: Final[Path] = WORK_DIR / "report.txt"

INPUT_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00039-pv-disaggregation-cams-injection" / "work"
"""Experiment 00039's private work folder, read and never written."""
FRAME_PATH: Final[Path] = INPUT_DIR / "frame.parquet"
SITES_PATH: Final[Path] = INPUT_DIR / "sites.parquet"
REPORT_CHECKS_PATH: Final[Path] = INPUT_DIR / "report_checks.parquet"
REAL_00039_PATH: Final[Path] = INPUT_DIR / "real.parquet"
TIMESTAMP_SCAN_PATH: Final[Path] = INPUT_DIR / "timestamp_scan.parquet"
FRAME_ROWS: Final[int] = 2_891_184

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

ARMS: Final[tuple[str, ...]] = ("M1", "M1L", "M2", "M2h", "M1_s", "M2h_s")
"""The six detector arms, in the order of every arm vector."""
ARM_M1D: Final[str] = "M1d"
ALL_FITS: Final[tuple[str, ...]] = (*ARMS, ARM_M1D)
"""The six arms followed by the daily fit, which gives the headline and is not a detector arm."""
HOUR_BANDS: Final[tuple[tuple[int, int], ...]] = ((0, 20), (20, 28), (28, 48))
"""Local half-hour bands (start inclusive, end exclusive) of arms M2h and M2h_s."""
SHAPES: Final[tuple[str, ...]] = (
    "flat", "lowsun", "winter", "dayL", "evening", "threshold", "mixture", "sunload",
)  # fmt: skip
OUT_OF_ARM_SHAPES: Final[tuple[str, ...]] = ("lowsun", "dayL", "evening", "threshold", "mixture")
"""The shapes no arm contains; flat and winter are in some arms and sunload is the blind spot."""
EVENING_HALF_HOUR: Final[int] = 30
THRESHOLD_CLOUD: Final[float] = 0.6
AMPLITUDES: Final[tuple[float, ...]] = (0.1, 0.25, 0.5, 1.0)
POWER_AMPLITUDE: Final[float] = 0.25
BOOTSTRAP_DRAWS: Final[int] = 200
BLOCK_DAYS: Final[int] = 7
BOOTSTRAP_SEED: Final[int] = 40
FLAG_QUANTILE: Final[float] = 0.95
CALIBRATION_BAND: Final[tuple[float, float]] = (0.9, 1.1)
MAX_FALSE_POSITIVE_RATE: Final[float] = 0.10
MIN_POWER: Final[float] = 0.8
MAX_BOUND: Final[float] = 0.25
MIN_CLEAN_PRIMARIES: Final[int] = 6
K1_ALTERNATIVE_CSE: Final[float] = 4.0
"""The descriptive replacement for K1's second clause: capacity over standard error above this."""
CLUSTER_BOOTSTRAP_DRAWS: Final[int] = 1000
CLUSTER_BOOTSTRAP_SEED: Final[int] = 400
