"""Paths and constants of experiment 00079 (PVDIS-7).

The experiment asks whether the direction of flow, and the solar capacity, can be recovered from
the magnitude of the power at a Primary substation. The plan is
`docs/ml-experiments/00079-pvdis-7-mw-at-mva-only-primaries/result.md`. Every script runs on four
cores:

    POLARS_MAX_THREADS=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --with holidays python ...

Experiment 00039's frame is read and never written.
"""

from pathlib import Path
from typing import Final

from studies.sources import CAMS_SITE_POINTS_DIR, ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 79
SLUG: Final[str] = "pvdis-7-mw-at-mva-only-primaries"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables; private and never committed."""
CHART_DIR: Final[Path] = WORK_DIR / "charts"
SHAPES_PATH: Final[Path] = WORK_DIR / "shapes.parquet"
KNOWN_PATH: Final[Path] = WORK_DIR / "known.parquet"
REAL_PATH: Final[Path] = WORK_DIR / "real.parquet"
REAL_BOOT_PATH: Final[Path] = WORK_DIR / "real_boot.parquet"
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
SUMMER_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)
MIN_WINDOW_ROWS: Final[int] = 1000
HOUR_BANDS: Final[tuple[tuple[int, int], ...]] = ((0, 20), (20, 28), (28, 48))
RATIO: Final[float] = 1.15
"""DC:AC ratio of the clipped solar shape (the fleet ratio)."""
TILT: Final[float] = 25.0
AZIMUTHS: Final[dict[str, float]] = {"E": 90.0, "S": 180.0, "W": 270.0}
FLEET_MIX: Final[dict[str, float]] = {"E": 0.2, "S": 0.6, "W": 0.2}

MW_PRIMARIES: Final[tuple[str, ...]] = (
    "D5", "D9", "D10", "D11", "D14", "D15", "D16", "D18",
)  # fmt: skip
ALMOST_NEVER_NEGATIVE: Final[tuple[str, ...]] = ("D5", "D9", "D11", "D14", "D16")
MVA_PRIMARIES: Final[tuple[str, ...]] = (
    "D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22",
)  # fmt: skip
PRIMARY_NAMES: Final[dict[str, str]] = {
    "D5": "Kirton", "D9": "Trusthorpe", "D10": "Spilsby", "D11": "Horncastle",
    "D14": "Mount Bridge Boston", "D15": "Wrangle T1", "D16": "Donington", "D18": "Wrangle T2",
    "D2": "Warth Lane Skegness", "D3": "Sleaford Road Boston", "D4": "Ingoldmells",
    "D8": "Marsh Lane", "D12": "Stickney", "D19": "Chapel St Leonards", "D21": "Alford",
    "D22": "Langrick",
}  # fmt: skip
FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F")

LEVELS: Final[tuple[float, ...]] = (0.5, 1.0, 1.5)
"""Injected solar capacity as a multiple of the window's median absolute midday power."""
MIDDAY_HALF_HOURS: Final[tuple[int, int]] = (22, 28)
"""Local half-hours 22 to 27 are 11:00 to 13:30; half-hour `h` starts at `h / 2` hours."""
REACTIVE_MAIN: Final[float] = 0.2
"""The reactive-power floor `q` that enters contrast C3, as a share of midday power."""
START_MULTIPLES: Final[tuple[float, ...]] = (0.5, 1.0, 1.5, 2.0)
MAX_ITERATIONS: Final[int] = 100
TIE_MARGIN: Final[float] = 0.005
"""A sign-flipping start must lower the all-positive start's loss by this share to win."""
REACTIVE_SHARES: Final[tuple[float, ...]] = (0.1, 0.2, 0.4)
SIGN_FLOOR_SHARE: Final[float] = 0.02
"""Rows with an absolute true flow under this share of the window's median absolute midday
power are not scored for sign, because the true sign is not meaningful there."""
MIN_CLASS_ROWS: Final[int] = 50
"""A window needs this many reverse-flow and this many forward-flow half-hours for a sign score."""
FALSE_REVERSE_LIMIT: Final[float] = 0.02
CAPACITY_ERROR_LIMIT: Final[float] = 0.25
BALANCED_ACCURACY_LIMIT: Final[float] = 0.80
C2_LOWER_LIMIT: Final[float] = 0.25
BOOTSTRAP_DRAWS: Final[int] = 4000
BOOTSTRAP_SEED: Final[int] = 79
LEVEL: Final[float] = 1 - 0.05 / 5
"""The Bonferroni level for the five planned contrasts."""
WEEK_BOOTSTRAP_DRAWS: Final[int] = 200
BLOCK_DAYS: Final[int] = 7
