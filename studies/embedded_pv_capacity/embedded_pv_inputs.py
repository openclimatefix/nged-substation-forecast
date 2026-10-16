"""Paths, labels, and constants shared by the scripts of the embedded-solar capacity study.

The study fits nothing. It reads the arm vectors that experiment 00040 stored (and the frame of
experiment 00039) and builds the capacity table from them.
"""

from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR, study_dir_for

STUDY_DIR: Final[Path] = study_dir_for(study="embedded_pv_capacity")
"""Where this study writes its tables, reports, and charts."""
CHART_DIR: Final[Path] = STUDY_DIR / "charts"
TABLE_PATH: Final[Path] = STUDY_DIR / "capacity_table.parquet"
DRAWS_PATH: Final[Path] = STUDY_DIR / "capacity_draws.parquet"
TREND_PATH: Final[Path] = STUDY_DIR / "trend.parquet"
BIAS_PATH: Final[Path] = STUDY_DIR / "undetected_bias.parquet"
RAW_PATH: Final[Path] = STUDY_DIR / "raw_gaps.parquet"
REPORT_PATH: Final[Path] = STUDY_DIR / "report.md"

WORK_00040: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00040-pv-capacity-error-bars" / "work"
"""Experiment 00040's private work folder, read and never written."""
UNIT_VECTORS_PATH: Final[Path] = WORK_00040 / "unit_vectors.parquet"
REAL_VECTORS_PATH: Final[Path] = WORK_00040 / "real_vectors.parquet"
BOOTSTRAP_PATH: Final[Path] = WORK_00040 / "bootstrap.parquet"
REAL_SCORES_PATH: Final[Path] = WORK_00040 / "real_scores.parquet"
UNIT_SCORES_PATH: Final[Path] = WORK_00040 / "unit_scores.parquet"
WORK_00039: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00039-pv-disaggregation-cams-injection" / "work"
)
FRAME_PATH: Final[Path] = WORK_00039 / "frame.parquet"
SITES_PATH: Final[Path] = WORK_00039 / "sites.parquet"

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
"""The 8 Primaries whose meter reads a signed MW flow, so solar above demand does not fold."""
LEFT_OUT_PRIMARIES: Final[frozenset[str]] = frozenset({"D15"})
"""Wrangle T1 is left out: its misfit test detects no confound shape reliably."""
FLAGGED_PRIMARY_SHARE: Final[float] = 0.5
"""A Primary whose misfit flag fires in more than this share of its season-years is flagged."""
KIRTON: Final[str] = "D5"
DONINGTON: Final[str] = "D16"
FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F")

CLEAN_BOUND_AMPLITUDE: Final[float] = 0.25
"""A season-year is `clean` when the largest amplitude the test misses (flags in under 80% of
resamples) is at most this: the test then flags a day-long load of twice this size reliably."""
INTERVAL_QUANTILES: Final[tuple[float, float]] = (0.025, 0.975)
REPRODUCTION_TOLERANCE: Final[float] = 1e-9
BIAS_SHAPES: Final[tuple[str, ...]] = ("dayL", "sunload")
BIAS_AMPLITUDES: Final[tuple[float, ...]] = (0.1, 0.25, 0.5)
MIN_YEARS_FOR_TREND: Final[int] = 6
TREND_BOOTSTRAP_DRAWS: Final[int] = 2000
TREND_BOOTSTRAP_SEED: Final[int] = 1016
MIN_DAYLIGHT_ROWS_PER_DAY: Final[int] = 10
MAX_ZENITH_DEG: Final[float] = 75.0
CLEAR_INDEX: Final[float] = 0.8
OVERCAST_INDEX: Final[float] = 0.3
SUMMER_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)
WINTER_MONTHS: Final[tuple[int, ...]] = (11, 2)
"""Winter days that can be classified. At 53 degrees north the sun does not stay 15 degrees above
the horizon for 10 half-hours in December or January, so only November and February days count."""
MIDDAY_HALF_HOURS: Final[tuple[int, int]] = (22, 28)
"""Local half-hours 11:00 to 14:00, start inclusive and end exclusive."""

FIRST_SEASON_YEAR: Final[int] = 2019
LAST_SEASON_YEAR: Final[int] = 2025
"""The last complete season-year, September 2025 to August 2026."""

PAGE_PATH: Final[Path] = Path("docs/studies/embedded-pv-capacity.md")
CSV_PATH: Final[Path] = Path("docs/studies/assets/embedded_pv_capacity_table.csv")
TABLE_START_MARKER: Final[str] = "<!-- capacity-table-start -->"
TABLE_END_MARKER: Final[str] = "<!-- capacity-table-end -->"
