"""Paths, constants, and week helpers of experiment 00045 (PVDIS-4).

The experiment fits a weekly embedded-solar capacity series for 8 Primary substations under an
asymmetric L1 penalty (`docs/ml-experiments/00045-pvdis-4-weekly-capacity-asymmetric-l1/
result.md`). Every script runs on at most four cores:

    POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run python ...

Experiment 00039's `frame.parquet` and experiment 00037's logged-closure table are read from their
private folders and never written.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import polars as pl
from studies.anonymise import SITE_LABELS
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 45
SLUG: Final[str] = "pvdis-4-weekly-capacity-asymmetric-l1"
FOLDER: Final[str] = f"{EXPERIMENT_ID:05d}-{SLUG}"

DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / FOLDER
"""Private outputs of this experiment; never committed."""
DOCS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / FOLDER
"""The rendered page's folder, which holds the figures."""

FRAME_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00039-pv-disaggregation-cams-injection" / "work" / "frame.parquet"
)
FRAME_ROWS: Final[int] = 2_891_184
"""The row count of experiment 00039's frame, which the validation asserts."""
LOGGED_CLOSURES_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00037-switching-first-look" / "intervals.parquet"
)
"""Experiment 00037's logged closure intervals (start_utc, end_utc, and the two Primary series)."""
INPUTS_PATH: Final[Path] = DATA_DIR / "inputs_primary_weeks.parquet"
HALF_HOURLY_PATH: Final[Path] = DATA_DIR / "inputs_half_hourly.parquet"
FARM_WEEKS_PATH: Final[Path] = DATA_DIR / "farm_weeks.parquet"
FARM_HALF_HOURLY_PATH: Final[Path] = DATA_DIR / "farm_half_hourly.parquet"
VALIDATION_REPORT_PATH: Final[Path] = DATA_DIR / "validation_report.txt"
WEEKLY_FREE_PATH: Final[Path] = DATA_DIR / "raw_free_weekly_capacity.parquet"

PRIMARIES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D15", "D16", "D18")
"""Frame site codes of the 8 Primaries metered in MW, in the order of `PRIMARY_NAMES`."""
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
"""Substation names; substations may be named on a published chart."""
PRIMARY_SERIES_IDS: Final[dict[str, int]] = {
    "D5": 17,
    "D9": 7,
    "D10": 10,
    "D11": 8,
    "D14": 16,
    "D15": 12,
    "D16": 18,
    "D18": 13,
}
"""Each frame site code's `time_series_id` in `TimeSeriesMetadata`, which the closure log uses."""
PRIMARY_STATUS: Final[dict[str, str]] = {
    "D5": "range only",
    "D9": "low power",
    "D10": "clean",
    "D11": "clean",
    "D14": "clean",
    "D15": "not clean",
    "D16": "clean",
    "D18": "clean",
}
"""Kirton is flagged range-only, Wrangle T1 not-clean, and Trusthorpe low-power."""
FARMS: Final[tuple[str, ...]] = SITE_LABELS
FARM_NAMES: Final[dict[str, str]] = {farm: f"Solar farm {farm}" for farm in FARMS}

FIRST_WEEK_START: Final[datetime] = datetime(2019, 9, 2, tzinfo=UTC)
"""Monday of week 0."""
N_WEEKS: Final[int] = 361
LAST_DATA_END: Final[datetime] = datetime(2026, 8, 1, tzinfo=UTC)
"""Midnight ending the last week's 5 days (Monday 2026-07-27 to Friday 2026-07-31)."""
HALF_HOURS_PER_WEEK: Final[int] = 336

MAX_ZENITH_DEG: Final[float] = 75.0
MIN_USABLE_HALF_HOURS_PER_DAY: Final[int] = 10
"""A day with fewer usable daylight half-hours than this is not a usable day."""
MIN_USABLE_DAYS_PER_WEEK: Final[int] = 3
"""A week with fewer usable days is `unobserved`."""
MASK_PAD_HOURS: Final[int] = 24
MAX_FARM_MISSING_DAYS: Final[int] = 2
"""A farm-week with output missing on more days than this is dropped from injection units."""
DROPOUT_GHI_W_M2: Final[float] = 100.0
"""A farm half-hour at or below zero power while CAMS shows more than this is a dropout."""
SUMMER_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)
CLEAR_INDEX: Final[float] = 0.8
OVERCAST_INDEX: Final[float] = 0.3


def week_start(*, week: int) -> datetime:
    """Return the UTC Monday that starts `week`."""
    return FIRST_WEEK_START + timedelta(weeks=week)


def week_expr(*, time: pl.Expr) -> pl.Expr:
    """Return the week index of a period-ending UTC stamp (its day is that of the period's middle).

    A stamp marks the end of its half-hour, so 00:00 on a Monday belongs to the previous Sunday.
    """
    return (
        (time.dt.offset_by("-15m").dt.date().cast(pl.Datetime("us", "UTC")) - FIRST_WEEK_START)
        .dt.total_days()
        .floordiv(7)
        .cast(pl.Int32)
    )


def day_expr(*, time: pl.Expr) -> pl.Expr:
    """Return the UTC calendar date of the half-hour that a period-ending stamp closes."""
    return time.dt.offset_by("-15m").dt.date()
