"""Paths and constants shared by the scripts of experiment 186."""

from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "switching-d8-gas-peaker-spark-spread-and-triads"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00186-{SLUG}"
PUBLIC_DIR: Final[Path] = DATA_DIR / "public"
WORK_DIR: Final[Path] = DATA_DIR / "work"
E185_PUBLIC: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00185-dp-price-dfs-national-signals-past-demand" / "public"
)
E177_PUBLIC: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00177-dp-worst-days-other-data-sources" / "public"
)
E159_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00159-switching-d8-multi-day-drops-recurrence-and-report"
)
ARRAYS_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "briefs" / "sw_false_alarms" / "work" / "arrays.pkl"
)
D8_SERIES: Final[int] = 26
SERIES_LABELS: Final[dict[int, str]] = {26: "D8", 3: "D3", 16: "D14", 17: "D5"}
GRID_START_UTC: Final[str] = "2022-01-01T00:00:00+00:00"
LOCAL_TZ: Final[str] = "Europe/London"
