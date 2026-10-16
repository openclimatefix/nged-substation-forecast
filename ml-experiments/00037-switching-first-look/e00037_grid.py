"""The half-hour grid of experiment 00037: stamp 0 is 2022-01-01 00:00 UTC (period ending)."""

from datetime import UTC, datetime, timedelta
from typing import Final

GRID_START: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
GRID_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
GRID_LENGTH: Final[int] = int((GRID_END - GRID_START).total_seconds() // 1800)
"""Number of half-hour stamps from `GRID_START` to `GRID_END`."""
LOG_COVERAGE_END: Final[datetime] = datetime(2026, 3, 26, tzinfo=UTC)
"""The switching logs run to 2026-03-25; later months are reported apart."""


def stamp_index(*, utc_time: datetime) -> int:
    """Return the index of the half-hour containing `utc_time` (floor), from `GRID_START`."""
    return int((utc_time - GRID_START).total_seconds() // 1800)


COVERAGE_END_STAMP: Final[int] = stamp_index(utc_time=LOG_COVERAGE_END)


def stamp_months() -> list[int]:
    """Return the UTC calendar month (1 to 12) of each grid stamp."""
    return [(GRID_START + timedelta(minutes=30 * i)).month for i in range(GRID_LENGTH)]
