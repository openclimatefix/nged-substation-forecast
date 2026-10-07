"""Shared synthetic power values, for any test suite in the repo.

Importable by bare name via the ``pythonpath = ["tests"]`` pytest setting.
"""

from datetime import UTC, datetime
from typing import Final

_EPOCH: Final[datetime] = datetime(2020, 1, 1, tzinfo=UTC)


def power_at(time: datetime, offset: int = 0) -> float:
    """Integer power equal to the half-hour index since a fixed epoch, mod 1999, shifted by -999.

    The modulus 1999 is prime, and the manual heuristic's lags are multiples of 336 half-hours,
    so the 13 lag values at one target time are distinct. A lag off by one half-hour lands on yet
    another value. ``offset`` is added to the half-hour index, so that different series carry
    different values.
    """
    half_hour_index = int((time - _EPOCH).total_seconds() // 1800) + offset
    return float(half_hour_index % 1999 - 999)
