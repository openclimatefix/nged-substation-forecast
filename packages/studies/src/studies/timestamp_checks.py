"""Measure which instant an irradiance series' timestamps describe, from the sun alone.

**A series of global irradiance tracks the cosine of the solar zenith angle at the instant it
describes, whatever the clouds do.** Clouds scale each hour's value up and down, but they do so
equally in the morning and the afternoon, so over a year of hours the correlation between the
series and the cosine peaks at the offset where the two line up. A mean over the hour ending at
the label peaks near 30 minutes before the label, a snapshot near its own stamp, and a mean over the
hour beginning at the label near 30 minutes after it.

The check needs no second product, so it cannot be satisfied by two products carrying the same
timing error.
"""

from collections.abc import Sequence
from typing import Final

import numpy as np
import polars as pl

from studies.solar import cos_zenith, zenith

CANDIDATE_OFFSETS_MINUTES: Final[tuple[int, ...]] = tuple(range(-60, 65, 5))
"""The offsets a series' stamps are tested at, in minutes, from an hour early to an hour late."""

HOUR_ENDING_OFFSET_MINUTES: Final[int] = -30
"""Where the correlation should peak for a mean over the hour ending at the label."""

MAX_OFFSET_ERROR_MINUTES: Final[int] = 15
"""How far the peak may sit from `HOUR_ENDING_OFFSET_MINUTES` before the series fails.

The errors the check exists to catch are an hour-beginning label, 60 minutes from the right answer,
and an instantaneous value read as an hourly mean, 30 minutes from it. A 15-minute tolerance
separates both from the right answer while allowing for a satellite that scans the trial area some
minutes after its nominal slot time.
"""


def correlation_by_offset(
    *,
    times: pl.Series,
    ghi: np.ndarray,
    latitude: float,
    longitude: float,
    offsets_minutes: Sequence[int] = CANDIDATE_OFFSETS_MINUTES,
) -> dict[int, float]:
    """Return the correlation between a series and the sun's height at each offset from its stamps.

    Args:
        times: The series' UTC timestamps.
        ghi: Global horizontal irradiance at each stamp, in W m⁻².
        latitude: Where the series applies, in degrees north.
        longitude: Where the series applies, in degrees east.
        offsets_minutes: The offsets to test. A negative offset compares each value with the sun
            that many minutes before its stamp.

    Returns:
        The Pearson correlation at each offset.

    Raises:
        ValueError: If `times` and `ghi` differ in length, or fewer than two hours are given.
    """
    if times.len() != len(ghi):
        msg = f"{times.len()} stamps against {len(ghi)} values"
        raise ValueError(msg)
    if len(ghi) < 2:
        msg = "a correlation needs at least two hours"
        raise ValueError(msg)
    values = np.asarray(ghi, dtype=np.float64)
    return {
        offset: float(
            np.corrcoef(
                values,
                cos_zenith(
                    zenith_deg=zenith(
                        stamps=times.dt.offset_by(f"{offset}m"),
                        latitude=latitude,
                        longitude=longitude,
                    )
                ),
            )[0, 1]
        )
        for offset in offsets_minutes
    }


def best_offset_minutes(*, correlations: dict[int, float]) -> int:
    """Return the offset at which the correlation peaks.

    Args:
        correlations: The output of `correlation_by_offset`.

    Returns:
        The offset, in minutes, with the highest correlation.
    """
    return max(correlations, key=lambda offset: correlations[offset])


def check_hour_ending(*, correlations: dict[int, float], name: str) -> None:
    """Assert a series' correlation peaks where a mean over the hour ending at its label would.

    Args:
        correlations: The output of `correlation_by_offset`.
        name: The series, for the error message, such as `SARAH-3 at site A`.

    Raises:
        ValueError: If the peak sits more than `MAX_OFFSET_ERROR_MINUTES` from
            `HOUR_ENDING_OFFSET_MINUTES`.
    """
    best = best_offset_minutes(correlations=correlations)
    if abs(best - HOUR_ENDING_OFFSET_MINUTES) > MAX_OFFSET_ERROR_MINUTES:
        msg = (
            f"{name} tracks the sun best {best:+d} minutes from its stamps, where a mean over the "
            f"hour ending at the stamp would peak near {HOUR_ENDING_OFFSET_MINUTES:+d}; settle "
            "which window the stamps name before training on it"
        )
        raise ValueError(msg)
