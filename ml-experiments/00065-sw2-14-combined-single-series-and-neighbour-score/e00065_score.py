"""Match detector intervals to known events, and combine recall with precision into event F1.

An event is found when a detector interval on one of the event's series has an onset within
`ONSET_TOLERANCE` stamps of the true onset and an end within the larger of `END_MINIMUM` stamps and
`END_FRACTION` of the duration. The interval's end is the end estimate once it is known (at its
`known_from` stamp), so `known_delay` records how late that is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from e00065_intervals import Interval

ONSET_TOLERANCE: Final[int] = 4
"""Two hours."""
END_MINIMUM: Final[int] = 12
"""Six hours."""
END_FRACTION: Final[float] = 0.10


@dataclass(frozen=True)
class Match:
    """How one known event was found."""

    onset_found: bool
    found: bool
    end_error: float
    """Smallest absolute end error (stamps) among intervals with a matching onset, else NaN."""
    known_delay: float
    """Stamps from the true end to the matched interval's `known_from`, else NaN."""


def end_tolerance(*, onset: int, end: int) -> float:
    """The allowed end error in stamps: the larger of 6 hours and 10% of the duration."""
    return max(float(END_MINIMUM), END_FRACTION * (end - onset))


def by_series(*, intervals: list[Interval]) -> dict[int, list[Interval]]:
    """Group intervals by the series they mask."""
    out: dict[int, list[Interval]] = {}
    for interval in intervals:
        out.setdefault(interval.series, []).append(interval)
    return out


def match_event(
    *, series: tuple[int, ...], onset: int, end: int, intervals: dict[int, list[Interval]]
) -> Match:
    """Find the best interval on `series` for an event from `onset` to `end`."""
    tolerance = end_tolerance(onset=onset, end=end)
    onset_found, found = False, False
    errors: list[float] = []
    delays: list[float] = []
    for s in series:
        for interval in intervals.get(s, []):
            if abs(interval.onset - onset) > ONSET_TOLERANCE:
                continue
            onset_found = True
            error = abs(interval.end - end)
            errors.append(error)
            if error <= tolerance:
                found = True
                delays.append(interval.known_from - end)
    return Match(
        onset_found=onset_found,
        found=found,
        end_error=min(errors) if errors else float("nan"),
        known_delay=min(delays) if delays else float("nan"),
    )


def event_f1(*, recall: float, precision: float) -> float:
    """The harmonic mean of recall and precision."""
    return 0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)


def month_bootstrap(
    *, months: np.ndarray, values: np.ndarray, n_resamples: int, seed: int
) -> np.ndarray:
    """Resample whole calendar months of per-event 0/1 values; return the resampled means."""
    rng = np.random.default_rng(seed)
    labels = np.unique(months)
    sums = np.array([values[months == m].sum() for m in labels], dtype=float)
    counts = np.array([(months == m).sum() for m in labels], dtype=float)
    picks = rng.integers(0, len(labels), size=(n_resamples, len(labels)))
    return sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
