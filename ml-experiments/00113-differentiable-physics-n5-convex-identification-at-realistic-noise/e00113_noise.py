"""The noise of experiment 113: iid Gaussian, and real convex residuals from other substations.

**Real noise is built month by month.** Each target month takes a whole month of real training
residuals of the convex model's all-years D1 fit, from one of the other decision substations and a
year other than the target's, in the same calendar month, aligned by day-of-month offset and
half-hour, with a random sign per month. Slots the donor does not hold are zero.
"""

from dataclasses import dataclass
from typing import Final

import e00113_design as design_module
import numpy as np
from studies.dp_features import FeatureCache

MIN_DONOR_COVERAGE: Final = 0.25
"""A donor month is a candidate only if at least this share of its half-hours hold a residual."""
HALF_HOURS_PER_DAY: Final = 48


@dataclass(frozen=True)
class RealNoise:
    """One draw of real noise on a target's training rows.

    Attributes:
        values: The noise per training row, in points.
        zero_share: The share of rows whose donor slot was missing and so hold zero.
        donors: The donor substation of each of the target's months.
    """

    values: np.ndarray
    zero_share: float
    donors: list[str]


class ResidualPool:
    """The residual series of the donor substations, on the grid."""

    def __init__(self, *, cache: FeatureCache, residuals: dict[str, np.ndarray]) -> None:
        """Store the donors' grid-length residual arrays (NaN where there is none).

        Args:
            cache: The feature cache, for the grid.
            residuals: Each donor's residual on every grid row.
        """
        self.end_times = cache.end_times
        self.residuals = residuals
        self._coverage: dict[tuple[str, int], float] = {}

    def _month_coverage(self, *, donor: str, month: np.datetime64) -> float:
        key = (donor, int(month.astype(np.int64)))
        if key not in self._coverage:
            start = month.astype("datetime64[us]")
            stop = (month + np.timedelta64(1, "M")).astype("datetime64[us]")
            lo = np.searchsorted(self.end_times, start, side="right")
            hi = np.searchsorted(self.end_times, stop, side="right")
            days = (stop - start) / np.timedelta64(1, "D")
            held = np.isfinite(self.residuals[donor][lo:hi]).sum()
            self._coverage[key] = float(held / (days * HALF_HOURS_PER_DAY))
        return self._coverage[key]

    def options(self, *, target: str, month: np.datetime64) -> list[tuple[str, np.datetime64]]:
        """Return the (donor, month) pairs a target month may take its noise from."""
        found = []
        for donor in self.residuals:
            if donor == target:
                continue
            for years in range(-7, 8):
                if years == 0:
                    continue
                candidate = month + np.timedelta64(12 * years, "M")
                if self._month_coverage(donor=donor, month=candidate) >= MIN_DONOR_COVERAGE:
                    found.append((donor, candidate))
        return found

    def draw(self, *, design: design_module.Design, rng: np.random.Generator) -> RealNoise:
        """Draw one real-noise series on the design's training rows."""
        end = self.end_times[design.index]
        values = np.zeros(design.n)
        missing = 0
        donors = []
        for position, month in enumerate(design.months):
            options = self.options(target=design.site, month=month)
            donor, donor_month = options[rng.integers(len(options))]
            sign = 1.0 if rng.random() < 0.5 else -1.0
            donors.append(donor)
            rows = np.flatnonzero(design.month_of_row == position)
            offset = end[rows] - month.astype("datetime64[us]")
            wanted = donor_month.astype("datetime64[us]") + offset
            slot = np.clip(np.searchsorted(self.end_times, wanted), 0, len(self.end_times) - 1)
            exact = (self.end_times[slot] == wanted) & (
                self.end_times[slot].astype("datetime64[M]") == donor_month
            )
            donor_values = np.where(exact, self.residuals[donor][slot], np.nan)
            absent = ~np.isfinite(donor_values)
            missing += int(absent.sum())
            values[rows] = sign * np.where(absent, 0.0, donor_values)
        return RealNoise(values=values, zero_share=missing / design.n, donors=donors)


def iid_noise(*, n: int, sd: float, rng: np.random.Generator) -> np.ndarray:
    """Return independent Gaussian noise."""
    return rng.normal(0.0, sd, size=n)


def autocorrelation(*, design: design_module.Design, noise: np.ndarray, lag: int) -> float:
    """Return the autocorrelation of `noise` at `lag` half-hours, over pairs of training rows."""
    low = design.index[0]
    size = design.index[-1] - low + 1
    series = np.zeros(size)
    mask = np.zeros(size)
    series[design.index - low] = noise
    mask[design.index - low] = 1.0
    pairs = float((mask[:-lag] * mask[lag:]).sum())
    numerator = float((series[:-lag] * series[lag:] * mask[:-lag] * mask[lag:]).sum()) / pairs
    return numerator / float((noise**2).mean())
