"""Real noise for experiment 126's synthetic substations: whole months of donor residuals.

**Each target month takes a whole month of real training residuals of B0 from one of the other
decision substations and a year other than the target's, in the same calendar month**, aligned by
offset from the start of the month, with a random sign per month. Slots the donor does not hold are
zero. The donors' residuals are B0's in-sample training residuals at the 2024-07-01 cut, so no
residual from the replicate or scored year enters.
"""

from dataclasses import dataclass
from typing import Final

import e00126_common as common
import e00126_design as design
import numpy as np
from studies.dp_features import FeatureCache
from studies.dp_pinned_capacity import read_vintages

MIN_DONOR_COVERAGE: Final[float] = 0.25
"""A donor month is a candidate only if at least this share of its half-hours hold a residual."""
HALF_HOURS_PER_DAY: Final[int] = 48
MAX_YEAR_OFFSET: Final[int] = 7


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
            for years in range(-MAX_YEAR_OFFSET, MAX_YEAR_OFFSET + 1):
                if years == 0:
                    continue
                candidate = month + np.timedelta64(12 * years, "M")
                if self._month_coverage(donor=donor, month=candidate) >= MIN_DONOR_COVERAGE:
                    found.append((donor, candidate))
        return found

    def draw(self, *, site: str, index: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw one real-noise series on the grid rows `index`.

        Args:
            site: The target substation, which never donates.
            index: The grid rows the noise covers.
            rng: The random generator.

        Returns:
            The noise per row, in points.
        """
        end = self.end_times[index]
        months, month_of_row = np.unique(end.astype("datetime64[M]"), return_inverse=True)
        values = np.zeros(len(index))
        for position, month in enumerate(months):
            options = self.options(target=site, month=month)
            donor, donor_month = options[rng.integers(len(options))]
            sign = 1.0 if rng.random() < 0.5 else -1.0
            rows = np.flatnonzero(month_of_row == position)
            offset = end[rows] - month.astype("datetime64[us]")
            wanted = donor_month.astype("datetime64[us]") + offset
            slot = np.clip(np.searchsorted(self.end_times, wanted), 0, len(self.end_times) - 1)
            exact = (self.end_times[slot] == wanted) & (
                self.end_times[slot].astype("datetime64[M]") == donor_month
            )
            donor_values = np.where(exact, self.residuals[donor][slot], np.nan)
            values[rows] = sign * np.where(np.isfinite(donor_values), donor_values, 0.0)
        return values


def build_pool(*, cache: FeatureCache) -> ResidualPool:
    """Build the donors' pool: B0's training residuals at the replicate cut at each decision site.

    Args:
        cache: The feature cache.

    Returns:
        The pool.
    """
    vintages = read_vintages()
    residuals: dict[str, np.ndarray] = {}
    for site in common.DECISION_SITES:
        base = design.build_base(
            cache=cache, vintages=vintages, site=site, cut=common.REPLICATE_CUT
        )
        context = design.ArmContext(aux=design.site_aux(cache=cache, site=site))
        fit = design.fit_arm(cache=cache, base=base, arm=design.B0, context=context)
        stream = np.full(len(cache.end_times), np.nan)
        stream[base.index] = base.target - base.x0 @ fit.theta
        residuals[site] = stream
    return ResidualPool(cache=cache, residuals=residuals)


@dataclass(frozen=True)
class NoiseSummary:
    """A noise draw's size, for the report.

    Attributes:
        standard_deviation: The standard deviation of the noise, in points.
        zero_share: The share of rows whose donor slot was missing.
    """

    standard_deviation: float
    zero_share: float
