"""The pinned PV capacity of the differentiable-physics (DP) series: a real-time input, not a fit.

**A capacity that moves with the fit's starting point adds noise to every history contrast**, and
the pooled model's capacity is not identified by the data (different starts reach different
capacities at nearly the same fit). Pinning takes the capacity from the real-time `frozen` vintages
of experiment 45 instead, so the model fits everything else around a fixed solar term.

**The rule for one fit with cut `c`.**

- **Which vintage.** The latest vintage whose origin Monday is on or before `c`. It lists a weekly
  capacity back to the substation's first week.
- **A cut before the first vintage.** The first origin is 2023-08-07 (2025-04-14 at D9, 2026-08-03
  at D18). When the earliest vintage's origin is at most `EARLY_VINTAGE_MAX_GAP_DAYS` after `c`,
  that vintage is used as it is, and the write-up discloses the leak of those weeks. When the gap is
  longer, the vintage's weekly shape is hindsight too far past the cut to use, so only its level at
  the cut is kept (the median of its last `HOLD_WEEKS` weeks up to `c`) and earlier weeks are the
  level scaled back at `CAPACITY_GROWTH_PER_YEAR` per year.
- **Unobserved weeks** hold the last observed value, and leading unobserved weeks take the first
  observed value. A zero is never read as capacity.
- **After the cut,** and for any week beyond the last week of the vintage (the vintage of D14 stops
  at 2023-04-17), the capacity is the median of the last `HOLD_WEEKS` weeks up to `c`, not the last
  week alone: the latest week of a real-time capacity has 3.63 times the path error of the same
  week in the final fit.
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Final, Literal

import numpy as np
import polars as pl

from studies.dp_features import FeatureCache, SiteTable
from studies.sources import ML_EXPERIMENTS_DATA_DIR

VINTAGES_PATH: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00045-pvdis-4-weekly-capacity-asymmetric-l1"
    / "capacity_weekly_vintages.parquet"
)
"""Experiment 45's real-time capacity vintages, one block of weekly capacities per origin Monday."""

VINTAGE_VARIANT: Final[str] = "frozen"
"""The only variant that uses past data alone; the `recommended` constant is hindsight."""

CAPACITY_GROWTH_PER_YEAR: Final[float] = 0.05
"""The annual growth of capacity assumed before a vintage that is too late to use.

It is the median of the 12 season-year increments of capacity at the five long-history Primaries
(D5, D10, D11, D14, and D16) in the earliest vintage, which is 5.4%, rounded down to 5%. The
increments run from 0% (D5, whose capacity is flat) to 219% (D11's first season-year).
"""

HOLD_WEEKS: Final[int] = 13
"""The capacity held after the cut is the median of this many weeks up to the cut."""

EARLY_VINTAGE_MAX_GAP_DAYS: Final[int] = 42
"""A vintage whose origin is at most this long after the cut is used as it is (6 weeks)."""

DAYS_PER_YEAR: Final[float] = 365.25

CapacitySourceType = Literal["vintage", "early_vintage", "growth_prior"]
"""Where the pinned capacity came from: a vintage at or before the cut, a vintage up to 6 weeks
after the cut, or the level at the cut scaled back at the growth prior."""


@dataclass(frozen=True)
class PinnedCapacity:
    """One substation's pinned weekly capacity for a fit with a given cut.

    Attributes:
        site: The substation's label.
        cut: The fit's cut.
        source: Where the capacity came from.
        origin: The origin Monday of the vintage used.
        weeks: Week-start Mondays as `datetime64[D]`, ascending, one per weekly value.
        points: The capacity of each week in points of effective capacity, with unobserved weeks
            filled.
        held_points: The capacity held after the cut and beyond the last week.
        n_unobserved: How many weeks of the vintage were unobserved before filling.
    """

    site: str
    cut: datetime
    source: CapacitySourceType
    origin: date
    weeks: np.ndarray
    points: np.ndarray
    held_points: float
    n_unobserved: int

    def at_rows(self, *, cache: FeatureCache, index: np.ndarray) -> np.ndarray:
        """Return the capacity in points at each grid row.

        Args:
            cache: The cache, for the rows' times.
            index: Grid rows.

        Returns:
            The weekly capacity for a row inside the weeks of the vintage that ends no later than
            the cut, `held_points` for a row after the cut or after the last week, and the first
            week's capacity for a row before the first week.
        """
        ends = cache.end_times[index]
        day = (ends - np.timedelta64(30, "m")).astype("datetime64[D]")
        monday = day - ((day.astype(np.int64) + 3) % 7).astype("timedelta64[D]")
        position = np.clip(np.searchsorted(self.weeks, monday, side="right") - 1, 0, None)
        capacity = self.points[position]
        cut_day = np.datetime64(self.cut.date().isoformat(), "D")
        beyond = (monday > self.weeks[-1]) | (monday > cut_day)
        return np.where(beyond, self.held_points, capacity)


def _fill_unobserved(*, capacity: np.ndarray, unobserved: np.ndarray) -> np.ndarray:
    """Hold the last observed value through unobserved weeks; leading ones take the first observed.

    Raises:
        ValueError: If no week is observed.
    """
    observed = np.flatnonzero(~unobserved & np.isfinite(capacity))
    if len(observed) == 0:
        msg = "a capacity vintage has no observed week"
        raise ValueError(msg)
    position = np.searchsorted(observed, np.arange(len(capacity)), side="right") - 1
    return capacity[observed[np.clip(position, 0, None)]]


def pinned_capacity(*, vintages: pl.DataFrame, table: SiteTable, cut: datetime) -> PinnedCapacity:
    """Build a substation's pinned capacity for a fit with cut `cut`.

    Args:
        vintages: Experiment 45's `capacity_weekly_vintages.parquet`, with columns
            `time_series_id`, `variant`, `origin_monday`, `week_start`, `capacity_mw`, and
            `unobserved`.
        table: The substation, for its series identifier and effective capacity.
        cut: The fit's cut; no vintage newer than `EARLY_VINTAGE_MAX_GAP_DAYS` after it is used.

    Returns:
        The pinned capacity.

    Raises:
        ValueError: If the substation has no vintage, or its vintage has no week on or before
            the cut.
    """
    site_rows = vintages.filter(
        pl.col("variant") == VINTAGE_VARIANT, pl.col("time_series_id") == table.time_series_id
    )
    if site_rows.is_empty():
        msg = f"{table.site} has no `{VINTAGE_VARIANT}` capacity vintage"
        raise ValueError(msg)
    cut_date = cut.date()
    before = site_rows.filter(pl.col("origin_monday") <= cut_date)
    source: CapacitySourceType = "vintage"
    if before.is_empty():
        origin = site_rows["origin_monday"].min()
        source = "early_vintage"
    else:
        origin = before["origin_monday"].max()
    if not isinstance(origin, date):
        msg = "a vintage origin is not a date"
        raise TypeError(msg)
    if source == "early_vintage" and (origin - cut_date).days > EARLY_VINTAGE_MAX_GAP_DAYS:
        source = "growth_prior"
    weekly = site_rows.filter(pl.col("origin_monday") == origin)
    weekly = weekly.sort("week_start")
    weeks = weekly["week_start"].to_numpy().astype("datetime64[D]")
    unobserved = weekly["unobserved"].to_numpy()
    points = _fill_unobserved(
        capacity=100.0
        * weekly["capacity_mw"].to_numpy().astype(np.float64)
        / table.effective_capacity_mw,
        unobserved=unobserved,
    )
    cut_day = np.datetime64(cut_date.isoformat(), "D")
    up_to_cut = np.flatnonzero(weeks <= cut_day)
    if len(up_to_cut) == 0:
        msg = f"{table.site}'s capacity vintage has no week on or before {cut_date}"
        raise ValueError(msg)
    held = float(np.median(points[up_to_cut][-HOLD_WEEKS:]))
    if source == "growth_prior":
        years_before = (cut_day - weeks).astype(np.float64) / DAYS_PER_YEAR
        points = np.where(
            weeks <= cut_day, held / (1.0 + CAPACITY_GROWTH_PER_YEAR) ** years_before, held
        )
    return PinnedCapacity(
        site=table.site,
        cut=cut,
        source=source,
        origin=origin,
        weeks=weeks,
        points=points,
        held_points=held,
        n_unobserved=int(unobserved.sum()),
    )


def read_vintages() -> pl.DataFrame:
    """Read experiment 45's capacity vintages, the `frozen` variant only."""
    return pl.read_parquet(VINTAGES_PATH).filter(pl.col("variant") == VINTAGE_VARIANT)
