"""The as-of, hindsight, no-PV, and pin capacities of experiment 139 as `PinnedCapacity` objects.

`fit_gam` takes a `PinnedCapacity` whose `cut` equals the fit's cut, so every arm builds one object
per Primary and monthly cut. The weekly path holds a capacity per week; rows after the cut read
`held_points`.

**As-of arm.** At a cut `c`, the origin is the latest of the as-of origins (1 July of 2023, 2024,
and 2025) at or before `c`. The path of a week in season-year `s` is the table's estimate for `s`
when `s` is a data year of the origin (at least 12,000 training half-hours, experiment 129's rule)
and `s` is no later than the origin's latest data year. A later season-year takes the latest data
year's estimate. An earlier season-year that is not a data year takes the pin's value at that
week. `held_points` is the latest data year's estimate.

**Hindsight arm.** The same path from experiment 129's published table, fitted on data to
2026-07-31 (a ceiling, never a result).
"""

from dataclasses import dataclass
from datetime import datetime

import e00139_common as common
import numpy as np
import polars as pl
from studies.dp_pinned_capacity import PinnedCapacity
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WEEKS_START = np.datetime64("2019-06-24", "D")
WEEKS_END = np.datetime64("2027-01-04", "D")
WEEKS: np.ndarray = np.arange(WEEKS_START, WEEKS_END, np.timedelta64(7, "D"))
"""Week-start Mondays that every path covers."""


# A Monday up to six days before 1 July labels its July days with the previous season-year, which
# affects at most one week per season-year.
def season_year_of_weeks(*, weeks: np.ndarray) -> np.ndarray:
    """Return the season-year (July to June, labelled by the July's year) of each Monday."""
    months = weeks.astype("datetime64[M]")
    year = months.astype("datetime64[Y]").astype(int) + 1970
    month = months.astype(int) % 12 + 1
    return np.where(month >= 7, year, year - 1)


def pin_on_weeks(*, pin: PinnedCapacity) -> np.ndarray:
    """Return the pin's capacity in each of `WEEKS`, with the pin's own after-cut rule."""
    position = np.clip(np.searchsorted(pin.weeks, WEEKS, side="right") - 1, 0, None)
    cut_day = np.datetime64(pin.cut.date().isoformat(), "D")
    beyond = (pin.weeks[-1] < WEEKS) | (cut_day < WEEKS)
    return np.where(beyond, pin.held_points, pin.points[position])


def with_path(
    *, pin: PinnedCapacity, points: np.ndarray, held: float, source: str
) -> PinnedCapacity:
    """Return a `PinnedCapacity` with the pin's site and cut and a new path."""
    return PinnedCapacity(
        site=pin.site,
        cut=pin.cut,
        source=pin.source if source == "pin" else "vintage",
        origin=pin.origin,
        weeks=WEEKS,
        points=points,
        held_points=held,
        n_unobserved=0,
    )


def pin_adaptor(*, pin: PinnedCapacity) -> PinnedCapacity:
    """Return the pin's own weekly path in the adaptor's form (the wiring identity check)."""
    return with_path(pin=pin, points=pin_on_weeks(pin=pin), held=pin.held_points, source="pin")


def no_pv_adaptor(*, pin: PinnedCapacity) -> PinnedCapacity:
    """Return a capacity of zero at every week."""
    return with_path(pin=pin, points=np.zeros(len(WEEKS)), held=0.0, source="zero")


def origin_for_cut(*, cut: datetime) -> datetime | None:
    """Return the latest as-of origin at or before `cut`, or `None`."""
    earlier = [origin for origin in common.ORIGINS if origin <= cut]
    return max(earlier) if earlier else None


def table_adaptor(
    *, pin: PinnedCapacity, estimates: dict[int, float], data_years: set[int]
) -> tuple[PinnedCapacity, int]:
    """Return a capacity built from per-season-year estimates, and a fallback week count.

    Args:
        pin: The pin at this cut, for the fallback and the site and cut.
        estimates: The estimate in points of effective capacity per season-year.
        data_years: The season-years whose estimate counts (enough training rows).

    Returns:
        The capacity, and how many weeks in the path took the pin's value.

    Raises:
        ValueError: If no season-year is a data year.
    """
    usable = sorted(year for year in data_years if np.isfinite(estimates.get(year, np.nan)))
    if not usable:
        msg = f"{pin.site} has no data year at {pin.cut:%Y-%m-%d}"
        raise ValueError(msg)
    latest = usable[-1]
    years = season_year_of_weeks(weeks=WEEKS)
    fallback = pin_on_weeks(pin=pin)
    points = np.empty(len(WEEKS))
    took_pin = np.zeros(len(WEEKS), dtype=bool)
    for k, year in enumerate(years):
        if year >= latest:
            points[k] = estimates[latest]
        elif year in usable:
            points[k] = estimates[int(year)]
        else:
            points[k] = fallback[k]
            took_pin[k] = True
    cut_day = np.datetime64(pin.cut.date().isoformat(), "D")
    in_window = (cut_day >= WEEKS) & (cut_day - np.timedelta64(731, "D") < WEEKS)
    return with_path(pin=pin, points=points, held=estimates[latest], source="table"), int(
        (took_pin & in_window).sum()
    )


def asof_estimates(
    *, table: pl.DataFrame, site: str, origin: datetime, column: str = "estimate_points"
) -> tuple[dict, set[int]]:
    """Return the as-of estimates per season-year and the origin's data years.

    `column` is `estimate_points` (the raw capacity divided by the recovery divisor) or
    `raw_points` (the capacity in the convex model's own units).
    """
    block = table.filter(pl.col("site") == site, pl.col("origin") == origin)
    estimates = dict(zip(block["season_year"], block[column], strict=True))
    data = {
        int(y)
        for y, n in zip(block["season_year"], block["training_rows"], strict=True)
        if n >= common.MIN_YEAR_ROWS
    }
    return {int(k): float(v) for k, v in estimates.items()}, data


@dataclass(frozen=True)
class HindsightTable:
    """Experiment 129's published estimates (data to 2026-07-31), in points per season-year."""

    estimates: dict[str, dict[int, float]]

    @classmethod
    def read(cls) -> HindsightTable:
        """Read `real_arm2_to_july.parquet` from experiment 129's private work folder."""
        path = (
            ML_EXPERIMENTS_DATA_DIR
            / "00129-differentiable-physics-pv-capacity-known-answer-test"
            / "work"
            / "real_arm2_to_july.parquet"
        )
        frame = pl.read_parquet(path).filter(pl.col("arm") == "arm2")
        out: dict[str, dict[int, float]] = {}
        for site, year, estimate in frame.select(
            "site", "season_year", "estimate_points"
        ).iter_rows():
            out.setdefault(site, {})[int(year)] = float(estimate)
        return cls(estimates=out)
