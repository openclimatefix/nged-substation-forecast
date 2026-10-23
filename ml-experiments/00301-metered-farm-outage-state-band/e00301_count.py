"""Count outage half-hours per farm in experiment 287's scored rows (experiment 301's stop rule).

An outage half-hour is a daylight half-hour (sun more than 5 degrees above the horizon) in which the
farm's output is under 5% of its plant forecast while that forecast exceeds 0.15 of capacity, on a
day with at least 6 such half-hours. Farms appear only as labels A to F.
"""

import sys
from pathlib import Path
from typing import Final

import polars as pl

sys.path.insert(
    0, str(Path(__file__).parents[1] / "00287-metered-farm-spread-binned-plant-heads")
)
from e00287_common import FARMS, load_farm  # noqa: E402

MIN_LOW_PER_DAY: Final[int] = 6
STOP_COUNT: Final[int] = 200
STOP_FARMS: Final[int] = 4


def main() -> None:
    """Print outage half-hours per farm and whether the stop rule fires."""
    n_below = 0
    for site in FARMS:
        farm = load_farm(site)
        table = (
            farm.frame.with_columns(scored=pl.Series(farm.scored))
            .filter(pl.col("scored"))
            .with_columns(forecast=pl.col("unit_mean") * farm.own_scale)
            .with_columns(
                low=(pl.col("output") < 0.05 * pl.col("forecast"))
                & (pl.col("forecast") > 0.15)
                & (pl.col("elevation_deg") > 5)
            )
            .with_columns(day=pl.col("time").dt.date())
        )
        days = (
            table.group_by("day")
            .agg(low=pl.col("low").sum(), rows=pl.len())
            .filter(pl.col("low") >= MIN_LOW_PER_DAY)
        )
        n_outage = int(days["low"].sum())
        n_days = days.height
        n_below += n_outage < STOP_COUNT
        print(f"farm {site}: outage half-hours {n_outage} on {n_days} days")
    print(f"farms under {STOP_COUNT}: {n_below}; stop rule fires: {n_below >= STOP_FARMS}")


if __name__ == "__main__":
    main()
