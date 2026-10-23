"""Count outage half-hours per farm in experiment 287's scored rows (experiment 301's stop rule).

An outage half-hour is a daylight half-hour (sun more than 5 degrees above the horizon) in which the
farm's output is under 5% of its plant forecast while that forecast exceeds 0.15 of capacity, on a
day with at least 6 such half-hours. The plant forecast is the unit response of experiment 224's
band-0 tables times the scale fitted on the months before the scored year. Farms appear only as
labels A to F.
"""

from datetime import UTC, datetime
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

E224_DIR: Final = ML_EXPERIMENTS_DATA_DIR / "00224-metered-farm-dayahead-two-number"
FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F")
SCORED_START: Final = datetime(2025, 9, 1, tzinfo=UTC)
SCORED_END: Final = datetime(2026, 9, 1, tzinfo=UTC)
MIN_LOW_PER_DAY: Final[int] = 6
STOP_COUNT: Final[int] = 200
STOP_FARMS: Final[int] = 4


def main() -> None:
    """Print outage half-hours per farm and whether the stop rule fires."""
    n_below = 0
    for site in FARMS:
        table = pl.read_parquet(E224_DIR / "members" / f"{site}_b0.parquet").sort("time")
        tuning = table.filter(pl.col("time") < SCORED_START)
        unit = tuning["unit_mean"].to_numpy()
        output = tuning["output"].to_numpy().astype(np.float64)
        scale = float(np.sum(output * unit) / np.sum(unit**2))
        scored = (
            table.filter((pl.col("time") >= SCORED_START) & (pl.col("time") < SCORED_END))
            .with_columns(
                forecast=pl.col("unit_mean") * scale, elevation=90.0 - pl.col("zenith_deg")
            )
            .with_columns(
                low=(pl.col("output") < 0.05 * pl.col("forecast"))
                & (pl.col("forecast") > 0.15)
                & (pl.col("elevation") > 5),
                day=pl.col("time").dt.date(),
            )
        )
        days = (
            scored.group_by("day")
            .agg(low=pl.col("low").sum())
            .filter(pl.col("low") >= MIN_LOW_PER_DAY)
        )
        n_outage = int(days["low"].sum())
        n_below += n_outage < STOP_COUNT
        n_loose = int(scored["low"].sum())
        loose_days = scored.filter(pl.col("low"))["day"].n_unique()
        before = table.filter(pl.col("time") < SCORED_START).with_columns(
            low=(pl.col("output") < 0.05 * pl.col("unit_mean") * scale)
            & (pl.col("unit_mean") * scale > 0.15)
            & (90.0 - pl.col("zenith_deg") > 5)
        )
        print(
            f"farm {site}: outage half-hours {n_outage} on {days.height} days; "
            f"loose count {n_loose} on {loose_days} days; "
            f"loose count before the scored year {int(before['low'].sum())}"
        )
    print(f"farms under {STOP_COUNT}: {n_below}; stop rule fires: {n_below >= STOP_FARMS}")


if __name__ == "__main__":
    main()
