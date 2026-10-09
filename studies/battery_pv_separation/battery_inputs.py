"""Load the inputs that every script of the battery study shares.

The loaders that other study folders also need (`market_frame`, `battery_output`,
`p99_output_mw`, and the calendar constants) live in `studies.battery_market`. This module adds what
only the battery-and-solar separation study uses: the four batteries, the within-day price rank, and
the output folder.
"""

from pathlib import Path
from typing import Final

import polars as pl
from studies.battery_market import (
    B1610_DIR,
    B1610_SUFFIX,
    FOLD_MONTHS,
    HALF_HOURS_PER_DAY,
    MINUTES_PER_HALF_HOUR,
    WEEK_START,
    WINDOW_START,
    battery_output,
    market_frame,
    p99_output_mw,
)
from studies.sources import BATTERY_PV_SEPARATION_DIR

__all__ = [
    "B1610_DIR",
    "B1610_SUFFIX",
    "BATTERIES",
    "FOLD_MONTHS",
    "HALF_HOURS_PER_DAY",
    "MINUTES_PER_HALF_HOUR",
    "NAMES",
    "OUTPUT_DIR",
    "TERCILE_LABELS",
    "WEEK_START",
    "WINDOW_START",
    "battery_frame",
    "battery_output",
    "market_frame",
    "p99_output_mw",
]

OUTPUT_DIR: Final[Path] = BATTERY_PV_SEPARATION_DIR
"""Where this study's results live."""
BATTERIES: Final[tuple[str, ...]] = ("T_LKSDB-1", "E_DOLLB-1", "T_THURB-1", "T_OCHLB-1")
"""The four batteries. The first two were named in advance. The other two have the largest energy
throughput among the list's batteries (pumped storage excluded) that are not a second unit at a
site already listed."""
NAMES: Final[dict[str, str]] = {
    "T_LKSDB-1": "Lakeside",
    "E_DOLLB-1": "Dollymans",
    "T_THURB-1": "Thurrock",
    "T_OCHLB-1": "Ocker Hill",
}
"""A short name for each battery, used in chart labels."""
TERCILE_LABELS: Final[tuple[str, str, str]] = (
    "Cheapest third of the day",
    "Middle third",
    "Dearest third of the day",
)
"""The within-day price terciles, cheapest first."""
EXPECTED_ROWS: Final[int] = 365 * HALF_HOURS_PER_DAY
"""Half-hours in the window."""


def battery_frame(*, bmu_id: str) -> pl.DataFrame:
    """Return a battery's output joined to prices, with the within-day price rank.

    Args:
        bmu_id: The Elexon BMU identifier.

    Returns:
        One row per half-hour in which both the output and the day-ahead price exist, with the
        output, the prices, `date` (the UTC day), `tod` (the half-hour of the UTC day, 0 to 47),
        `rank_pct` (the half-hour's day-ahead price rank among its day's 48, ties averaged, scaled
        to the open interval 0 to 1), `tercile` (0 for the cheapest third of the day), `month`
        (`YYYY-MM`), and `fold`.
    """
    joined = battery_output(bmu_id=bmu_id).join(market_frame(), on="time", how="inner")
    return (
        joined.drop_nulls("day_ahead_gbp_per_mwh")
        .with_columns(
            date=pl.col("time").dt.date(),
            tod=pl.col("time").dt.hour().cast(pl.Int32) * 2 + pl.col("time").dt.minute() // 30,
            month=pl.col("time").dt.strftime("%Y-%m"),
            calendar_month=pl.col("time").dt.month(),
        )
        .with_columns(
            rank_pct=(
                (pl.col("day_ahead_gbp_per_mwh").rank("average").over("date") - 0.5)
                / HALF_HOURS_PER_DAY
            ),
            fold=pl.col("calendar_month").replace_strict(FOLD_MONTHS, return_dtype=pl.Int64),
        )
        .with_columns(tercile=(pl.col("rank_pct") * 3).floor().cast(pl.Int64).clip(0, 2))
        .drop("calendar_month")
        .sort("time")
    )
