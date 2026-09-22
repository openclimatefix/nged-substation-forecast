"""Put half-hourly power readings onto an hourly grid, on the period-ending convention."""

from typing import Final

import polars as pl

HALF_HOURS_PER_HOUR: Final[int] = 2
"""How many half-hourly readings a complete hour is built from."""


def hourly_from_half_hourly(*, half_hourly: pl.DataFrame) -> pl.DataFrame:
    """Average half-hourly power onto the hourly, period-ending grid a weather product uses.

    Both frames label a period by its end, so the hour ending at `T` is the mean of the half-hours
    ending at `T - 30 min` and at `T`. Rolling each stamp forward 30 minutes and truncating to the
    hour is what maps both onto `T`.

    **Take the stamps at face value.** `contracts.PowerTimeSeries` states that `time` is the end of
    the observation period for every row, and `PowerTimeSeries.correct_late_timestamps` is what
    makes that true of the readings NGED stamped half an hour late. Shifting again here would undo
    the repair on 93% of the rows, and a doubly-shifted stamp is a plausible half-hour rather than
    an error, so nothing downstream would notice.

    An hour is produced only where both of its half-hours are present, so a partly-missing hour is
    dropped rather than silently becoming a one-reading mean. That filter is also what drops the
    orphan half-hour at the repair boundary: a repaired series has no reading at
    `POWER_TIMESTAMPS_CORRECTED_BEFORE - 30 min`, because NGED never published that half-hour, so
    the hour ending at `POWER_TIMESTAMPS_CORRECTED_BEFORE - 30 min` holds one reading and goes.

    Args:
        half_hourly: One row per `(site, time)`, carrying `power_mw`.

    Returns:
        One row per `(site, time)` on the hourly grid, carrying `power_mw` and
        `has_zero_half_hour`, sorted by site then time.
    """
    return (
        half_hourly.with_columns(hour_end=pl.col("time").dt.offset_by("30m").dt.truncate("1h"))
        .group_by("site", "hour_end")
        .agg(
            power_mw=pl.col("power_mw").mean(),
            n_half_hours=pl.len(),
            has_zero_half_hour=(pl.col("power_mw") == 0.0).any(),
        )
        .filter(pl.col("n_half_hours") == HALF_HOURS_PER_HOUR)
        .drop("n_half_hours")
        .rename({"hour_end": "time"})
        .sort("site", "time")
    )
