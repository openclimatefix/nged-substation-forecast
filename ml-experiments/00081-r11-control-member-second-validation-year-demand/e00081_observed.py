"""Build the observed power and its four lags for the 13 Primary substations of experiment 00081.

The table has the schema of the stage 1 pilot's `observed.parquet`: one row per observed
`(time_series_id, valid_time)` with `power` in MW and a `power_lag_<h>h` column for each of 24, 48,
168, and 336 hours, null where the lagged half-hour has no reading. It covers the half-hourly grid
from 2021-03-01 to 2024-04-30, read from the cleaned power table (every row that no cleaning rule
flagged). The script asserts that each series has a reading from before the first backfill run plus
336 hours, so the longest lag is defined for every run.

    uv run python ml-experiments/00081-*/e00081_observed.py
"""

from datetime import UTC, datetime

import e00081_common as common
import polars as pl
from studies.power import scan_power


def observed_power(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return cleaned power in MW on the half-hourly grid with the lagged values.

    Args:
        series: `common.series_table()`.

    Returns:
        One row per observed `(time_series_id, valid_time)` from `common.GRID_START` to
        `common.GRID_END`, with `power` and a `power_lag_<h>h` column per entry of
        `common.LAG_HOURS`.
    """
    power = (
        scan_power()
        .filter(
            pl.col("time_series_id").is_in(series["time_series_id"].to_list()),
            pl.col("time") >= common.GRID_START,
            pl.col("time") <= common.GRID_END,
        )
        .select("time_series_id", pl.col("time").alias("valid_time"), "power")
        .collect()
        .filter(pl.col("power").is_finite())
    )
    frame = power
    for name, hours in common.LAG_HOURS.items():
        lagged = power.select(
            "time_series_id",
            (pl.col("valid_time") + pl.duration(hours=int(hours))).alias("valid_time"),
            pl.col("power").alias(name),
        )
        frame = frame.join(lagged, on=["time_series_id", "valid_time"], how="left")
    return frame.sort("time_series_id", "valid_time")


def assert_history(*, observed: pl.DataFrame, series: pl.DataFrame) -> list[str]:
    """Check each series has power from before the first run plus the longest lag.

    Args:
        observed: `observed_power`'s output.
        series: `common.series_table()`.

    Returns:
        One report line per series: its first and last reading and its reading count.

    Raises:
        ValueError: If a series has no reading before the deadline.
    """
    first_run = datetime(
        common.BACKFILL_FIRST_RUN.year,
        common.BACKFILL_FIRST_RUN.month,
        common.BACKFILL_FIRST_RUN.day,
        tzinfo=UTC,
    )
    deadline = first_run + common.LAG_WARMUP
    lines = []
    for row in series.iter_rows(named=True):
        mine = observed.filter(pl.col("time_series_id") == row["time_series_id"])["valid_time"]
        if mine.is_empty() or mine.min() > deadline:  # ty: ignore[unsupported-operator]
            msg = f"{row['site']}: no cleaned power before {deadline}"
            raise ValueError(msg)
        lines.append(f"{row['site']}: {mine.min()} to {mine.max()}, {mine.len():,} readings")
    return lines


def main() -> None:
    """Write the observed table and print each series' span."""
    series = common.series_table()
    observed = observed_power(series=series)
    print("\n".join(assert_history(observed=observed, series=series)))
    common.atomic_write(frame=observed, target=common.OBSERVED_PATH)
    print(f"wrote {observed.height:,} rows to {common.OBSERVED_PATH}")


if __name__ == "__main__":
    main()
