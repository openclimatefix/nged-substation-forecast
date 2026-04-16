from pathlib import Path
from typing import cast

import patito as pt
import polars as pl
from contracts.power_schemas import PowerTimeSeries


class _MaxTimePerTimeSeriesId(pt.Model):
    time_series_id: int = pt.Field(dtype=PowerTimeSeries.dtypes["time_series_id"])
    max_time: int = pt.Field(dtype=PowerTimeSeries.dtypes["time"])


def append_to_delta(power_time_series: pt.DataFrame[PowerTimeSeries], delta_path: Path) -> None:
    """
    Appends data to a Delta table, ensuring no duplicates based on (time_series_id, period_end_time).
    Args:
        power_time_series: The Patito DataFrame to append.
        delta_path: The path to the Delta table.
    """
    if delta_path.exists():
        # Scan the existing delta table and find the max time per time_series_id
        max_times = cast(
            pl.DataFrame,
            pl.scan_delta(delta_path)
            .group_by("time_series_id")
            .agg(pl.max("time").alias("max_time"))
            .collect(),
        )
    else:
        delta_path.parent.mkdir(parents=True, exist_ok=True)
        # Create an empty DataFrame with the correct schema for the join
        max_times = pl.DataFrame(schema=_MaxTimePerTimeSeriesId.dtypes)

    _MaxTimePerTimeSeriesId.validate(max_times)

    # Left join the new data with the small max_times dataframe
    # Filter for rows where the time is strictly greater than the max existing time,
    # or where max_time is null (which means it's a brand new time_series_id)
    new_power_ts = cast(
        pl.DataFrame,
        power_time_series.lazy()
        .join(max_times.lazy(), on="time_series_id", how="left")
        .filter(pl.col("max_time").is_null() | (pl.col("time") > pl.col("max_time")))
        .drop("max_time")
        .collect(),
    )

    PowerTimeSeries.validate(new_power_ts)

    if new_power_ts.height > 0:
        new_power_ts.write_delta(
            delta_path, mode="append", delta_write_options={"partition_by": "time_series_id"}
        )
