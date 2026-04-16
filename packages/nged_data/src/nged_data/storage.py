from pathlib import Path

import patito as pt
import polars as pl
from contracts.power_schemas import PowerTimeSeries


def append_to_delta(power_time_series: pt.DataFrame[PowerTimeSeries], delta_path: Path) -> None:
    """
    Appends data to a Delta table, ensuring no duplicates based on (time_series_id, period_end_time).

    Args:
        power_time_series: The Patito DataFrame to append.
        delta_path: The path to the Delta table.
    """
    if delta_path.exists():
        old_df = pl.scan_delta(delta_path)
    else:
        delta_path.parent.mkdir(parents=True, exist_ok=True)
        old_df = pl.LazyFrame(schema=PowerTimeSeries.dtypes)

    grouped = old_df.group_by("time_series_id").agg(pl.max("time")).collect()
    del old_df  # Close the file

    new_power_ts = power_time_series.filter()

    if len(new_power_ts) > 0:
        new_power_ts.write_delta(
            delta_path, mode="append", delta_write_options={"partition_by": "time_series_id"}
        )
