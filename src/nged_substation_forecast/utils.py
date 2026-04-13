import polars as pl
import patito as pt
from datetime import datetime, timedelta, timezone
from pathlib import Path
from contracts.data_schemas import PowerTimeSeries
from typing import cast


def scan_delta_table(delta_path: str) -> pt.LazyFrame[PowerTimeSeries]:
    """Scan a Delta table."""
    return cast(pt.LazyFrame[PowerTimeSeries], pl.scan_delta(delta_path))


def get_partition_window(
    partition_key: str, lookback_days: int = 1
) -> tuple[datetime, datetime, datetime]:
    """Get the partition window with a lookback."""
    if len(partition_key) == 10:  # YYYY-MM-DD
        partition_date = datetime.strptime(partition_key, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        partition_end = partition_date + timedelta(days=1)
    elif len(partition_key) == 16:  # YYYY-MM-DD-HH:MM
        partition_date = datetime.strptime(partition_key, "%Y-%m-%d-%H:%M").replace(
            tzinfo=timezone.utc
        )
        partition_end = partition_date + timedelta(hours=6)
    else:
        raise ValueError(f"Unsupported partition_key format: {partition_key}")

    partition_start = partition_date
    lookback_start = partition_date - timedelta(days=lookback_days)
    return partition_start, partition_end, lookback_start


def filter_new_delta_records(
    df: pt.DataFrame[PowerTimeSeries], delta_path: Path
) -> pt.DataFrame[PowerTimeSeries]:
    """Filters a DataFrame to only include records newer than the maximum timestamp in an existing Delta table.

    This ensures idempotency when ingesting new data into an existing Delta table,
    preventing duplicate records from being written.
    """
    if not delta_path.exists():
        return df

    # Read the existing delta table to find the max timestamp.
    # We use scan_delta to avoid loading the whole table into memory.
    existing_data = scan_delta_table(str(delta_path))

    # Extract the max timestamp directly. If the table is empty, this returns None.
    # This avoids unnecessary DataFrame instantiation and is more idiomatic Polars.
    max_timestamp = cast(
        pl.DataFrame, existing_data.select(pl.col("period_end_time").max()).collect()
    ).item()

    if max_timestamp is None:
        return df

    # Filter the input dataframe to only include records newer than the max timestamp.
    return df.filter(pl.col("period_end_time") > max_timestamp)


def filter_to_partition_window(
    df: pt.DataFrame[PowerTimeSeries], partition_start: datetime, partition_end: datetime
) -> pt.DataFrame[PowerTimeSeries]:
    """Filters a DataFrame to strictly within a partition window [start, end).

    This is used to ensure that data cleaning and rolling window calculations
    are correctly scoped to the specific partition being processed, preventing
    data leakage across partition boundaries.
    """
    return df.filter(
        (pl.col("period_end_time") >= partition_start) & (pl.col("period_end_time") < partition_end)
    )


def get_delta_partition_predicate(
    partition_start: datetime, partition_end: datetime, time_column: str = "period_end_time"
) -> str:
    """Generates a SQL predicate string for Delta table partition replacement.

    This predicate is used when writing to Delta tables to ensure that only
    the data within the specified partition window is replaced, maintaining
    data integrity.
    """
    return f"{time_column} >= '{partition_start.isoformat()}' AND {time_column} < '{partition_end.isoformat()}'"
