from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import polars as pl


def find_one_match[T](predicate: Callable[[T], bool], haystack: list[T]) -> T:
    """Find exactly one match in haystack. Raise a ValueError if haystack is empty, or if there is
    more than 1 match."""
    if len(haystack) == 0:
        raise ValueError("haystack is empty!")
    filtered = list(filter(predicate, haystack))
    if len(filtered) != 1:
        raise ValueError(f"Found {len(filtered)} matches when we were expecting exactly 1!")
    return filtered[0]


def change_dataframe_column_names_to_snake_case(df: pl.DataFrame) -> pl.DataFrame:
    new_col_names = {col: to_snake_case(col) for col in df.columns}
    return df.rename(new_col_names)


def to_snake_case(s: str) -> str:
    return s.lower().replace(" ", "_")


def ensure_utc_timestamp_lazy(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Ensures the timestamp column is UTC timezone-aware lazily.

    This handles cases where Delta tables lose timezone metadata or Polars scans
    them as naive. It also handles non-UTC timezones by converting them.

    Args:
        lf: Polars LazyFrame with a 'timestamp' column.

    Returns:
        LazyFrame with UTC-aware 'timestamp' column.
    """
    schema = lf.collect_schema()
    # If the timestamp column is not present, we return early to avoid errors.
    # This is useful when scanning tables that might not have a timestamp column.
    if "timestamp" not in schema:
        return lf

    timestamp_dtype = schema["timestamp"]

    if isinstance(timestamp_dtype, pl.Datetime) and timestamp_dtype.time_zone is None:
        return lf.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
    elif isinstance(timestamp_dtype, pl.Datetime) and timestamp_dtype.time_zone != "UTC":
        return lf.with_columns(pl.col("timestamp").dt.convert_time_zone("UTC"))
    return lf


def scan_delta_table(path: str | Path) -> pl.LazyFrame:
    """Scans a Delta table and ensures the timestamp column is UTC-aware.

    Args:
        path: Path to the Delta table.

    Returns:
        Polars LazyFrame with UTC-aware 'timestamp' column (if present).
    """
    return ensure_utc_timestamp_lazy(pl.scan_delta(str(path)))


def get_partition_window(
    partition_key: str, lookback_days: int = 0
) -> tuple[datetime, datetime, datetime]:
    """Calculates the partition start, end, and lookback start times.

    Args:
        partition_key: The Dagster partition key (ISO format date string).
        lookback_days: Number of days to look back from the partition start.

    Returns:
        A tuple of (partition_start, partition_end, lookback_start).
    """
    # We use fromisoformat to handle the partition key, which is expected to be a date string.
    # We ensure the resulting datetime is UTC-aware.
    partition_start = datetime.fromisoformat(partition_key).replace(tzinfo=timezone.utc)
    partition_end = partition_start + timedelta(days=1)
    lookback_start = partition_start - timedelta(days=lookback_days)
    return partition_start, partition_end, lookback_start
