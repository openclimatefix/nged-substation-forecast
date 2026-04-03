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
    timestamp_dtype = schema["timestamp"]

    if isinstance(timestamp_dtype, pl.Datetime) and timestamp_dtype.time_zone is None:
        return lf.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
    elif isinstance(timestamp_dtype, pl.Datetime) and timestamp_dtype.time_zone != "UTC":
        return lf.with_columns(pl.col("timestamp").dt.convert_time_zone("UTC"))
    return lf
