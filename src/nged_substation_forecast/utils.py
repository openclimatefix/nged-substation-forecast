from datetime import datetime, timedelta, timezone
import polars as pl


def scan_delta_table(delta_path: str) -> pl.LazyFrame:
    """Scan a Delta table and ensure UTC timezone."""
    return pl.scan_delta(delta_path).with_columns(
        pl.col("start_time").cast(pl.Datetime("us", "UTC"))
    )


def get_partition_window(
    partition_key: str, lookback_days: int = 1
) -> tuple[datetime, datetime, datetime]:
    """Get the partition window with a lookback."""
    partition_date = datetime.strptime(partition_key, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    partition_start = partition_date
    partition_end = partition_date + timedelta(days=1)
    lookback_start = partition_date - timedelta(days=lookback_days)
    return partition_start, partition_end, lookback_start
