from datetime import datetime, timedelta
import polars as pl


def scan_delta_table(delta_path: str) -> pl.LazyFrame:
    """Scan a Delta table and ensure UTC timezone."""
    return pl.scan_delta(delta_path).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
    )


def get_partition_window(
    partition_key: str, lookback_days: int = 1
) -> tuple[datetime, datetime, datetime]:
    """Get the partition window with a lookback."""
    partition_date = datetime.strptime(partition_key, "%Y-%m-%d")
    partition_start = partition_date
    partition_end = partition_date + timedelta(days=1)
    lookback_start = partition_date - timedelta(days=lookback_days)
    return partition_start, partition_end, lookback_start
