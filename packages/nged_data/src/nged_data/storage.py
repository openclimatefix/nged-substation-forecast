import logging
from pathlib import Path
from typing import cast

import patito as pt
import polars as pl
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata

log = logging.getLogger(__name__)


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
    log.info(f"Preparing to append_to_delta at {delta_path}...")

    if delta_path.exists():
        # Scan the existing delta table and find the max time per time_series_id
        max_times = cast(
            pl.DataFrame,  # Cast to pl.DataFrame to keep type checkers happy.
            pl.scan_delta(delta_path)
            .group_by("time_series_id")
            .agg(pl.max("time").alias("max_time"))
            .collect(),
        )

        log.info(
            f"Loaded max times for {max_times.height} time_series_ids from {delta_path}."
            f" Earliest time = {max_times['max_time'].min()}."
            f" Latest time = {max_times['max_time'].max()}"
        )
    else:
        log.info(f"{delta_path=} does not exist. Creating...")
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

    new_power_ts = PowerTimeSeries.sort(new_power_ts)

    log.info(
        f"Appending {new_power_ts.height:,d} rows of new PowerTimeSeries"
        f" (from {new_power_ts['time'].min()} to {new_power_ts['time'].max()}) to {delta_path=}"
    )

    PowerTimeSeries.validate(new_power_ts)

    if not new_power_ts.is_empty():
        new_power_ts.write_delta(
            delta_path, mode="append", delta_write_options={"partition_by": "time_series_id"}
        )


def upsert_metadata(new_metadata: pt.DataFrame[TimeSeriesMetadata], metadata_path: Path) -> None:
    """
    Upserts metadata to a Parquet file.

    This function assumes it is called by the exclusive owner asset, so no
    explicit locking is required.

    If the local Parquet file does not exist, it saves the new_metadata.
    If it exists, it merges the new_metadata with the existing metadata,
    keeping the latest version for each time_series_id, and updates the
    Parquet file if there are differences.

    Args:
        new_metadata: The new metadata DataFrame.
        metadata_path: The path to the Parquet file where we store our local version of the
        metadata.
    """
    new_metadata: pl.DataFrame = new_metadata.sort("time_series_id")
    TimeSeriesMetadata.validate(new_metadata)

    if not metadata_path.exists():
        log.info(f"Metadata file not found at {metadata_path}. Creating new file.")
        new_metadata.write_parquet(metadata_path)
        return

    # Read existing metadata
    existing_metadata = TimeSeriesMetadata.validate(pl.read_parquet(metadata_path))

    # Merge metadata
    # Put new_metadata first so that unique(keep="first") keeps the new version
    merged_metadata = TimeSeriesMetadata.validate(
        pl.concat([new_metadata, existing_metadata]).unique(subset="time_series_id", keep="first")
    ).sort("time_series_id")

    # Compare metadata
    if existing_metadata.equals(merged_metadata):
        log.info("Metadata is up to date.")
    else:
        log.info(f"Metadata update detected at {metadata_path}. Updating metadata file.")
        merged_metadata.write_parquet(metadata_path)
