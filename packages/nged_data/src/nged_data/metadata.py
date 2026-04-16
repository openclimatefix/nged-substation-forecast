import logging
from pathlib import Path

import patito as pt
import polars as pl
from contracts.power_schemas import TimeSeriesMetadata

logger = logging.getLogger(__name__)


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

    if not metadata_path.exists():
        logger.info(f"Metadata file not found at {metadata_path}. Creating new file.")
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
        logger.info("Metadata is up to date.")
    else:
        logger.info(f"Metadata update detected at {metadata_path}. Updating metadata file.")
        merged_metadata.write_parquet(metadata_path)
