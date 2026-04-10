import polars as pl
from pathlib import Path
from dagster import get_dagster_logger
import io


def get_df_hash(df: pl.DataFrame) -> int:
    """Calculates a hash of the DataFrame."""
    buffer = io.BytesIO()
    df.write_parquet(buffer)
    return hash(buffer.getvalue())


def upsert_metadata(new_metadata: pl.DataFrame, metadata_path: Path) -> None:
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
        metadata_path: The path to the Parquet file.
    """
    logger = get_dagster_logger()

    if not metadata_path.exists():
        logger.info(f"Metadata file not found at {metadata_path}. Creating new file.")
        new_metadata.write_parquet(metadata_path)
        return

    # Read existing metadata
    existing_metadata = pl.read_parquet(metadata_path)

    # Merge metadata
    # Put new_metadata first so that unique() keeps the new version
    merged_metadata = pl.concat([new_metadata, existing_metadata]).unique(subset="time_series_id")

    # Compare metadata
    # We use `hash_rows().sum()` to check if the DataFrames are identical.
    if get_df_hash(existing_metadata) != get_df_hash(merged_metadata):
        logger.info(f"Metadata update detected at {metadata_path}. Updating metadata file.")
        merged_metadata.write_parquet(metadata_path)
    else:
        logger.info("Metadata is up to date.")
