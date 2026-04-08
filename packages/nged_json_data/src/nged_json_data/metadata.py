import polars as pl
from pathlib import Path
from dagster import get_dagster_logger


def upsert_metadata(new_metadata: pl.DataFrame, metadata_path: Path) -> None:
    """
    Upserts metadata to a Parquet file.

    This function assumes it is called by the exclusive owner asset, so no
    explicit locking is required.

    If the local Parquet file does not exist, it saves the new_metadata.
    If it exists, it reads it, compares it with new_metadata, and updates
    the Parquet file if there are differences, logging a warning.

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

    # Compare metadata
    # We use `equals` to check if the DataFrames are identical.
    if not existing_metadata.equals(new_metadata):
        logger.warning(f"Metadata mismatch detected at {metadata_path}. Updating metadata file.")
        new_metadata.write_parquet(metadata_path)
    else:
        logger.info("Metadata is up to date.")
