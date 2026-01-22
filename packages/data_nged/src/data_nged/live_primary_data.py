"""Logic for downloading Live Primary Data from NGED."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from .ckan_client import NGEDCKANClient

logger = logging.getLogger(__name__)


def download_live_primary_data(
    client: NGEDCKANClient,
    package_name: str,
) -> pl.DataFrame:
    """Download all live primary data for a given region (package).

    Args:
        client: The NGED CKAN client.
        package_name: The name of the package (e.g., "live-primary-data---south-wales").

    Returns:
        pl.DataFrame: A dataframe containing the combined data for all substations.
    """
    package = client.get_package_show(package_name)
    dfs = []

    # In a real scenario, we might want to parallelize this,
    # but let's start simple.
    for resource in package["resources"]:
        if resource["format"].lower() == "csv":
            # Extract substation name from resource name
            # Resource name is typically "Substation Name Primary Transformer Flows"
            substation_name = resource["name"].replace(" Primary Transformer Flows", "")

            try:
                # Read CSV directly from URL using polars
                df = pl.read_csv(resource["url"])

                # Standardize columns
                df = df.rename({"ValueDate": "timestamp", "MW": "mw", "MVAr": "mvar"})

                # Add substation_id/name
                df = df.with_columns(
                    [
                        pl.lit(substation_name).alias("substation_name"),
                        # We might want a more robust ID, but for now name is what we have
                        pl.lit(substation_name).alias("substation_id"),
                    ],
                )

                # Select only relevant columns
                df = df.select(["substation_id", "substation_name", "timestamp", "mw", "mvar"])

                # Cast timestamp to datetime
                df = df.with_columns(pl.col("timestamp").str.to_datetime())

                dfs.append(df)
            except Exception as e:
                # Log error and continue with other resources
                logger.error("Failed to download resource %s: %s", resource["name"], e)

    if not dfs:
        return pl.DataFrame()

    return pl.concat(dfs)


def download_substation_metadata(
    client: NGEDCKANClient,
) -> pl.DataFrame:
    """Download substation metadata (locations).

    Args:
        client: The NGED CKAN client.

    Returns:
        pl.DataFrame: A dataframe containing substation metadata.
    """
    package_name = "primary-substation-location-easting-northings"
    package = client.get_package_show(package_name)

    for resource in package["resources"]:
        if resource["format"].lower() == "csv":
            try:
                df = pl.read_csv(resource["url"])
                # We'll need to map Easting/Northing to Lat/Lon if needed,
                # but for now let's just return what we have.
                # Assuming columns: Substation Name, Easting, Northing
                return df
            except Exception as e:
                logger.error("Failed to download metadata: %s", e)

    return pl.DataFrame()
