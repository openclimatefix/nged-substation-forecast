"""Logic for downloading Live Primary Data from NGED."""

import logging
from pathlib import Path
from typing import IO, Any, cast

import patito as pt
import polars as pl
from contracts.data_schemas import SubstationFlows, SubstationLocations

from .ckan_client import NGEDCKANClient

logger = logging.getLogger(__name__)


def download_live_primary_data(
    client: NGEDCKANClient,
    package_name: str,
) -> pt.DataFrame[SubstationFlows]:
    """Download all live primary data for a given region (package).

    Args:
        client: The NGED CKAN client.
        package_name: The name of the package (e.g., "live-primary-data---south-wales").

    Returns:
        pt.DataFrame: A dataframe containing the combined data for all substations.
    """
    package = client.get_package_show(package_name)
    dfs = []

    # We might want to parallelize this, but let's start simple.
    for resource in package["resources"]:
        url = resource.get("url")
        if resource["format"].lower() == "csv" and url and not url.startswith("redacted"):
            # Extract substation name from resource name
            # Resource name is typically "Substation Name Primary Transformer Flows"
            substation_name = resource["name"].replace(" Primary Transformer Flows", "")

            try:
                df = _primary_substation_csv_to_dataframe(url, substation_name=substation_name)
            except Exception as e:
                # Log error and continue with other resources
                logger.error("Failed to download resource %s: %s", resource["name"], e)
            else:
                dfs.append(df)

    return pt.DataFrame(pl.concat(dfs))


def _primary_substation_csv_to_dataframe(
    csv_data: str | Path | IO[str] | IO[bytes] | bytes, substation_name: str
) -> pt.DataFrame[SubstationFlows]:
    # Read CSV directly from URL using polars
    df = pl.read_csv(csv_data)

    # The CSV format can vary between substations.
    # We try to map the columns we find to our schema.
    rename_map = {}
    if "ValueDate" in df.columns:
        rename_map["ValueDate"] = "timestamp"

    # Possible names for MW and MVAr
    mw_cols = ["MW Inst", "MW"]
    mvar_cols = ["MVAr Inst", "MVAr"]

    for col in mw_cols:
        if col in df.columns:
            rename_map[col] = "MW"
            break

    for col in mvar_cols:
        if col in df.columns:
            rename_map[col] = "MVAr"
            break

    df = df.rename(rename_map)
    df = df.with_columns(substation_name=pl.lit(substation_name))

    # If MW or MVAr are missing, we add them as nulls so the schema validation passes
    # (since they are optional in the contract)
    if "MW" not in df.columns:
        df = df.with_columns(MW=pl.lit(None, dtype=pl.Float32))
    if "MVAr" not in df.columns:
        df = df.with_columns(MVAr=pl.lit(None, dtype=pl.Float32))

    # Cast timestamp to datetime. We specify time_zone="UTC" because
    # the data has a +00:00 suffix.
    df = df.with_columns(pl.col("timestamp").str.to_datetime(time_zone="UTC"))

    # Cast columns to the types specified in the schema
    df = df.cast(cast(Any, SubstationFlows.dtypes))
    df = df.select(SubstationFlows.columns)

    return SubstationFlows.validate(df)


def download_substation_locations(
    client: NGEDCKANClient,
) -> pt.DataFrame[SubstationLocations]:
    """Download substation metadata (locations).

    Args:
        client: The NGED CKAN client.

    Returns:
        pt.DataFrame: A dataframe containing substation metadata.
    """
    package = client.get_package_show(package_id="primary-substation-location-easting-northings")

    for resource in package["resources"]:
        if resource["format"].lower() == "csv":
            try:
                df = pl.read_csv(resource["url"])
                # The CSV has Easting/Northing and Latitude/Longitude
                # We rename to match our schema if necessary
                if "Substation Name" in df.columns:
                    df = df.rename({"Substation Name": "substation_name"})
                if "Latitude" in df.columns:
                    df = df.rename({"Latitude": "latitude"})
                if "Longitude" in df.columns:
                    df = df.rename({"Longitude": "longitude"})

                return SubstationLocations.validate(df, drop_superfluous_columns=True)
            except Exception as e:
                logger.error("Failed to download metadata: %s", e)

    return pt.DataFrame[SubstationLocations](pl.DataFrame())
