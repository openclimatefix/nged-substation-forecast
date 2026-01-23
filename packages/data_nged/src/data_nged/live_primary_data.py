"""Logic for downloading Live Primary Data from NGED."""

import logging
from pathlib import Path
from typing import IO

import patito as pt
import polars as pl
from contracts.data_schemas import SubstationFlows, SubstationLocations

from .ckan_client import NGEDCKANClient

logger = logging.getLogger(__name__)


def get_substation_resource_urls(
    client: NGEDCKANClient,
    package_name: str,
) -> dict[str, str]:
    """Get the URLs for all substation resources in a package. Return mapping from name->url"""
    package = client.get_package_show(package_name)
    resources = package.get("resources", [])

    substation_resources = {}
    for resource in resources:
        url = resource.get("url")
        resource_format = resource.get("format", "")
        if resource_format.lower() == "csv" and url and not url.startswith("redacted"):
            # Extract substation name from resource name
            substation_name = resource["name"].replace(" Primary Transformer Flows", "")
            substation_resources[substation_name] = url

    return substation_resources


def read_primary_substation_csv(
    csv_data: str | Path | IO[str] | IO[bytes] | bytes, substation_name: str
) -> pt.DataFrame[SubstationFlows]:
    """Read a primary substation CSV and validate it against the schema."""
    df: pl.DataFrame = pl.read_csv(csv_data)
    df = df.with_columns(substation_name=pl.lit(substation_name))

    # The CSV column names vary between NGED license areas:
    # East Midlands (e.g. Abington)  : ValueDate,                       MVA,                     Volts
    # West Midlands (e.g. Albrighton): ValueDate, Amps,                 MVA, MVAr,      MW,      Volts
    # South Wales   (e.g. Aberaeron) : ValueDate, Current Inst, Derived MVA, MVAr Inst, MW Inst, Volts Inst
    # South West    (e.g. Filton Dc) : ValueDate, Current Inst,              MVAr Inst, MW Inst, Volts Inst
    # New format    (e.g. Regent St) : site, time, unit, value
    if "unit" in df.columns and "value" in df.columns:
        if (df["unit"] == "MVA").all():
            # Handle Regent Street primary substation (in the East Midlands), which uses a completely
            # different CSV structure. See `example_csv_data/regent-street.csv`.
            df = df.rename({"time": "timestamp", "value": "MVA"}, strict=False)
        elif (df["unit"] == "MW").all():
            # Handle milford-haven-grid.csv.
            df = df.rename({"time": "timestamp", "value": "MW"}, strict=False)
        else:
            raise ValueError(f"Unexpected unit in CSV: {df['unit'].unique().to_list()}")

    df = df.rename(
        {"ValueDate": "timestamp", "MW Inst": "MW", "MVAr Inst": "MVAr", "Derived MVA": "MVA"},
        strict=False,
    )

    df = df.with_columns(pl.col("timestamp").str.to_datetime(time_zone="UTC"))
    columns = set(SubstationFlows.columns).intersection(df.columns)
    df = df.select(columns)
    dtypes = {col: SubstationFlows.dtypes[col] for col in columns}
    df = df.cast(dtypes)

    return SubstationFlows.validate(df, allow_missing_columns=True)


def download_substation_locations(
    client: NGEDCKANClient,
) -> pt.DataFrame[SubstationLocations]:
    """Download substation metadata (locations)."""
    package = client.get_package_show(package_id="primary-substation-location-easting-northings")

    for resource in package["resources"]:
        if resource["format"].lower() == "csv":
            df = pl.read_csv(resource["url"])
            df = df.rename(
                {
                    "Substation Name": "substation_name",
                    "Latitude": "latitude",
                    "Longitude": "longitude",
                },
                strict=False,
            )
            return SubstationLocations.validate(df, drop_superfluous_columns=True)

    return pt.DataFrame[SubstationLocations](pl.DataFrame(schema=SubstationLocations.dtypes))
