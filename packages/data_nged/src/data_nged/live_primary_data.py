"""Logic for downloading Live Primary Data from NGED."""

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import IO

import patito as pt
import polars as pl
from contracts.data_schemas import SubstationFlows, SubstationLocations

from .ckan_client import NGEDCKANClient

logger = logging.getLogger(__name__)


@dataclass
class SubstationDownloadResult:
    """The result of downloading and validating a single substation."""

    substation_name: str
    df: pt.DataFrame[SubstationFlows] | None = None
    errors: list[str] = field(default_factory=list)


def download_live_primary_data(
    client: NGEDCKANClient,
    package_name: str,
) -> Iterable[SubstationDownloadResult]:
    """Download all live primary data for a given region (package).

    Args:
        client: The NGED CKAN client.
        package_name: The name of the package (e.g., "live-primary-data---south-wales").

    Yields:
        SubstationDownloadResult: The result for each substation.
    """
    package = client.get_package_show(package_name)
    resources = package.get("resources", [])
    logger.info("Found %d resources in package %s", len(resources), package_name)

    for resource in resources:
        url = resource.get("url")
        resource_format = resource.get("format", "")
        if resource_format.lower() == "csv" and url and not url.startswith("redacted"):
            # Extract substation name from resource name
            substation_name = resource["name"].replace(" Primary Transformer Flows", "")

            try:
                # Use the client session to download the data with retries
                response = client.session.get(url, timeout=30)
                response.raise_for_status()

                df = __read_primary_substation_csv(
                    BytesIO(response.content), substation_name=substation_name
                )
                yield SubstationDownloadResult(
                    substation_name=substation_name,
                    df=df,
                )
            except Exception as e:
                logger.error(
                    "Failed to download or validate resource %s from %s: %s",
                    resource["name"],
                    url,
                    e,
                )
                yield SubstationDownloadResult(
                    substation_name=substation_name,
                    errors=[str(e)],
                )


def __read_primary_substation_csv(
    csv_data: str | Path | IO[str] | IO[bytes] | bytes, substation_name: str
) -> pt.DataFrame[SubstationFlows]:
    df: pl.DataFrame = pl.read_csv(csv_data)
    df = df.with_columns(substation_name=pl.lit(substation_name))

    # The CSV column names vary between NGED license areas:
    # East Midlands (e.g. Abington)  : ValueDate,                       MVA,                     Volts
    # West Midlands (e.g. Albrighton): ValueDate, Amps,                 MVA, MVAr,      MW,      Volts
    # South Wales   (e.g. Aberaeron) : ValueDate, Current Inst, Derived MVA, MVAr Inst, MW Inst, Volts Inst
    # South West    (e.g. Filton Dc) : ValueDate, Current Inst,              MVAr Inst, MW Inst, Volts Inst
    # New format    (e.g. Regent St) : site, time, unit, value
    if "unit" in df.columns and "value" in df.columns:
        # Handle Regent Street primary substation (in the East Midlands), which uses a completely
        # different CSV structure. See `example_csv_data/regent-street.csv`.
        if not (df["unit"] == "MVA").all():
            raise ValueError(f"Unexpected unit in CSV: {df['unit'].unique().to_list()}")
        df = df.rename({"time": "timestamp", "value": "MVA"}, strict=False)

    df = df.rename(
        {"ValueDate": "timestamp", "MW Inst": "MW", "MVAr Inst": "MVAr", "Derived MVA": "MVA"},
        strict=False,
    )

    df = df.with_columns(pl.col("timestamp").str.to_datetime(time_zone="UTC"))
    columns = set(SubstationFlows.columns).intersection(df.columns)
    df = df.select(columns)

    # Cast to ensure consistency with the schema
    for col in columns:
        df = df.with_columns(pl.col(col).cast(SubstationFlows.dtypes[col]))

    return SubstationFlows.validate(df, allow_missing_columns=True)


def download_substation_locations(
    client: NGEDCKANClient,
) -> pt.DataFrame[SubstationLocations]:
    """Download substation metadata (locations)."""
    package = client.get_package_show(package_id="primary-substation-location-easting-northings")

    for resource in package["resources"]:
        if resource["format"].lower() == "csv":
            try:
                # Use the client session to download the data with retries
                response = client.session.get(resource["url"], timeout=30)
                response.raise_for_status()

                df = pl.read_csv(BytesIO(response.content))
                df = df.rename(
                    {
                        "Substation Name": "substation_name",
                        "Latitude": "latitude",
                        "Longitude": "longitude",
                    },
                    strict=False,
                )
                return SubstationLocations.validate(df, drop_superfluous_columns=True)
            except Exception as e:
                logger.error("Failed to download metadata: %s", e)

    return pt.DataFrame[SubstationLocations](pl.DataFrame(schema=SubstationLocations.dtypes))
