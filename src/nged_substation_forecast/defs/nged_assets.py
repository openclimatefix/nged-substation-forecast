"""Dagster assets for NGED data."""

import polars as pl
from dagster import AssetExecutionContext, Config, Output, asset
from contracts.data_schemas import SubstationFlows
from data_nged.ckan_client import NGEDCKANClient
from data_nged.live_primary_data import download_live_primary_data


class NgedLivePrimaryDataConfig(Config):
    """Configuration for NGED live primary data."""

    output_path: str


@asset(group_name="nged")
def nged_live_primary_data_south_wales(context: AssetExecutionContext) -> Output[pl.DataFrame]:
    """Live primary data for South Wales from NGED."""
    client = NGEDCKANClient()
    df = download_live_primary_data(client, "live-primary-data---south-wales")
    # Validate against our contract
    SubstationFlows.validate(df)
    context.log.info(f"Downloaded and validated {len(df)} rows of data")
    return Output(df)


@asset(group_name="nged")
def nged_live_primary_data(
    context: AssetExecutionContext,
    config: NgedLivePrimaryDataConfig,
    nged_live_primary_data_south_wales: pl.DataFrame,
    nged_live_primary_data_south_west: pl.DataFrame,
    nged_live_primary_data_west_midlands: pl.DataFrame,
    nged_live_primary_data_east_midlands: pl.DataFrame,
) -> None:
    """Consolidated live primary data for all regions, saved to Parquet."""
    df = pl.concat(
        [
            nged_live_primary_data_south_wales,
            nged_live_primary_data_south_west,
            nged_live_primary_data_west_midlands,
            nged_live_primary_data_east_midlands,
        ]
    )

    # Sort first by substation_name, and then by datetime (timestamp)
    df = df.sort(["substation_name", "timestamp"])

    # Partition by month
    # We create a temporary month column for partitioning
    df = df.with_columns(month=pl.col("timestamp").dt.strftime("%Y-%m"))

    context.log.info(f"Saving {len(df)} rows to {config.output_path} partitioned by month")

    # Save to Parquet. Polars can use obstore under the hood for cloud storage
    # when the appropriate dependencies are installed.
    df.write_parquet(config.output_path, partition_by="month")


@asset(group_name="nged")
def nged_live_primary_data_west_midlands(context: AssetExecutionContext) -> Output[pl.DataFrame]:
    """Live primary data for West Midlands from NGED."""
    client = NGEDCKANClient()
    df = download_live_primary_data(client, "live-primary-data---west-midlands")
    SubstationFlows.validate(df)
    context.log.info(f"Downloaded and validated {len(df)} rows of data")
    return Output(df)


@asset(group_name="nged")
def nged_live_primary_data_east_midlands(context: AssetExecutionContext) -> Output[pl.DataFrame]:
    """Live primary data for East Midlands from NGED."""
    client = NGEDCKANClient()
    df = download_live_primary_data(client, "live-primary-data---east-midlands")
    SubstationFlows.validate(df)
    context.log.info(f"Downloaded and validated {len(df)} rows of data")
    return Output(df)
