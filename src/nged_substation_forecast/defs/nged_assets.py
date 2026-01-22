"""Dagster assets for NGED data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dagster import AssetExecutionContext, Output, asset
from data_nged.ckan_client import NGEDCKANClient
from data_nged.live_primary_data import download_live_primary_data

if TYPE_CHECKING:
    import polars as pl


@asset(group_name="nged")
def nged_live_primary_data_south_wales(context: AssetExecutionContext) -> Output[pl.DataFrame]:
    """Live primary data for South Wales from NGED."""
    client = NGEDCKANClient()
    df = download_live_primary_data(client, "live-primary-data---south-wales")
    context.log.info(f"Downloaded {len(df)} rows of data")
    return Output(df)


@asset(group_name="nged")
def nged_live_primary_data_south_west(context: AssetExecutionContext) -> Output[pl.DataFrame]:
    """Live primary data for South West from NGED."""
    client = NGEDCKANClient()
    df = download_live_primary_data(client, "live-primary-data---south-west")
    context.log.info(f"Downloaded {len(df)} rows of data")
    return Output(df)


@asset(group_name="nged")
def nged_live_primary_data_west_midlands(context: AssetExecutionContext) -> Output[pl.DataFrame]:
    """Live primary data for West Midlands from NGED."""
    client = NGEDCKANClient()
    df = download_live_primary_data(client, "live-primary-data---west-midlands")
    context.log.info(f"Downloaded {len(df)} rows of data")
    return Output(df)


@asset(group_name="nged")
def nged_live_primary_data_east_midlands(context: AssetExecutionContext) -> Output[pl.DataFrame]:
    """Live primary data for East Midlands from NGED."""
    client = NGEDCKANClient()
    df = download_live_primary_data(client, "live-primary-data---east-midlands")
    context.log.info(f"Downloaded {len(df)} rows of data")
    return Output(df)
