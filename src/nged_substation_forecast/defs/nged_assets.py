"""Dagster assets for NGED data."""

from collections.abc import Iterable

import polars as pl
from contracts.data_schemas import SubstationFlows
from dagster import (
    AssetCheckResult,
    AssetCheckSpec,
    AssetExecutionContext,
    AssetObservation,
    Config,
    MetadataValue,
    Output,
    asset,
)
from data_nged.ckan_client import NGEDCKANClient
from data_nged.live_primary_data import SubstationDownloadResult, download_live_primary_data


class NgedLivePrimaryDataConfig(Config):
    """Configuration for NGED live primary data."""

    output_path: str


def _download_and_validate_region(
    context: AssetExecutionContext, package_id: str
) -> Iterable[AssetObservation | AssetCheckResult | Output[pl.DataFrame]]:
    """Download data for a region and yield outputs, observations and checks."""
    client = NGEDCKANClient()

    dfs: list[pl.DataFrame] = []
    all_errors: dict[str, list[str]] = {}  # maps from substation_name to list of error strings.
    total_count: int = 0

    try:
        substation_results: Iterable[SubstationDownloadResult] = download_live_primary_data(
            client, package_id
        )

        for substation_result in substation_results:
            total_count += 1

            metadata = {
                "substation": substation_result.substation_name,
                "passed": len(substation_result.errors) == 0,
                # The following attributes will be updated later in the code if appropriate:
                "validation_errors": "None",
                "row_count": 0,
                "first_timestamp": "N/A",
                "last_timestamp": "N/A",
            }

            if substation_result.df is not None and not substation_result.df.is_empty():
                dfs.append(substation_result.df)
                metadata["first_timestamp"] = str(substation_result.df["timestamp"].min())
                metadata["last_timestamp"] = str(substation_result.df["timestamp"].max())
                metadata["row_count"] = substation_result.df.height

            if substation_result.errors:
                all_errors[substation_result.substation_name] = substation_result.errors
                metadata["validation_errors"] = MetadataValue.json(substation_result.errors)

            yield AssetObservation(asset_key=context.asset_key, metadata=metadata)
    except Exception as e:
        context.log.error(f"Failed to fetch package {package_id}: {e}")
        all_errors["_package_fetch"] = [str(e)]

    # Yield the overall integrity check for the asset
    yield AssetCheckResult(
        check_name="substation_integrity",
        passed=len(all_errors) == 0,
        metadata={
            "total_substations": total_count,
            "failed_count": len(all_errors),
            "errors_by_substation": MetadataValue.json(all_errors),
        },
    )

    if not dfs:
        context.log.warning(f"No data found for region {package_id}. Yielding empty DataFrame.")
        yield Output(pl.DataFrame(schema=SubstationFlows.dtypes))
    else:
        yield Output(pl.concat(dfs, how="diagonal"))


@asset(
    group_name="nged",
    check_specs=[
        AssetCheckSpec(name="substation_integrity", asset="nged_live_primary_data_south_wales")
    ],
)
def nged_live_primary_data_south_wales(context: AssetExecutionContext):
    """Live primary data for South Wales from NGED."""
    yield from _download_and_validate_region(context, "live-primary-data---south-wales")


@asset(
    group_name="nged",
    check_specs=[
        AssetCheckSpec(name="substation_integrity", asset="nged_live_primary_data_south_west")
    ],
)
def nged_live_primary_data_south_west(context: AssetExecutionContext):
    """Live primary data for South West from NGED."""
    yield from _download_and_validate_region(context, "live-primary-data---south-west")


@asset(
    group_name="nged",
    check_specs=[
        AssetCheckSpec(name="substation_integrity", asset="nged_live_primary_data_west_midlands")
    ],
)
def nged_live_primary_data_west_midlands(context: AssetExecutionContext):
    """Live primary data for West Midlands from NGED."""
    yield from _download_and_validate_region(context, "live-primary-data---west-midlands")


@asset(
    group_name="nged",
    check_specs=[
        AssetCheckSpec(name="substation_integrity", asset="nged_live_primary_data_east_midlands")
    ],
)
def nged_live_primary_data_east_midlands(context: AssetExecutionContext):
    """Live primary data for East Midlands from NGED."""
    yield from _download_and_validate_region(context, "live-primary-data---east-midlands")


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
    df.write_parquet(config.output_path, partition_by="month")
