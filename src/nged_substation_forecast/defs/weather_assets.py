from datetime import datetime, timezone

import dagster as dg
import polars as pl
from contracts.settings import Settings
from dagster import (
    AssetCheckExecutionContext,
    AssetCheckResult,
    AssetCheckSeverity,
    AssetExecutionContext,
    DailyPartitionsDefinition,
    ResourceParam,
    asset,
    asset_check,
    define_asset_job,
)
from dynamical_data.processing import download_and_scale_ecmwf

weather_partitions = DailyPartitionsDefinition(start_date="2024-04-01", end_offset=1)


# The `pool="ECMWF"` works in conjunction with `concurrent.pools.default_limit` in
# $DAGSTER_HOME/dagster.yaml to limit the number of times this asset can be run concurrently.
# The ECMWF download script uses a lot of RAM, so it's best to run it one-by-one.
# See: https://docs.dagster.io/guides/operate/managing-concurrency/concurrency-pools
@asset(partitions_def=weather_partitions, pool="ECMWF")
def ecmwf_ens_forecast(context: AssetExecutionContext, settings: ResourceParam[Settings]) -> None:
    """Download and process ECMWF ENS forecast for Great Britain."""
    partition_key = context.partition_key
    nwp_init_time = datetime.strptime(partition_key, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    context.log.info(f"Downloading ECMWF ENS for {partition_key}")
    scaled_df = download_and_scale_ecmwf(nwp_init_time)

    output_dir = settings.nwp_data_path / "ECMWF" / "ENS"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{nwp_init_time.strftime('%Y-%m-%dT%H')}Z.parquet"
    output_path = output_dir / filename

    scaled_df.write_parquet(output_path, compression="zstd", compression_level=14)

    context.log.info(f"Saved {len(scaled_df)} rows to {output_path}")


@asset_check(asset=ecmwf_ens_forecast)
def check_ecmwf_historical_bounds(
    context: AssetCheckExecutionContext, settings: ResourceParam[Settings]
) -> AssetCheckResult:
    """Check if any weather variables hit the absolute historical bounds (0 or 255)."""
    partition_key = context.partition_key
    nwp_init_time = datetime.strptime(partition_key, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Locate the parquet file for this partition
    filename = f"{nwp_init_time.strftime('%Y-%m-%dT%H')}Z.parquet"
    filepath = settings.nwp_data_path / "ECMWF" / "ENS" / filename

    if not filepath.exists():
        return AssetCheckResult(passed=False, description="Parquet file not found.")

    # Lazily scan the parquet file
    lf = pl.scan_parquet(filepath)

    # Find all UInt8 columns
    uint8_cols = [
        name
        for name, dtype in zip(lf.collect_schema().names(), lf.collect_schema().dtypes())
        if dtype == pl.UInt8
    ]

    # Count how many values hit 0 or 255 in a single optimized pass
    exprs = [((pl.col(col) == 0) | (pl.col(col) == 255)).sum().alias(col) for col in uint8_cols]

    import typing

    boundary_counts = typing.cast(pl.DataFrame, lf.select(exprs).collect()).to_dicts()[0]

    # Filter to only columns that actually hit the boundaries
    hit_boundaries = {col: count for col, count in boundary_counts.items() if count > 0}

    if hit_boundaries:
        return AssetCheckResult(
            passed=True,  # We still want the pipeline to succeed, just warn us!
            severity=AssetCheckSeverity.WARN,
            description="Extreme weather event detected: Values hit historical min/max bounds.",
            metadata={"boundary_hits": hit_boundaries},
        )

    return AssetCheckResult(passed=True, description="All values within historical bounds.")


update_ecmwf_ens_forecast = define_asset_job(
    name="update_ecmwf_ens_forecast",
    selection=[ecmwf_ens_forecast],
    executor_def=dg.in_process_executor,
)
