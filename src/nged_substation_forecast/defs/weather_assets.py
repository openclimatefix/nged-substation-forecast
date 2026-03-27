from datetime import datetime, timezone
from typing import cast

import dagster as dg
import polars as pl
from contracts.settings import Settings
from dagster import (
    AssetCheckExecutionContext,
    AssetCheckResult,
    AssetCheckSeverity,
    AssetExecutionContext,
    AssetIn,
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


@asset(deps=[ecmwf_ens_forecast])
def all_nwp_data(settings: ResourceParam[Settings]) -> pl.LazyFrame:
    """Provides a LazyFrame scanning all downloaded NWP data."""
    return pl.scan_parquet(settings.nwp_data_path / "ECMWF" / "ENS" / "*.parquet")


@asset(
    ins={
        "all_nwp_data": AssetIn("all_nwp_data"),
        "substation_metadata": AssetIn("substation_metadata"),
    }
)
def processed_nwp_data(
    all_nwp_data: pl.LazyFrame, substation_metadata: pl.DataFrame
) -> pl.LazyFrame:
    """Process NWP data: lead-time filtering and 30m interpolation for all members."""
    # 1. Filter by H3 indices to reduce data size
    h3_indices = substation_metadata["h3_res_5"].unique().to_list()
    lf = all_nwp_data.filter(pl.col("h3_index").is_in(h3_indices))

    # 2. Calculate Lead Time and Filter (Fixing Leakage)
    # We strictly exclude lead_time == 0 because accumulated variables are null there.
    # This also prevents the model from learning from "perfect" 0-hour forecasts.
    lf = lf.with_columns(
        lead_time_hours=(pl.col("valid_time") - pl.col("init_time"))
        .dt.total_hours()
        .cast(pl.Float32)
    ).filter((pl.col("lead_time_hours") > 0) & (pl.col("lead_time_hours") <= 336))

    # 3. Interpolation (Fixing Nulls)
    # Since we've reduced the data size, we can collect and interpolate.
    df = cast(pl.DataFrame, lf.collect())

    # Variables to interpolate (all numeric ones except metadata)
    nwp_vars = [
        col
        for col in df.columns
        if col not in ["valid_time", "h3_index", "lead_time", "init_time", "ensemble_member"]
    ]

    # Upsample to 30m and interpolate for each H3 index, ensemble member, and init_time
    # We group by all three to ensure we interpolate within a single forecast trajectory.
    groups = df.select(["h3_index", "ensemble_member", "init_time"]).unique()
    upsampled_parts = []
    for group in groups.iter_rows(named=True):
        h3 = group["h3_index"]
        ens = group["ensemble_member"]
        init = group["init_time"]
        group_df = df.filter(
            (pl.col("h3_index") == h3)
            & (pl.col("ensemble_member") == ens)
            & (pl.col("init_time") == init)
        ).sort("valid_time")
        upsampled = group_df.upsample(time_column="valid_time", every="30m")
        # Interpolate only the weather variables
        upsampled = upsampled.with_columns([pl.col(c).interpolate() for c in nwp_vars])
        # Fill in the metadata columns
        upsampled = upsampled.with_columns(
            h3_index=pl.lit(h3, dtype=pl.UInt64),
            ensemble_member=pl.lit(ens, dtype=pl.UInt8),
            init_time=pl.lit(init, dtype=pl.Datetime("us", "UTC")),
        )
        # Recalculate lead_time_hours for the new 30m timestamps
        upsampled = upsampled.with_columns(
            lead_time_hours=(pl.col("valid_time") - pl.col("init_time"))
            .dt.total_hours()
            .cast(pl.Float32)
        )
        upsampled_parts.append(upsampled)

    processed_df = pl.concat(upsampled_parts)

    return processed_df.lazy()


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
