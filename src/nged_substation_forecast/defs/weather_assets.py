from datetime import datetime, timezone

from pydantic import Field

from typing import cast

import dagster as dg
import patito as pt
import polars as pl
from contracts.data_schemas import H3GridWeights, Nwp
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
from xgboost_forecaster.data import process_nwp_data

weather_partitions = DailyPartitionsDefinition(start_date="2024-04-01", end_offset=1)


# The `pool="ECMWF"` works in conjunction with `concurrent.pools.default_limit` in
# $DAGSTER_HOME/dagster.yaml to limit the number of times this asset can be run concurrently.
# The ECMWF download script uses a lot of RAM, so it's best to run it one-by-one.
# See: https://docs.dagster.io/guides/operate/managing-concurrency/concurrency-pools
@asset(partitions_def=weather_partitions, pool="ECMWF")
def ecmwf_ens_forecast(
    context: AssetExecutionContext,
    settings: ResourceParam[Settings],
    gb_h3_grid_weights: pl.DataFrame,
) -> None:
    """Download and process ECMWF ENS forecast for Great Britain."""
    partition_key = context.partition_key
    nwp_init_time = datetime.strptime(partition_key, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    context.log.info(f"Downloading ECMWF ENS for {partition_key}")
    scaled_df = download_and_scale_ecmwf(
        nwp_init_time, h3_grid=cast(pt.DataFrame[H3GridWeights], gb_h3_grid_weights)
    )

    output_dir = settings.nwp_data_path / "ECMWF" / "ENS"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{nwp_init_time.strftime('%Y-%m-%dT%H')}Z.parquet"
    output_path = output_dir / filename

    scaled_df = scaled_df.unique(subset=["time", "latitude", "longitude", "ensemble_member"])
    scaled_df.write_parquet(output_path, compression="zstd", compression_level=3)

    context.log.info(f"Saved {len(scaled_df)} rows to {output_path}")


@asset(deps=[ecmwf_ens_forecast])
def all_nwp_data(settings: ResourceParam[Settings]) -> pl.LazyFrame:
    """Provides a LazyFrame scanning all downloaded NWP data.

    The returned LazyFrame adheres to the contracts.data_schemas.Nwp schema.
    """
    nwp_dir = settings.nwp_data_path / "ECMWF" / "ENS"
    if not nwp_dir.exists():
        # Return an empty LazyFrame with the expected schema if the directory doesn't exist.
        # This prevents the pipeline from crashing before any data has been downloaded.
        return cast(pt.LazyFrame[Nwp], pl.LazyFrame(schema=Nwp.dtypes))

    return cast(pt.LazyFrame[Nwp], pl.scan_parquet(nwp_dir / "*.parquet"))


class ProcessedNWPConfig(dg.Config):
    """Configuration for the processed NWP data asset."""

    substation_ids: list[int] | None = Field(
        default=None, description="Optional list of substation IDs to include."
    )
    start_date: str | None = Field(
        default=None, description="Optional start date for filtering NWP data (YYYY-MM-DD)."
    )
    end_date: str | None = Field(
        default=None, description="Optional end date for filtering NWP data (YYYY-MM-DD)."
    )


@asset(
    ins={
        "all_nwp_data": AssetIn("all_nwp_data"),
    }
)
def processed_nwp_data(
    context: AssetExecutionContext,
    config: ProcessedNWPConfig,
    all_nwp_data: pl.LazyFrame,
    settings: ResourceParam[Settings],
) -> pl.LazyFrame:
    """Process NWP data: lead-time filtering and 30m interpolation for all members.

    WARNING: The 30m interpolation step can be memory-intensive and may cause OOM errors
    if the input NWP data or the number of substations is very large.
    """
    context.log.info(f"processed_nwp_data config: {config}")
    context.log.info(f"all_nwp_data schema: {all_nwp_data.collect_schema().names()}")
    time_series_metadata_path = settings.nged_data_path / "parquet" / "time_series_metadata.parquet"
    metadata = pl.read_parquet(time_series_metadata_path)

    if config.start_date:
        all_nwp_data = all_nwp_data.filter(
            pl.col("init_time")
            >= datetime.fromisoformat(config.start_date).replace(tzinfo=timezone.utc)
        )
    if config.end_date:
        all_nwp_data = all_nwp_data.filter(
            pl.col("init_time")
            <= datetime.fromisoformat(config.end_date).replace(
                tzinfo=timezone.utc, hour=23, minute=59, second=59
            )
        )

    if config.substation_ids:
        metadata = metadata.filter(pl.col("time_series_id").is_in(config.substation_ids))
    h3_indices = metadata["h3_res_5"].unique().to_list()
    return process_nwp_data(
        all_nwp_data,
        h3_indices,
    )


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

    # Find all UInt8 columns, excluding categorical variables which are
    # expected to hit 0 (e.g., "no precipitation").
    categorical_vars = ["categorical_precipitation_type_surface"]
    uint8_cols = [
        name
        for name, dtype in zip(lf.collect_schema().names(), lf.collect_schema().dtypes())
        if dtype == pl.UInt8 and name not in categorical_vars
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


@asset_check(asset=ecmwf_ens_forecast)
def check_ecmwf_categorical_bounds(
    context: AssetCheckExecutionContext, settings: ResourceParam[Settings]
) -> AssetCheckResult:
    """Check if categorical precipitation type falls within the expected range (0-8)."""
    partition_key = context.partition_key
    nwp_init_time = datetime.strptime(partition_key, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Locate the parquet file for this partition
    filename = f"{nwp_init_time.strftime('%Y-%m-%dT%H')}Z.parquet"
    filepath = settings.nwp_data_path / "ECMWF" / "ENS" / filename

    if not filepath.exists():
        return AssetCheckResult(passed=False, description="Parquet file not found.")

    # Lazily scan the parquet file
    lf = pl.scan_parquet(filepath)

    # Check if the column exists
    if "categorical_precipitation_type_surface" not in lf.collect_schema().names():
        return AssetCheckResult(
            passed=True, description="Categorical precipitation column not found."
        )

    # Count values outside [0, 8]
    import typing

    invalid_count = typing.cast(
        pl.DataFrame,
        lf.filter(
            (pl.col("categorical_precipitation_type_surface") < 0)
            | (pl.col("categorical_precipitation_type_surface") > 8)
        )
        .select(pl.len())
        .collect(),
    ).item()

    if invalid_count > 0:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            description=f"Found {invalid_count} invalid categorical precipitation values (outside 0-8).",
            metadata={"invalid_count": invalid_count},
        )

    return AssetCheckResult(passed=True, description="All categorical values within range (0-8).")


update_ecmwf_ens_forecast = define_asset_job(
    name="update_ecmwf_ens_forecast",
    selection=[ecmwf_ens_forecast],
    executor_def=dg.in_process_executor,
)
