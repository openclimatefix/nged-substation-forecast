"""Data loading and preprocessing for XGBoost forecasting."""

import dataclasses
import logging
from datetime import date, datetime
from pathlib import Path
from typing import cast

import patito as pt
import polars as pl
from contracts.data_schemas import (
    Nwp,
    SubstationMetadata,
)
from contracts.settings import Settings


log = logging.getLogger(__name__)

_SETTINGS = Settings()


@dataclasses.dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    base_power_path: Path = _SETTINGS.nged_data_path / "delta" / "live_primary_flows"
    base_weather_path: Path = _SETTINGS.nwp_data_path / "ECMWF" / "ENS"
    h3_res: int = 5  # TODO: This should probably be stored somewhere like nwp_data_path/ECMWF/ENS/metadata.json?
    resolution: str = "30m"


def get_substation_metadata(config: DataConfig | None = None) -> pt.DataFrame[SubstationMetadata]:
    """Load substation metadata and filter for those with available power data."""
    config = config or DataConfig()
    metadata_path = _SETTINGS.nged_data_path / "parquet" / "substation_metadata.parquet"
    metadata_df = SubstationMetadata.validate(pl.read_parquet(metadata_path))

    # Only return substations we have local power data for in Delta Lake
    substations_with_telemetry = (
        pl.read_delta(str(config.base_power_path))
        .select("substation_number")
        .unique()
        .to_series()
        .to_list()
    )

    return metadata_df.filter(pl.col("substation_number").is_in(substations_with_telemetry))


def load_nwp_run(
    init_time: datetime, h3_indices: list[int], config: DataConfig | None = None
) -> pt.DataFrame[Nwp]:
    """Load a single NWP forecast run.

    Args:
        init_time: The initialization time of the NWP run.
        h3_indices: List of H3 indices to filter for.
        config: Data configuration.

    Returns:
        A Patito DataFrame containing the NWP data.

    Raises:
        FileNotFoundError: If the NWP file for the given init_time does not exist.
    """
    config = config or DataConfig()
    filename = f"{init_time.strftime('%Y-%m-%dT%H')}Z.parquet"
    file_path = config.base_weather_path / filename

    df = pl.read_parquet(file_path)
    df = df.filter(pl.col("h3_index").is_in(h3_indices))

    if df.is_empty():
        raise RuntimeError(f"No data found for requested H3 indices in {file_path}")

    return Nwp.validate(df)


def construct_historical_weather(
    start_date: date, end_date: date, h3_indices: list[int], config: DataConfig | None = None
) -> pt.DataFrame[Nwp]:
    """Construct a continuous historical weather timeseries by stitching NWP runs.

    Args:
        start_date: Start date for the timeseries.
        end_date: End date for the timeseries.
        h3_indices: List of H3 indices to filter for.
        config: Data configuration.

    Returns:
        A Patito DataFrame containing the stitched NWP data.

    Raises:
        RuntimeError: If no NWP files are found in the date range.
    """
    config = config or DataConfig()
    files = sorted(config.base_weather_path.glob("*.parquet"))
    relevant_files = [
        f
        for f in files
        if start_date <= datetime.strptime(f.stem[:10], "%Y-%m-%d").date() <= end_date
    ]

    if not relevant_files:
        raise RuntimeError(f"No NWP files found between {start_date} and {end_date}")

    weather_dfs = []
    for f in relevant_files:
        df = pl.read_parquet(f)
        df = df.filter(pl.col("h3_index").is_in(h3_indices))

        if df.is_empty():
            continue

        if not df.is_empty():
            weather_dfs.append(df)

    if not weather_dfs:
        raise RuntimeError(f"No weather data found for H3 indices in range {start_date}-{end_date}")

    # Combine all forecasts to retain a distribution of lead times
    combined = pl.concat(weather_dfs, how="diagonal")
    combined = combined.sort(["h3_index", "ensemble_member", "valid_time", "init_time"])

    return Nwp.validate(combined)


def downsample_power_flows(flows: pl.LazyFrame) -> pl.LazyFrame:
    """Downsample power flows to 30m using period-ending semantics.

    Args:
        flows: Historical power flow data.

    Returns:
        Downsampled power flows.
    """
    return (
        flows.sort("timestamp")
        .group_by_dynamic(
            "timestamp",
            every="30m",
            group_by="substation_number",
            closed="right",
            label="right",
        )
        .agg(
            [
                pl.col("MW").mean(),
                pl.col("MVA").mean(),
            ]
        )
        .with_columns(
            pl.when(pl.col("MW").is_not_null())
            .then(pl.col("MW"))
            .otherwise(pl.col("MVA"))
            .alias("MW_or_MVA")
        )
    )


def process_nwp_data(nwp: pl.LazyFrame, h3_indices: list[int]) -> pl.LazyFrame:
    """Process NWP data: lead-time filtering and 30m interpolation for all members.

    Args:
        nwp: Raw NWP data.
        h3_indices: List of H3 indices to filter for.

    Returns:
        Processed NWP data.
    """
    # 1. Filter by H3 indices to reduce data size
    lf = nwp.filter(pl.col("h3_index").is_in(h3_indices))

    # 2. Calculate Lead Time and Filter (Fixing Leakage)
    # We strictly exclude lead_time == 0 because accumulated variables are null there.
    # This also prevents the model from learning from "perfect" 0-hour forecasts.
    lf = (
        lf.with_columns(
            lead_time_hours=(pl.col("valid_time") - pl.col("init_time")).dt.total_minutes() / 60.0
        )
        .with_columns(pl.col("lead_time_hours").cast(pl.Float32))
        .filter((pl.col("lead_time_hours") > 0) & (pl.col("lead_time_hours") <= 336))
    )

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
            lead_time_hours=(
                (pl.col("valid_time") - pl.col("init_time")).dt.total_minutes() / 60.0
            ).cast(pl.Float32)
        )
        upsampled_parts.append(upsampled)

    if not upsampled_parts:
        return pl.DataFrame(schema=df.schema).lazy()

    processed_df = pl.concat(upsampled_parts)

    return processed_df.lazy()
