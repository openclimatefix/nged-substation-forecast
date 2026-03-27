"""Data loading and preprocessing for XGBoost forecasting."""

import dataclasses
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import cast

import patito as pt
import polars as pl
from contracts.data_schemas import (
    Nwp,
    ProcessedNwp,
    SubstationMetadata,
)
from contracts.settings import Settings

from xgboost_forecaster.types import EnsembleSelection

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

        # Stitching logic: keep only short lead times (e.g. <= 24h) to form a continuous series
        # This avoids using long-range forecasts for historical training data.
        df = df.filter((pl.col("valid_time") - pl.col("init_time")) <= timedelta(hours=24))

        if not df.is_empty():
            weather_dfs.append(df)

    if not weather_dfs:
        raise RuntimeError(f"No weather data found for H3 indices in range {start_date}-{end_date}")

    # Combine and keep the latest forecast for each valid_time/h3_index/ensemble_member
    combined = pl.concat(weather_dfs, how="diagonal")
    combined = combined.sort(["h3_index", "ensemble_member", "valid_time", "init_time"])
    combined = combined.unique(subset=["h3_index", "ensemble_member", "valid_time"], keep="last")

    return Nwp.validate(combined)


def process_weather_data(
    weather: pt.DataFrame[Nwp], selection: EnsembleSelection, config: DataConfig | None = None
) -> pt.DataFrame[ProcessedNwp]:
    """Process raw NWP data: ensemble selection, grouping, and interpolation.

    Args:
        weather: Raw NWP data.
        selection: Ensemble selection method.
        config: Data configuration.

    Returns:
        A Patito DataFrame containing the processed weather data.
    """
    config = config or DataConfig()

    # Filter out lead time 0 (where valid_time == init_time)
    # because accumulated variables are null at lead time 0.
    weather = weather.filter(pl.col("valid_time") > pl.col("init_time"))

    # Calculate target timestamp: use valid_time
    weather = weather.with_columns(valid_time=pl.col("valid_time").cast(pl.Datetime("us", "UTC")))

    # Variables to keep (all numeric ones except metadata)
    nwp_vars = [
        col
        for col in weather.columns
        if weather[col].dtype in [pl.Float32, pl.Float64, pl.UInt8]
        and col not in ["valid_time", "h3_index", "lead_time", "init_time", "ensemble_member"]
    ]

    group_cols = ["h3_index", "valid_time"]
    if selection == EnsembleSelection.ALL:
        group_cols.append("ensemble_member")

    # Ensemble selection
    if selection == EnsembleSelection.SINGLE:
        weather_df = weather.filter(pl.col("ensemble_member") == 0)
    elif selection == EnsembleSelection.MEAN:
        weather_df = weather.group_by(group_cols).agg([pl.col(c).mean() for c in nwp_vars])
    else:
        weather_df = cast(pl.DataFrame, weather)

    # Resample and Interpolate to match target resolution
    ts_min, ts_max = weather_df.select(
        min=pl.col("valid_time").min(), max=pl.col("valid_time").max()
    ).row(0)

    if ts_min is None or ts_max is None:
        raise RuntimeError("No timestamps found in weather data for interpolation")

    time_grid = (
        pl.datetime_range(ts_min, ts_max, interval=config.resolution, time_zone="UTC", eager=True)
        .alias("valid_time")
        .to_frame()
    )

    group_cols_only_idx = ["h3_index"]
    if selection == EnsembleSelection.ALL:
        group_cols_only_idx.append("ensemble_member")

    groups = weather_df.select(group_cols_only_idx).unique()

    upsampled_parts = []
    for group in groups.iter_rows(named=True):
        group_df = weather_df.filter([pl.col(k) == v for k, v in group.items()]).sort("valid_time")
        upsampled = time_grid.join(group_df, on="valid_time", how="left")
        # Interpolate only the weather variables, not the group columns
        upsampled = upsampled.with_columns([pl.col(c).interpolate() for c in nwp_vars])
        # Fill in the group columns
        upsampled = upsampled.with_columns(
            [pl.lit(v, dtype=weather_df.schema[k]).alias(k) for k, v in group.items()]
        )
        upsampled_parts.append(upsampled)

    weather_df = pl.concat(upsampled_parts)

    # Ensure Float32 for memory efficiency and contract compliance
    # But keep h3_index as UInt64 and ensemble_member as UInt8 if present
    cast_exprs = [pl.col(c).cast(pl.Float32) for c in nwp_vars] + [
        pl.col("h3_index").cast(pl.UInt64)
    ]
    if "ensemble_member" in weather_df.columns:
        cast_exprs.append(pl.col("ensemble_member").cast(pl.UInt8))

    weather_df = weather_df.with_columns(cast_exprs)

    return ProcessedNwp.validate(weather_df, drop_superfluous_columns=True)
