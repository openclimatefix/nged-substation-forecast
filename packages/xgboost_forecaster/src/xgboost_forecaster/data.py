"""Data loading and preprocessing for XGBoost forecasting."""

import dataclasses
import logging
import math
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
    # Resolution 5 is the fixed standard for this model as it balances spatial
    # precision with feature dimensionality for the XGBoost model.
    h3_res: int = 5
    resolution: str = "30m"


def get_substation_metadata(config: DataConfig | None = None) -> pt.DataFrame[SubstationMetadata]:
    """Load substation metadata and filter for those with available power data."""
    config = config or DataConfig()
    metadata_path = _SETTINGS.nged_data_path / "parquet" / "substation_metadata.parquet"
    metadata_df = SubstationMetadata.validate(pl.read_parquet(metadata_path))

    # Only return substations we have local power data for in Delta Lake.
    # We use `scan_delta` to perform a lazy, optimized scan of the Delta Lake
    # table, which is more memory-efficient than `read_delta` for large tables.
    substations_with_telemetry = (
        cast(
            pl.DataFrame,
            pl.scan_delta(str(config.base_power_path))
            .select("substation_number")
            .unique()
            .collect(),
        )
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
    # The filename format (`YYYY-MM-DDTHHZ.parquet`) is a strict contract with
    # the upstream data pipeline.
    filename = f"{init_time.strftime('%Y-%m-%dT%H')}Z.parquet"
    file_path = config.base_weather_path / filename

    try:
        df = pl.read_parquet(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"NWP file not found at {file_path}. Expected format: YYYY-MM-DDTHHZ.parquet"
        )

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
    # We expect NWP files to follow the `YYYY-MM-DDTHHZ.parquet` naming contract.
    # We use a robust parsing mechanism to ignore files that do not match this
    # format, preventing crashes from unexpected files in the directory.
    files = sorted(config.base_weather_path.glob("*.parquet"))
    relevant_files = []
    for f in files:
        try:
            file_date = datetime.strptime(f.stem[:10], "%Y-%m-%d").date()
            if start_date <= file_date <= end_date:
                relevant_files.append(f)
        except ValueError:
            # Skip files that don't match the expected date format
            continue

    if not relevant_files:
        raise RuntimeError(f"No NWP files found between {start_date} and {end_date}")

    weather_dfs = []
    for f in relevant_files:
        df = pl.read_parquet(f)
        df = df.filter(pl.col("h3_index").is_in(h3_indices))

        if not df.is_empty():
            weather_dfs.append(df)

    if not weather_dfs:
        raise RuntimeError(f"No weather data found for H3 indices in range {start_date}-{end_date}")

    # Combine all forecasts to retain a distribution of lead times
    combined = pl.concat(weather_dfs, how="diagonal")
    combined = combined.sort(["h3_index", "ensemble_member", "valid_time", "init_time"])

    return Nwp.validate(combined)


def process_nwp_data(
    nwp: pl.LazyFrame,
    h3_indices: list[int],
) -> pl.LazyFrame:
    """Process NWP data: lead-time filtering and 30m interpolation for all members.

    Note: Accumulated variables (e.g., precipitation, radiation) are already
    de-accumulated by Dynamical.org prior to download, and should not be differenced.

    Warning: This function collects the input LazyFrame into memory to perform
    upsampling and interpolation. For large historical datasets, ensure the input
    is partitioned or filtered to avoid out-of-memory errors.

    Args:
        nwp: Raw NWP data.
        h3_indices: List of H3 indices to filter for.

    Returns:
        Processed NWP data.
    """
    # 1. Filter by H3 indices to reduce data size
    lf = nwp.filter(pl.col("h3_index").is_in(h3_indices))

    # 2. Calculate Lead Time and Filter (Fixing Leakage)
    # We cap the lead time at 336 hours (14 days) because ECMWF ENS
    # reliability drops significantly after day 14, and the model is only
    # validated for a 14-day horizon.
    lf = (
        lf.with_columns(
            lead_time_hours=(pl.col("valid_time") - pl.col("init_time")).dt.total_minutes() / 60.0
        )
        .with_columns(pl.col("lead_time_hours").cast(pl.Float32))
        .filter(pl.col("lead_time_hours") <= 336)
    )

    # 3. Interpolation (Fixing Nulls)
    # Vectorized approach using upsample and interpolate
    # Add defensive check before collection
    row_count = cast(pl.DataFrame, lf.select(pl.len()).collect()).item()
    if row_count > 5_000_000:
        log.warning(
            f"Eagerly collecting a large NWP dataset ({row_count} rows) for interpolation. "
            "This may cause OOM errors."
        )

    df = cast(pl.DataFrame, lf.collect())

    if df.is_empty():
        return lf.limit(0)

    # Ensure each group has at least two points for interpolation.
    # Groups with only 1 row cannot be interpolated and would violate the
    # 30-minute temporal resolution contract.
    group_counts = df.group_by(["init_time", "h3_index", "ensemble_member"]).len()
    total_groups = group_counts.height
    single_row_groups = group_counts.filter(pl.col("len") == 1)

    if single_row_groups.height > 0:
        dropped_pct = (single_row_groups.height / total_groups) * 100 if total_groups > 0 else 0
        log.warning(
            f"Dropping {single_row_groups.height} groups ({dropped_pct:.2f}%) with only 1 row as they cannot be interpolated."
        )
        df = df.filter(pl.len().over(["init_time", "h3_index", "ensemble_member"]) > 1)

    if df.is_empty():
        return lf.limit(0)

    # Sort by valid_time as required by upsample
    df = df.sort("valid_time")

    # Upsample to 30m and interpolate for each H3 index, ensemble member, and init_time
    processed = df.upsample(
        time_column="valid_time",
        every="30m",
        group_by=["init_time", "h3_index", "ensemble_member"],
    ).sort(["init_time", "h3_index", "ensemble_member", "valid_time"])

    # Interpolate all numeric columns, but forward-fill categorical ones.
    categorical_cols = [
        c for c in ["categorical_precipitation_type_surface"] if c in processed.columns
    ]

    numeric_cols = [
        col
        for col, dtype in processed.schema.items()
        if dtype.is_numeric()
        and col
        not in [
            "valid_time",
            "h3_index",
            "ensemble_member",
            "init_time",
            "lead_time_hours",
        ]
        and col not in categorical_cols
    ]

    # TEMPORAL INTERPOLATION & LEAKAGE:
    # Interpolating over `valid_time` within a single `init_time` is NOT data leakage.
    # All `valid_time` predictions in a single forecast run are generated simultaneously
    # at `init_time`. We are not looking into the future of when the forecast was made,
    # but merely interpolating the forecast's own future predictions to a higher
    # temporal resolution (30m).
    #
    # WIND VECTOR INTERPOLATION:
    # We interpolate Cartesian components (u, v) linearly, which is physically
    # realistic and avoids phantom high winds during direction shifts.
    #
    # RADIATION INTERPOLATION CAVEAT:
    # Linear interpolation for solar radiation (`downward_short_wave_radiation_flux_surface`)
    # between 3-hourly NWP points will "cut the corners" of the diurnal solar
    # cycle, potentially underestimating peak solar generation. It is used as a
    # baseline, and future iterations could use a clear-sky model to better
    # preserve the diurnal cycle.
    processed = processed.with_columns(
        [
            pl.col(c).interpolate().over(["init_time", "h3_index", "ensemble_member"])
            for c in numeric_cols
        ]
        + [
            # CATEGORICAL FORWARD-FILL:
            # Linear interpolation is physically meaningless for categorical variables.
            # For example, a value of 1.5 between 'rain' (1) and 'snow' (2) has no
            # physical interpretation. We use forward-fill to maintain the discrete
            # state of the weather condition until the next forecast step.
            pl.col(c).forward_fill().over(["init_time", "h3_index", "ensemble_member"])
            for c in categorical_cols
        ]
    )

    # PHYSICAL WIND CALCULATION:
    # After interpolating U and V components, we calculate physical wind speed
    # and direction. This ensures the circular topology of wind direction is
    # preserved without needing complex circular interpolation logic.
    wind_cols = ["wind_u_10m", "wind_v_10m", "wind_u_100m", "wind_v_100m"]
    if all(c in processed.columns for c in wind_cols):
        processed = processed.with_columns(
            [
                (pl.col("wind_u_10m") ** 2 + pl.col("wind_v_10m") ** 2)
                .sqrt()
                .alias("wind_speed_10m"),
                ((pl.arctan2("wind_u_10m", "wind_v_10m") * 180 / math.pi + 180) % 360).alias(
                    "wind_direction_10m"
                ),
                (pl.col("wind_u_100m") ** 2 + pl.col("wind_v_100m") ** 2)
                .sqrt()
                .alias("wind_speed_100m"),
                ((pl.arctan2("wind_u_100m", "wind_v_100m") * 180 / math.pi + 180) % 360).alias(
                    "wind_direction_100m"
                ),
            ]
        ).drop(wind_cols)

    # Recalculate lead_time_hours for the new 30m timestamps
    processed = processed.with_columns(
        lead_time_hours=(
            (pl.col("valid_time") - pl.col("init_time")).dt.total_minutes() / 60.0
        ).cast(pl.Float32)
    )

    # Final cast to Float32 for all physical variables to satisfy data contracts.
    # We exclude H3 index and ensemble member as they are identifiers, and
    # categorical columns which must remain as integers.
    physical_cols = [
        col
        for col, dtype in processed.schema.items()
        if dtype.is_numeric()
        and col not in ["h3_index", "ensemble_member"]
        and col not in categorical_cols
    ]
    processed = processed.with_columns([pl.col(c).cast(pl.Float32) for c in physical_cols])

    return processed.lazy()
