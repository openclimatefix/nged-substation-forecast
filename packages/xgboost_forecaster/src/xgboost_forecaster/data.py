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

        if not df.is_empty():
            weather_dfs.append(df)

    if not weather_dfs:
        raise RuntimeError(f"No weather data found for H3 indices in range {start_date}-{end_date}")

    # Combine all forecasts to retain a distribution of lead times
    combined = pl.concat(weather_dfs, how="diagonal")
    combined = combined.sort(["h3_index", "ensemble_member", "valid_time", "init_time"])

    return Nwp.validate(combined)


def process_nwp_data(nwp: pl.LazyFrame, h3_indices: list[int]) -> pl.LazyFrame:
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
    # We strictly exclude lead_time < 3 because NWPs have a 3-hour publication delay.
    # This prevents the model from learning from weather data that would not be available in real-time.
    lf = (
        lf.with_columns(
            lead_time_hours=(pl.col("valid_time") - pl.col("init_time")).dt.total_minutes() / 60.0
        )
        .with_columns(pl.col("lead_time_hours").cast(pl.Float32))
        # We cap the lead time at 336 hours (14 days) because ECMWF ENS
        # reliability drops significantly after day 14, and the model is only
        # validated for a 14-day horizon.
        .filter((pl.col("lead_time_hours") >= 3) & (pl.col("lead_time_hours") <= 336))
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

    # Sort by valid_time as required by upsample
    df = df.sort("valid_time")

    # Upsample to 30m and interpolate for each H3 index, ensemble member, and init_time
    processed = df.upsample(
        time_column="valid_time",
        every="30m",
        group_by=["init_time", "h3_index", "ensemble_member"],
    ).sort(["init_time", "h3_index", "ensemble_member", "valid_time"])

    # Interpolate all numeric columns, but forward-fill categorical ones.
    # Circular variables (wind direction) require sine/cosine decomposition
    # to interpolate correctly across the 0/360 boundary.
    categorical_cols = [
        c for c in ["categorical_precipitation_type_surface"] if c in processed.columns
    ]
    circular_cols = [
        c for c in ["wind_direction_10m", "wind_direction_100m"] if c in processed.columns
    ]

    # Decompose circular variables into sine and cosine components
    for col in circular_cols:
        # CIRCULAR INTERPOLATION ASSUMPTION (FLAW-2):
        # We assume the 0-255 UInt8 scale maps linearly to 0-360 degrees (0-2pi radians).
        # Note: 255 is treated as 2pi, which is equivalent to 0 degrees. This mapping
        # allows us to use sine/cosine decomposition to interpolate correctly across
        # the 0/360 boundary.
        processed = processed.with_columns(
            [
                (pl.col(col).cast(pl.Float32) / 255.0 * 2 * math.pi).sin().alias(f"{col}_sin"),
                (pl.col(col).cast(pl.Float32) / 255.0 * 2 * math.pi).cos().alias(f"{col}_cos"),
            ]
        )

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
        and col not in circular_cols
    ]

    # TEMPORAL INTERPOLATION & LEAKAGE (FLAW-3):
    # Interpolating over `valid_time` within a single `init_time` is NOT data leakage.
    # All `valid_time` predictions in a single forecast run are generated simultaneously
    # at `init_time`. We are not looking into the future of when the forecast was made,
    # but merely interpolating the forecast's own future predictions to a higher
    # temporal resolution (30m).
    #
    # RADIATION INTERPOLATION CAVEAT (FLAW-4):
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
            # CATEGORICAL FORWARD-FILL (FLAW-1):
            # Linear interpolation is physically meaningless for categorical variables.
            # For example, a value of 1.5 between 'rain' (1) and 'snow' (2) has no
            # physical interpretation. We use forward-fill to maintain the discrete
            # state of the weather condition until the next forecast step.
            pl.col(c).forward_fill().over(["init_time", "h3_index", "ensemble_member"])
            for c in categorical_cols
        ]
    )

    # Reconstruct circular variables from interpolated sine and cosine components
    for col in circular_cols:
        # arctan2 returns values in [-pi, pi].
        # We add 2pi and take modulo 2pi to get [0, 2pi].
        # Then map back to the 0-255 scale.
        processed = processed.with_columns(
            (
                (pl.arctan2(f"{col}_sin", f"{col}_cos") + 2 * math.pi)
                % (2 * math.pi)
                / (2 * math.pi)
                * 255.0
            )
            .cast(pl.Float32)
            .alias(col)
        )

    # Drop temporary sine and cosine columns
    processed = processed.drop(
        [f"{col}_{suffix}" for col in circular_cols for suffix in ["sin", "cos"]]
    )

    # Recalculate lead_time_hours for the new 30m timestamps
    processed = processed.with_columns(
        lead_time_hours=(
            (pl.col("valid_time") - pl.col("init_time")).dt.total_minutes() / 60.0
        ).cast(pl.Float32)
    )

    return processed.lazy()
