"""Data loading and preprocessing for XGBoost forecasting."""

import dataclasses
import logging
import math
from datetime import date, datetime
from pathlib import Path

import patito as pt
import polars as pl
from contracts.data_schemas import (
    Nwp,
)
from contracts.settings import Settings
from ml_core.scaling import uint8_to_physical_unit
from xgboost_forecaster.scaling import load_scaling_params


log = logging.getLogger(__name__)

_SETTINGS = Settings()


@dataclasses.dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    base_power_path: Path = _SETTINGS.nged_data_path / "delta" / "cleaned_power_time_series"
    base_weather_path: Path = _SETTINGS.nwp_data_path / "ECMWF" / "ENS"
    # Resolution 5 is the fixed standard for this model as it balances spatial
    # precision with feature dimensionality for the XGBoost model.
    h3_res: int = 5
    resolution: str = "30m"


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

    # Descale data immediately to physical units (Float32)
    params = load_scaling_params()
    scaling_cols = params.select("col_name").to_series().to_list()

    # Only descale columns that are actually UInt8 to prevent double-descaling
    schema = df.schema
    uint8_cols = [col for col, dtype in schema.items() if dtype == pl.UInt8 and col in scaling_cols]

    if uint8_cols:
        descale_exprs = uint8_to_physical_unit(params.filter(pl.col("col_name").is_in(uint8_cols)))
        df = df.with_columns(descale_exprs)

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

    # Descale data immediately to physical units (Float32)
    params = load_scaling_params()
    scaling_cols = params.select("col_name").to_series().to_list()

    # Only descale columns that are actually UInt8 to prevent double-descaling
    schema = combined.schema
    uint8_cols = [col for col, dtype in schema.items() if dtype == pl.UInt8 and col in scaling_cols]

    if uint8_cols:
        descale_exprs = uint8_to_physical_unit(params.filter(pl.col("col_name").is_in(uint8_cols)))
        combined = combined.with_columns(descale_exprs)

    return Nwp.validate(combined)


def process_nwp_data(
    nwp: pl.LazyFrame,
    h3_indices: list[int],
) -> pl.LazyFrame:
    """Process NWP data: lead-time filtering and 30m interpolation for all members.

    Note: Accumulated variables (e.g., precipitation, radiation) are already
    de-accumulated by Dynamical.org prior to download, and should not be differenced.

    Args:
        nwp: Raw NWP data.
        h3_indices: List of H3 indices to filter for.

    Returns:
        Processed NWP data.
    """
    # 1. Filter by H3 indices to reduce data size
    lf = nwp.filter(pl.col("h3_index").is_in(h3_indices))

    # Descale data immediately to physical units (Float32) before any interpolation
    # or feature engineering. This ensures that interpolation happens in the
    # physical space and prevents mixing scaled (UInt8) and unscaled variables
    # in downstream calculations (like windchill or wind speed).
    params = load_scaling_params()
    scaling_cols = params.select("col_name").to_series().to_list()

    # Only descale columns that are actually UInt8 to prevent double-descaling
    # (e.g., if the function is called with already-descaled data in tests).
    schema = lf.collect_schema()
    uint8_cols = [col for col, dtype in schema.items() if dtype == pl.UInt8 and col in scaling_cols]

    if uint8_cols:
        descale_exprs = uint8_to_physical_unit(params.filter(pl.col("col_name").is_in(uint8_cols)))
        lf = lf.with_columns(descale_exprs)

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
    # Ensure each group has at least two points for interpolation.
    # Groups with only 1 row cannot be interpolated and would violate the
    # 30-minute temporal resolution contract.
    lf = lf.filter(pl.len().over(["init_time", "h3_index", "ensemble_member"]) > 1)

    # Create a complete 30-minute time grid for each (init_time, h3_index, ensemble_member).
    # This replaces the eager `upsample` with a native lazy operation.
    grid = (
        lf.select(["init_time", "h3_index", "ensemble_member", "valid_time"])
        .group_by(["init_time", "h3_index", "ensemble_member"])
        .agg(
            [
                pl.col("valid_time").min().alias("start"),
                pl.col("valid_time").max().alias("end"),
            ]
        )
        .with_columns(valid_time=pl.datetime_ranges("start", "end", interval="30m"))
        .explode("valid_time")
        .drop(["start", "end"])
    )

    # Join the grid with the original data to create gaps for interpolation
    lf = grid.join(lf, on=["init_time", "h3_index", "ensemble_member", "valid_time"], how="left")

    # Identify columns for interpolation and forward-fill
    categorical_cols = ["categorical_precipitation_type_surface"]
    schema = lf.collect_schema()
    exclude_cols = ["valid_time", "h3_index", "ensemble_member", "init_time", "lead_time_hours"]
    numeric_cols = [
        col
        for col, dtype in schema.items()
        if dtype.is_numeric() and col not in exclude_cols and col not in categorical_cols
    ]

    # TEMPORAL INTERPOLATION & LEAKAGE:
    # Interpolating over `valid_time` within a single `init_time` is NOT data leakage.
    # All `valid_time` predictions in a single forecast run are generated simultaneously
    # at `init_time`. We are not looking into the future of when the forecast was made,
    # but merely interpolating the forecast's own future predictions to a higher
    # temporal resolution (30m).
    lf = lf.with_columns(
        [
            pl.col(c)
            .interpolate()
            .forward_fill()
            .backward_fill()
            .over(["init_time", "h3_index", "ensemble_member"])
            for c in numeric_cols
        ]
        + [
            # CATEGORICAL FORWARD-FILL:
            # Linear interpolation is physically meaningless for categorical variables.
            # We use forward-fill to maintain the discrete state of the weather condition.
            pl.col(c).forward_fill().over(["init_time", "h3_index", "ensemble_member"])
            for c in categorical_cols
            if c in schema.names()
        ]
    )

    # PHYSICAL WIND CALCULATION (Lazy):
    # After interpolating U and V components, we calculate physical wind speed
    # and direction. This ensures the circular topology of wind direction is
    # preserved without needing complex circular interpolation logic.
    wind_cols = ["wind_u_10m", "wind_v_10m", "wind_u_100m", "wind_v_100m"]
    if all(c in lf.collect_schema().names() for c in wind_cols):
        lf = lf.with_columns(
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
    lf = lf.with_columns(
        lead_time_hours=(
            (pl.col("valid_time") - pl.col("init_time")).dt.total_minutes() / 60.0
        ).cast(pl.Float32)
    )

    # Final cast to Float32 for all physical variables to satisfy data contracts.
    # We exclude H3 index and ensemble member as they are identifiers, and
    # categorical columns which must remain as integers.
    schema = lf.collect_schema()
    physical_cols = [
        col
        for col, dtype in schema.items()
        if dtype.is_numeric()
        and col not in ["h3_index", "ensemble_member"]
        and col not in categorical_cols
    ]
    lf = lf.with_columns([pl.col(c).cast(pl.Float32) for c in physical_cols])

    return lf
