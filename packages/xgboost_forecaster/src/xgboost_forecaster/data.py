"""Data loading and preprocessing for XGBoost forecasting."""

import logging
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Final, cast

import h3.api.numpy_int as h3
import polars as pl
from nged_data import ckan
from nged_data.substation_names.align import join_location_table_to_live_primaries

log = logging.getLogger(__name__)

# TODO: Configure these paths via a shared config/env.
BASE_POWER_PATH: Final[Path] = Path("data/NGED/parquet/live_primary_flows")
BASE_WEATHER_PATH: Final[Path] = Path("packages/dynamical_data/data")
H3_RES: Final[int] = 5


def get_substation_metadata() -> pl.DataFrame:
    """Join substation locations with their live flow parquet filenames."""
    locations = ckan.get_primary_substation_locations()
    live_primaries = ckan.get_csv_resources_for_live_primary_substation_flows()

    df = join_location_table_to_live_primaries(live_primaries=live_primaries, locations=locations)

    # Add H3 index based on lat/lng
    df = df.with_columns(
        h3_index=pl.struct(["latitude", "longitude"]).map_elements(
            lambda x: h3.latlng_to_cell(x["latitude"], x["longitude"], H3_RES),
            return_dtype=pl.UInt64,
        ),
        parquet_filename=pl.col("url").map_elements(
            lambda url: PurePosixPath(url.path).with_suffix(".parquet").name, return_dtype=pl.String
        ),
    )

    # Only return substations we have local power data for
    df = df.filter(
        pl.col("parquet_filename").map_elements(
            lambda f: (BASE_POWER_PATH / f).exists(), return_dtype=pl.Boolean
        )
    )
    return df


def load_substation_power(parquet_filename: str) -> pl.DataFrame:
    """Load and validate power data for a single substation."""
    path = BASE_POWER_PATH / parquet_filename
    if not path.exists():
        raise FileNotFoundError(f"Power data not found at {path}")

    df = pl.read_parquet(path)
    # Ensure standard column names and types
    power_col = "MW" if "MW" in df.columns else "MVA"
    df = df.select(
        [
            pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
            pl.col(power_col).alias("power_mw").cast(pl.Float32),
        ]
    ).drop_nulls()

    return df


def load_weather_data(h3_indices: list[int], start_date: str, end_date: str) -> pl.DataFrame:
    """Load weather data for specific H3 cells and date range, averaging ensembles."""
    # We'll load all files between start and end date
    files = sorted(BASE_WEATHER_PATH.glob("*.parquet"))
    relevant_files = [f for f in files if start_date <= f.stem.split("T")[0] <= end_date]

    if not relevant_files:
        log.warning(f"No weather files found for range {start_date} to {end_date}")
        return pl.DataFrame()

    weather_dfs = []
    for f in relevant_files:
        df = pl.read_parquet(f)
        df = df.filter(pl.col("h3_index").is_in(h3_indices))
        if not df.is_empty():
            weather_dfs.append(df)

    if not weather_dfs:
        return pl.DataFrame()

    weather = pl.concat(weather_dfs, how="diagonal")

    # Calculate target timestamp: init_time + lead_time
    weather = weather.with_columns(
        timestamp=(pl.col("init_time") + pl.col("lead_time")).cast(pl.Datetime("us", "UTC"))
    )

    # Average across ensemble members
    numeric_cols = [
        col
        for col in weather.columns
        if weather[col].dtype in [pl.Float32, pl.Float64]
        and col not in ["timestamp", "h3_index", "lead_time"]
    ]

    weather = weather.group_by(["h3_index", "timestamp"]).agg(
        [pl.col(c).mean() for c in numeric_cols]
    )

    return weather


def prepare_training_data(substation_name: str, metadata: pl.DataFrame) -> pl.DataFrame:
    """Join power and weather data for a single substation and add features."""
    sub_meta = metadata.filter(pl.col("substation_name_in_location_table") == substation_name)
    if sub_meta.is_empty():
        raise ValueError(f"Substation {substation_name} not found in metadata")

    h3_index = sub_meta["h3_index"][0]
    parquet_file = sub_meta["parquet_filename"][0]

    power = load_substation_power(parquet_file)

    power_min = cast(datetime, power["timestamp"].min())
    power_max = cast(datetime, power["timestamp"].max())

    if power_min is None or power_max is None:
        raise ValueError(f"No power data found for substation {substation_name}")

    end_date = power_max.strftime("%Y-%m-%d")

    # We need weather data that covers the power data period.
    weather_start = (power_min - timedelta(days=2)).strftime("%Y-%m-%d")
    weather = load_weather_data([h3_index], weather_start, end_date)

    if weather.is_empty():
        raise ValueError(f"No weather data found for substation {substation_name} at H3 {h3_index}")

    # Join on timestamp
    data = power.join(weather, on="timestamp", how="inner")

    if data.is_empty():
        log.warning(f"No overlapping data for {substation_name}.")
        return pl.DataFrame()

    # Temporal features
    data = data.with_columns(
        [
            pl.col("timestamp").dt.hour().alias("hour"),
            pl.col("timestamp").dt.weekday().alias("day_of_week"),
            pl.col("timestamp").dt.month().alias("month"),
        ]
    )

    return data
