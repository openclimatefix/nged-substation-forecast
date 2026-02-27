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

    # Sanity check: Check if data gets "stuck" (identical values for too long)
    # If we have more than 6 identical consecutive values (30 mins at 5-min res), mark as suspect.
    # We'll just filter out substations that have long streaks of identical values in the training set.
    streaks = df.with_columns(
        is_new_streak=(pl.col("power_mw") != pl.col("power_mw").shift(1)).fill_null(True)
    ).with_columns(streak_id=pl.col("is_new_streak").cum_sum())

    max_streak = streaks.group_by("streak_id").len().select(pl.col("len").max()).item()

    if max_streak > 12:  # 1 hour of identical values
        log.warning(
            f"Substation data in {parquet_filename} looks 'stuck' (max streak: {max_streak})"
        )
        return pl.DataFrame()

    return df


def load_weather_data(
    h3_indices: list[int],
    start_date: str,
    end_date: str,
    init_time: datetime | None = None,
    average_ensembles: bool = True,
    upsample_to_5min: bool = True,
) -> pl.DataFrame:
    """Load weather data for specific H3 cells and date range.

    Args:
        h3_indices: List of H3 cells to load.
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        init_time: If provided, only load this specific initialization time.
        average_ensembles: If True, average across all ensemble members.
        upsample_to_5min: If True, upsample the hourly weather data to 5-minutely.
    """
    # We'll load all files between start and end date
    files = sorted(BASE_WEATHER_PATH.glob("*.parquet"))
    relevant_files = [f for f in files if start_date <= f.stem.split("T")[0] <= end_date]

    if init_time:
        init_str = init_time.strftime("%Y-%m-%dT%H")
        relevant_files = [f for f in relevant_files if f.stem == init_str]

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

    # Variables to keep
    nwp_vars = [
        col
        for col in weather.columns
        if weather[col].dtype in [pl.Float32, pl.Float64]
        and col not in ["timestamp", "h3_index", "lead_time"]
    ]

    group_cols = ["h3_index", "timestamp"]
    if not average_ensembles:
        group_cols.append("ensemble_member")

    if average_ensembles:
        weather = weather.group_by(group_cols).agg([pl.col(c).mean() for c in nwp_vars])
    else:
        # Keep ensemble members separate
        weather = weather.select(group_cols + nwp_vars)

    if upsample_to_5min:
        # Create a full 5-minute grid for the time range
        ts_min = weather["timestamp"].min()
        ts_max = weather["timestamp"].max()
        if ts_min is not None and ts_max is not None:
            time_grid = (
                pl.datetime_range(ts_min, ts_max, interval="5m", time_zone="UTC", eager=True)
                .alias("timestamp")
                .to_frame()
            )

            # We'll join each group to the time grid and interpolate
            # This is more robust than map_groups for multi-column grouping
            group_cols = ["h3_index"]
            if "ensemble_member" in weather.columns:
                group_cols.append("ensemble_member")

            # Get unique groups
            groups = weather.select(group_cols).unique()

            upsampled_parts = []
            for group in groups.iter_rows(named=True):
                group_df = weather.filter(*[pl.col(k) == v for k, v in group.items()]).sort(
                    "timestamp"
                )

                # Join grid and interpolate
                upsampled = time_grid.join(group_df, on="timestamp", how="left")

                # Fill group columns and interpolate numeric vars
                upsampled = upsampled.with_columns(
                    [pl.lit(v).alias(k) for k, v in group.items()]
                ).interpolate()

                upsampled_parts.append(upsampled)

            weather = pl.concat(upsampled_parts)

    return weather


def add_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add cyclical and standard temporal features to a dataframe with a timestamp column."""
    return df.with_columns(
        [
            # Cyclical Hour (24h)
            (pl.col("timestamp").dt.hour() * 2 * 3.14159 / 24).sin().alias("hour_sin"),
            (pl.col("timestamp").dt.hour() * 2 * 3.14159 / 24).cos().alias("hour_cos"),
            # Cyclical Day of Year (365.25d)
            (pl.col("timestamp").dt.ordinal_day() * 2 * 3.14159 / 365.25)
            .sin()
            .alias("day_of_year_sin"),
            (pl.col("timestamp").dt.ordinal_day() * 2 * 3.14159 / 365.25)
            .cos()
            .alias("day_of_year_cos"),
            # Keep day of week as it's useful for work/weekend patterns
            pl.col("timestamp").dt.weekday().alias("day_of_week"),
        ]
    )


def add_weather_features(
    weather: pl.DataFrame, history: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Add lags and trends to weather data.

    If history is provided, lags are calculated using history + weather.
    Otherwise, they are calculated from weather alone (suitable for training).
    """
    if history is not None:
        full_weather = (
            pl.concat([history, weather.select(history.columns)])
            .unique("timestamp")
            .sort("timestamp")
        )
    else:
        full_weather = weather

    # 7d and 14d lags
    weather_lag_7d = full_weather.select(
        [
            (pl.col("timestamp") + timedelta(days=7)).alias("timestamp"),
            pl.col("temperature_2m").alias("temperature_2m_lag_7d"),
            pl.col("downward_short_wave_radiation_flux_surface").alias("sw_radiation_lag_7d"),
        ]
    )
    weather_lag_14d = full_weather.select(
        [
            (pl.col("timestamp") + timedelta(days=14)).alias("timestamp"),
            pl.col("temperature_2m").alias("temperature_2m_lag_14d"),
            pl.col("downward_short_wave_radiation_flux_surface").alias("sw_radiation_lag_14d"),
        ]
    )
    # 6h trend
    weather_trend_6h = full_weather.select(
        [
            (pl.col("timestamp") + timedelta(hours=6)).alias("timestamp"),
            pl.col("temperature_2m").alias("temperature_2m_6h_ago"),
        ]
    )

    # Join features back to the original weather dataframe
    weather = (
        weather.join(weather_lag_7d, on="timestamp", how="left")
        .join(weather_lag_14d, on="timestamp", how="left")
        .join(weather_trend_6h, on="timestamp", how="left")
    )

    return weather.with_columns(
        temp_trend_6h=(pl.col("temperature_2m") - pl.col("temperature_2m_6h_ago"))
    )


def prepare_training_data(
    substation_names: list[str], metadata: pl.DataFrame, use_lags: bool = True
) -> pl.DataFrame:
    """Join power and weather data for multiple substations and add features.

    Args:
        substation_names: List of substation names.
        metadata: Substation metadata.
        use_lags: If True, adds 7-day and 14-day lagged power features.
    """
    all_subs_data = []

    for sub_name in substation_names:
        sub_meta = metadata.filter(pl.col("substation_name_in_location_table") == sub_name)
        if sub_meta.is_empty():
            log.warning(f"Substation {sub_name} not found in metadata")
            continue

        h3_index = sub_meta["h3_index"][0]
        sub_id = sub_meta["substation_number"][0]
        parquet_file = sub_meta["parquet_filename"][0]

        power = load_substation_power(parquet_file)
        if power.is_empty():
            continue

        if use_lags:
            # Robust time-based lags: join power with itself at offsets
            power_lag_7d = power.select(
                [
                    (pl.col("timestamp") + timedelta(days=7)).alias("timestamp"),
                    pl.col("power_mw").alias("power_mw_lag_7d"),
                ]
            )
            power_lag_14d = power.select(
                [
                    (pl.col("timestamp") + timedelta(days=14)).alias("timestamp"),
                    pl.col("power_mw").alias("power_mw_lag_14d"),
                ]
            )
            power = power.join(power_lag_7d, on="timestamp", how="left").join(
                power_lag_14d, on="timestamp", how="left"
            )

        power_min = cast(datetime, power["timestamp"].min())
        power_max = cast(datetime, power["timestamp"].max())

        if power_min is None or power_max is None:
            continue

        end_date = power_max.strftime("%Y-%m-%d")
        weather_start = (power_min - timedelta(days=2)).strftime("%Y-%m-%d")

        weather = load_weather_data(
            [h3_index],
            weather_start,
            end_date,
            average_ensembles=True,
            upsample_to_5min=True,
        )

        if weather.is_empty():
            continue

        # Add weather lags and trends
        weather = add_weather_features(weather)

        # Join on timestamp
        sub_data = power.join(weather, on="timestamp", how="inner")

        if not sub_data.is_empty():
            # Add substation ID and temporal features
            sub_data = sub_data.with_columns(pl.lit(sub_id).alias("substation_id").cast(pl.Int32))
            sub_data = add_temporal_features(sub_data)
            all_subs_data.append(sub_data)

    if not all_subs_data:
        return pl.DataFrame()

    return pl.concat(all_subs_data, how="diagonal")
