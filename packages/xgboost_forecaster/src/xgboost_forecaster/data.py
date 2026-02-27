"""Data loading and preprocessing for XGBoost forecasting."""

import logging
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Final, cast

import h3.api.numpy_int as h3
import polars as pl
from nged_data import ckan
from nged_data.substation_names.align import join_location_table_to_live_primaries
from xgboost_forecaster.scaling import get_scaling_expressions, load_scaling_params

log = logging.getLogger(__name__)

# TODO: Configure these paths via a shared config/env.
BASE_POWER_PATH: Final[Path] = Path("data/NGED/parquet/live_primary_flows")
BASE_WEATHER_PATH: Final[Path] = Path("packages/dynamical_data/data")
H3_RES: Final[int] = 5
RESOLUTION: Final[str] = "30m"


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
    """Load, validate and downsample power data for a single substation."""
    path = BASE_POWER_PATH / parquet_filename
    if not path.exists():
        raise FileNotFoundError(f"Power data not found at {path}")

    df = pl.read_parquet(path)
    # Ensure standard column names and types
    power_col = "MW" if "MW" in df.columns else "MVA"
    df = (
        df.select(
            [
                pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
                pl.col(power_col).alias("power_mw").cast(pl.Float32),
            ]
        )
        .drop_nulls()
        .sort("timestamp")
    )

    if df.is_empty():
        return df

    # Sanity check: Check if data gets "stuck" (identical values for too long)
    streaks = df.with_columns(
        is_new_streak=(pl.col("power_mw") != pl.col("power_mw").shift(1)).fill_null(True)
    ).with_columns(streak_id=pl.col("is_new_streak").cum_sum())

    max_streak = streaks.group_by("streak_id").len().select(pl.col("len").max()).item()

    if max_streak > 24:  # 2 hours of identical values (at 5min resolution)
        log.warning(
            f"Substation data in {parquet_filename} looks 'stuck' (max streak: {max_streak})"
        )
        return pl.DataFrame()

    # Downsample to 30-minutely (period ending)
    df = df.group_by_dynamic("timestamp", every="30m", closed="right", label="right").agg(
        pl.col("power_mw").mean()
    )

    return df


def load_weather_data(
    h3_indices: list[int],
    start_date: str,
    end_date: str,
    init_time: datetime | None = None,
    average_ensembles: bool = True,
    resample_to: str = RESOLUTION,
    scale_to_uint8: bool = True,
) -> pl.DataFrame:
    """Load weather data for specific H3 cells and date range."""
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
        if weather[col].dtype in [pl.Float32, pl.Float64, pl.UInt8]
        and col not in ["timestamp", "h3_index", "lead_time", "init_time", "ensemble_member"]
    ]

    group_cols = ["h3_index", "timestamp"]
    if not average_ensembles:
        group_cols.append("ensemble_member")

    # We MUST group by the target columns to avoid duplicate rows for the same timestamp
    weather = weather.group_by(group_cols).agg([pl.col(c).mean() for c in nwp_vars])

    # Resample and Interpolate to match target resolution
    ts_min = weather["timestamp"].min()
    ts_max = weather["timestamp"].max()
    if ts_min is not None and ts_max is not None:
        time_grid = (
            pl.datetime_range(ts_min, ts_max, interval=resample_to, time_zone="UTC", eager=True)
            .alias("timestamp")
            .to_frame()
        )

        group_cols_only_idx = ["h3_index"]
        if "ensemble_member" in weather.columns:
            group_cols_only_idx.append("ensemble_member")

        groups = weather.select(group_cols_only_idx).unique()

        upsampled_parts = []
        for group in groups.iter_rows(named=True):
            group_df = weather.filter(*[pl.col(k) == v for k, v in group.items()]).sort("timestamp")
            upsampled = time_grid.join(group_df, on="timestamp", how="left")
            upsampled = upsampled.with_columns(
                [pl.lit(v).alias(k) for k, v in group.items()]
            ).interpolate()
            upsampled_parts.append(upsampled)

        weather = pl.concat(upsampled_parts)

    if scale_to_uint8:
        scaling_params = load_scaling_params()
        available_vars = set(scaling_params["col_name"].to_list())
        vars_to_scale = [v for v in nwp_vars if v in available_vars]
        scaling_exprs = get_scaling_expressions(
            scaling_params.filter(pl.col("col_name").is_in(vars_to_scale))
        )
        weather = weather.with_columns(scaling_exprs)

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
    """Add lags and trends to weather data."""
    if history is not None:
        full_weather = (
            pl.concat(
                [
                    history.select(pl.all().exclude("ensemble_member")),
                    weather.select(pl.all().exclude("ensemble_member")),
                ],
                how="diagonal",
            )
            .unique("timestamp")
            .sort("timestamp")
        )
    else:
        full_weather = weather

    def get_lagged_view(df: pl.DataFrame, offset: timedelta, suffix: str) -> pl.DataFrame:
        lagged = df.select(
            [
                (pl.col("timestamp") + offset).alias("timestamp"),
                pl.col("temperature_2m").alias(f"temperature_2m_{suffix}"),
                pl.col("downward_short_wave_radiation_flux_surface").alias(
                    f"sw_radiation_{suffix}"
                ),
            ]
            + ([pl.col("ensemble_member")] if "ensemble_member" in df.columns else [])
        )
        return lagged

    weather_lag_7d = get_lagged_view(full_weather, timedelta(days=7), "lag_7d")
    weather_lag_14d = get_lagged_view(full_weather, timedelta(days=14), "lag_14d")
    weather_trend_6h = full_weather.select(
        [
            (pl.col("timestamp") + timedelta(hours=6)).alias("timestamp"),
            pl.col("temperature_2m").alias("temperature_2m_6h_ago"),
        ]
        + ([pl.col("ensemble_member")] if "ensemble_member" in full_weather.columns else [])
    )

    join_on = ["timestamp"]
    if "ensemble_member" in weather.columns and "ensemble_member" in weather_lag_7d.columns:
        join_on.append("ensemble_member")

    weather = (
        weather.join(weather_lag_7d, on=join_on, how="left")
        .join(weather_lag_14d, on=join_on, how="left")
        .join(weather_trend_6h, on=join_on, how="left")
    )

    return weather.with_columns(
        temp_trend_6h=(
            pl.col("temperature_2m").cast(pl.Int16) - pl.col("temperature_2m_6h_ago").cast(pl.Int16)
        )
    )


def prepare_training_data(
    substation_names: list[str],
    metadata: pl.DataFrame,
    use_lags: bool = True,
    member_selection: str = "mean",
) -> pl.DataFrame:
    """Join power and weather data for multiple substations and add features."""
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
            average_ensembles=(member_selection == "mean"),
            scale_to_uint8=True,
        )

        if weather.is_empty():
            continue

        if member_selection == "single":
            weather = weather.filter(pl.col("ensemble_member") == 0)

        weather = add_weather_features(weather)
        sub_data = power.join(weather, on="timestamp", how="inner")

        if not sub_data.is_empty():
            sub_data = sub_data.with_columns(pl.lit(sub_id).alias("substation_id").cast(pl.Int32))
            sub_data = add_temporal_features(sub_data)
            all_subs_data.append(sub_data)

    if not all_subs_data:
        return pl.DataFrame()

    return pl.concat(all_subs_data, how="diagonal")
