"""Data loading and preprocessing for XGBoost forecasting."""

import dataclasses
import logging
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import cast

import polars as pl
from contracts.data_schemas import SubstationLocations
from contracts.settings import Settings
from nged_data import ckan
from nged_data.substation_names.align import join_location_table_to_live_primaries

from xgboost_forecaster.features import add_temporal_features, add_weather_features

log = logging.getLogger(__name__)

_SETTINGS = Settings()


@dataclasses.dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    base_power_path: Path = _SETTINGS.nged_data_path / "delta" / "live_primary_flows"
    base_weather_path: Path = _SETTINGS.nwp_data_path / "ECMWF" / "ENS"
    # TODO: Remove `ckan_token` after teaching Dagster to download substation locations
    ckan_token: str = _SETTINGS.nged_ckan_token
    h3_res: int = 5  # TODO: This should probably be stored somewhere like nwp_data_path/ECMWF/ENS/metadata.json?
    resolution: str = "30m"


# TODO: Most of (or maybe all of) this function should be computed by Dagster once.
def get_substation_metadata(config: DataConfig | None = None) -> pl.DataFrame:
    """Join substation locations with their live flow parquet filenames."""
    config = config or DataConfig()
    api_key = config.ckan_token
    locations_path = _SETTINGS.nged_data_path / "parquet" / "substation_locations.parquet"
    locations = SubstationLocations.validate(pl.read_parquet(locations_path))
    live_primaries = ckan.get_csv_resources_for_live_primary_substation_flows(api_key=api_key)

    df = join_location_table_to_live_primaries(live_primaries=live_primaries, locations=locations)

    # Add H3 index based on lat/lng
    df = df.rename({"h3_res_5": "h3_index"})
    df = df.with_columns(
        parquet_filename=pl.col("url").map_elements(
            lambda url: PurePosixPath(url.path).with_suffix(".parquet").name, return_dtype=pl.String
        ),
    )

    # Only return substations we have local power data for
    try:
        available_subs = (
            pl.read_delta(str(config.base_power_path))
            .select("substation_name")
            .unique()
            .to_series()
            .to_list()
        )

        df = df.filter(pl.col("parquet_filename").str.replace(".parquet", "").is_in(available_subs))
    except Exception:
        # If delta table doesn't exist or has an issue, return empty
        df = df.filter(pl.lit(False))

    return df


def load_substation_power(sub_name: str, config: DataConfig | None = None) -> pl.DataFrame:
    """Load, validate and downsample power data for a single substation."""
    config = config or DataConfig()

    delta_path = str(config.base_power_path)
    try:
        df = pl.read_delta(
            delta_path, pyarrow_options={"filters": [("substation_name", "=", sub_name)]}
        )
    except Exception:
        return pl.DataFrame()

    if df.is_empty():
        return pl.DataFrame()

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

    # Downsample to target resolution (period ending)
    df = df.group_by_dynamic(
        "timestamp", every=config.resolution, closed="right", label="right"
    ).agg(pl.col("power_mw").mean())

    return df


def load_weather_data(
    h3_indices: list[int],
    start_date: str,
    end_date: str,
    config: DataConfig | None = None,
    init_time: datetime | None = None,
    average_ensembles: bool = True,
    resample_to: str = "30m",
) -> pl.DataFrame:
    """Load weather data for specific H3 cells and date range."""
    config = config or DataConfig()
    # We'll load all files between start and end date
    files = sorted(config.base_weather_path.glob("*.parquet"))
    relevant_files = [f for f in files if start_date <= f.stem.split("T")[0] <= end_date]

    if init_time:
        init_str = init_time.strftime("%Y-%m-%dT%H")
        relevant_files = [
            f for f in relevant_files if f.stem == init_str or f.stem == f"{init_str}Z"
        ]

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

    # Calculate target timestamp: use valid_time
    weather = weather.with_columns(timestamp=pl.col("valid_time").cast(pl.Datetime("us", "UTC")))

    # Variables to keep
    nwp_vars = [
        col
        for col in weather.columns
        if weather[col].dtype in [pl.Float32, pl.Float64, pl.UInt8]
        and col
        not in ["timestamp", "h3_index", "lead_time", "init_time", "ensemble_member", "valid_time"]
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

    return weather


def prepare_data_for_substation(
    sub_name: str,
    metadata: pl.DataFrame,
    config: DataConfig | None = None,
    use_lags: bool = True,
    member_selection: str = "mean",
) -> pl.DataFrame:
    """Load and join data for a single substation."""
    config = config or DataConfig()

    # Check if we should filter by substation_name_in_location_table or the stem of parquet_filename
    # Since sub_name is usually the stem of parquet_filename (e.g. 'abbeywood-primary-transformer-flows')
    if "substation_name" in metadata.columns:
        sub_meta = metadata.filter(pl.col("substation_name") == sub_name)
    else:
        sub_meta = metadata.filter(
            pl.col("parquet_filename").str.replace(".parquet", "") == sub_name
        )

    if sub_meta.is_empty():
        log.warning(f"Substation {sub_name} not found in metadata")
        return pl.DataFrame()

    h3_index = sub_meta["h3_index"][0]
    sub_id = sub_meta["substation_number"][0]

    power = load_substation_power(sub_name, config)
    if power.is_empty():
        return pl.DataFrame()

    if use_lags:
        # Add power lags
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
        return pl.DataFrame()

    end_date = power_max.strftime("%Y-%m-%d")
    weather_start = (power_min - timedelta(days=2)).strftime("%Y-%m-%d")

    weather = load_weather_data(
        [h3_index],
        weather_start,
        end_date,
        config,
        average_ensembles=(member_selection == "mean"),
    )
    if weather.is_empty():
        return pl.DataFrame()

    if member_selection == "single":
        weather = weather.filter(pl.col("ensemble_member") == 0)

    weather = add_weather_features(weather)
    sub_data = power.join(weather, on="timestamp", how="inner")

    if not sub_data.is_empty():
        sub_data = sub_data.with_columns(
            pl.lit(sub_id).alias("substation_id").cast(pl.Int32),
            pl.lit(sub_name).alias("substation_name"),
        )
        sub_data = add_temporal_features(sub_data)

    return sub_data


def prepare_training_data(
    substation_names: list[str],
    metadata: pl.DataFrame,
    config: DataConfig | None = None,
    **kwargs,
) -> pl.DataFrame:
    """Join power and weather data for multiple substations and add features."""
    all_subs_data = []
    for sub_name in substation_names:
        sub_data = prepare_data_for_substation(sub_name, metadata, config, **kwargs)
        if not sub_data.is_empty():
            all_subs_data.append(sub_data)

    if not all_subs_data:
        return pl.DataFrame()

    return pl.concat(all_subs_data, how="diagonal")
