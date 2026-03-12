"""Data loading and preprocessing for XGBoost forecasting."""

import dataclasses
import logging
from datetime import datetime, timedelta
from pathlib import Path

import patito as pt
import polars as pl
from contracts.data_schemas import SubstationFlows, SubstationMetadata
from contracts.settings import Settings

from xgboost_forecaster.features import add_temporal_features, add_weather_features

log = logging.getLogger(__name__)

_SETTINGS = Settings()


@dataclasses.dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    base_power_path: Path = _SETTINGS.nged_data_path / "delta" / "live_primary_flows"
    base_weather_path: Path = _SETTINGS.nwp_data_path / "ECMWF" / "ENS"
    h3_res: int = 5  # TODO: This should probably be stored somewhere like nwp_data_path/ECMWF/ENS/metadata.json?
    resolution: str = "30m"


def get_substation_metadata(config: DataConfig | None = None) -> pl.DataFrame:
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


def load_substation_power(sub_number: int, config: DataConfig | None = None) -> pl.DataFrame:
    """Load, validate and downsample power data for a single substation."""
    config = config or DataConfig()

    df = SubstationFlows.validate(
        pl.scan_delta(config.base_power_path)  # type: ignore[invalid-argument-type]
        .filter(pl.col("substation_number") == sub_number)
        .collect()
    )

    # Ensure standard column names and types
    power_col = SubstationFlows.choose_power_column(df)
    df = df.rename({power_col: "power"})
    df = df.select(["timestamp", "power"]).drop_nulls().sort("timestamp")

    # Downsample to target resolution (period ending)
    df = df.group_by_dynamic(
        "timestamp", every=config.resolution, closed="right", label="right"
    ).agg(pl.col("power").mean())

    return df


def load_weather_data(
    h3_indices: list[int],
    start_date: str,
    end_date: str,
    config: DataConfig | None = None,
    init_time: datetime | None = None,
    average_ensembles: bool = True,
) -> pl.DataFrame:
    """Load weather data for specific H3 cells and date range."""
    config = config or DataConfig()
    # We'll load all files between start and end date
    # TODO(Jack): I'm not sure what's going on here! Why don't we just select based on a single
    # init time, and generating the filename from that init time. I need to look at how
    # `load_weather_data` is called.
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
    ts_min, ts_max = weather.select(
        min=pl.col("timestamp").min(),
        max=pl.col("timestamp").max(),
    ).row(0)
    if ts_min is not None and ts_max is not None:
        time_grid = (
            pl.datetime_range(
                ts_min, ts_max, interval=config.resolution, time_zone="UTC", eager=True
            )
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
    substation_number: int,
    substation_metadata: pt.DataFrame[SubstationMetadata],
    config: DataConfig | None = None,
    use_lags: bool = True,
    weather_ens_member_selection: str = "mean",
) -> pl.DataFrame:
    """Load and join data for a single substation."""
    config = config or DataConfig()

    sub_meta = substation_metadata.filter(pl.col("substation_number") == substation_number)

    if sub_meta.is_empty():
        raise RuntimeError(f"Substation {substation_number} not found in metadata")

    h3_index = sub_meta["h3_res_5"][0]

    power = load_substation_power(substation_number, config)
    if power.is_empty():
        raise RuntimeError(f"No power data for substation {substation_number}!")

    if use_lags:
        # Add power lags
        power_lag_7d = power.select(
            [
                (pl.col("timestamp") + timedelta(days=7)).alias("timestamp"),
                pl.col("power").alias("power_lag_7d"),
            ]
        )
        power_lag_14d = power.select(
            [
                (pl.col("timestamp") + timedelta(days=14)).alias("timestamp"),
                pl.col("power").alias("power_lag_14d"),
            ]
        )
        power = power.join(power_lag_7d, on="timestamp", how="left").join(
            power_lag_14d, on="timestamp", how="left"
        )

    first_timestamp, last_timestamp = power.select(
        min=pl.col("timestamp").min(),
        max=pl.col("timestamp").max(),
    ).row(0)

    if first_timestamp is None or last_timestamp is None:
        raise RuntimeError("first_timestamp or last_timestamp is None!")

    end_date = last_timestamp.strftime("%Y-%m-%d")
    weather_start = (first_timestamp - timedelta(days=2)).strftime("%Y-%m-%d")

    weather = load_weather_data(
        [h3_index],
        weather_start,
        end_date,
        config,
        average_ensembles=(weather_ens_member_selection == "mean"),
    )
    if weather.is_empty():
        return pl.DataFrame()

    if weather_ens_member_selection == "single":
        weather = weather.filter(pl.col("ensemble_member") == 0)

    weather = add_weather_features(weather)
    sub_data = power.join(weather, on="timestamp", how="inner")

    if not sub_data.is_empty():
        sub_data = sub_data.with_columns(
            pl.lit(substation_number).alias("substation_number").cast(pl.Int32),
        )
        sub_data = add_temporal_features(sub_data)

    return sub_data


def prepare_training_data(
    substation_numbers: list[int],
    metadata: pl.DataFrame,
    config: DataConfig | None = None,
    **kwargs,
) -> pl.DataFrame:
    """Join power and weather data for multiple substations and add features."""
    all_subs_data = []
    for sub_number in substation_numbers:
        sub_data = prepare_data_for_substation(sub_number, metadata, config, **kwargs)
        if not sub_data.is_empty():
            all_subs_data.append(sub_data)

    if not all_subs_data:
        return pl.DataFrame()

    return pl.concat(all_subs_data, how="diagonal")
