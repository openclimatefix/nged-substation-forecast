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
    SimplifiedSubstationFlows,
    SubstationFeatures,
    SubstationFlows,
    SubstationMetadata,
)
from contracts.settings import Settings

from xgboost_forecaster.features import add_temporal_features, add_weather_features
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


def load_substation_power(
    substation_number: int, config: DataConfig | None = None
) -> pt.DataFrame[SimplifiedSubstationFlows]:
    """Load, validate and downsample power data for a single substation."""
    config = config or DataConfig()

    df = cast(
        pt.DataFrame[SubstationFlows],
        pl.scan_delta(config.base_power_path)
        .filter(pl.col("substation_number") == substation_number)
        .collect(),
    )

    df = SubstationFlows.to_simplified_substation_flows(df)

    # Downsample to target resolution (period ending)
    df = (
        df.sort("timestamp")
        .group_by_dynamic("timestamp", every=config.resolution, closed="right", label="right")
        .agg(pl.col("MW_or_MVA").mean())
    )

    return cast(pt.DataFrame[SimplifiedSubstationFlows], df)


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


def join_features(
    power: pt.DataFrame[SimplifiedSubstationFlows],
    weather: pt.DataFrame[ProcessedNwp],
    substation_number: int,
    use_lags: bool = True,
    is_training: bool = False,
) -> pt.DataFrame[SubstationFeatures]:
    """Join power and weather data and add all features.

    Args:
        power: Simplified substation power data.
        weather: Processed weather data.
        substation_number: The substation number.
        use_lags: Whether to add power lags.
        is_training: Whether we are in training mode (drops rows with missing actuals).

    Returns:
        A Patito DataFrame containing the joined data and features.

    Raises:
        RuntimeError: If the join results in an empty DataFrame.
    """
    # Start with weather as the base (defines the valid_time range)
    df = weather.clone()

    if use_lags:
        # Add power lags
        power_lag_7d = power.select(
            [
                (pl.col("timestamp") + timedelta(days=7)).alias("valid_time"),
                pl.col("MW_or_MVA").alias("power_lag_7d"),
            ]
        )
        power_lag_14d = power.select(
            [
                (pl.col("timestamp") + timedelta(days=14)).alias("valid_time"),
                pl.col("MW_or_MVA").alias("power_lag_14d"),
            ]
        )
        df = df.join(power_lag_7d, on="valid_time", how="left").join(
            power_lag_14d, on="valid_time", how="left"
        )

    # Join with current power (the target)
    power_target = power.select(
        [
            pl.col("timestamp").alias("valid_time"),
            pl.col("MW_or_MVA"),
        ]
    )
    df = df.join(power_target, on="valid_time", how="left")

    # Add features
    df = add_weather_features(df)
    df = add_temporal_features(df)

    if is_training:
        # During training, we must have the target variable and all features
        df = df.drop_nulls()
    else:
        # During inference, we fill the target with 0.0 to satisfy the contract
        # But we still expect all features to be present (non-null)
        df = df.with_columns(MW_or_MVA=pl.col("MW_or_MVA").fill_null(0.0))

    if df.is_empty():
        raise RuntimeError(
            f"Join resulted in empty DataFrame for substation {substation_number}. "
            "Check if weather and power data overlap."
        )

    df = df.with_columns(
        substation_number=pl.lit(substation_number).cast(pl.Int32),
        MW_or_MVA=pl.col("MW_or_MVA").cast(pl.Float32),
    )

    if not is_training:
        # Check for nulls in features (except MW_or_MVA which we just filled)
        feature_cols = [c for c in df.columns if c != "MW_or_MVA"]
        null_counts = df.select([pl.col(c).is_null().sum().alias(c) for c in feature_cols])
        total_nulls = null_counts.sum_horizontal().item()
        if total_nulls > 0:
            # Find which columns have nulls
            null_cols = [c for c in feature_cols if null_counts[c][0] > 0]
            raise RuntimeError(
                f"Inference data for substation {substation_number} contains null features "
                f"in columns: {null_cols}. This usually indicates a data gap in historical "
                "power or weather data."
            )

    # Final cleanup and validation
    return SubstationFeatures.validate(df, drop_superfluous_columns=True)


def prepare_training_data(
    substation_numbers: list[int],
    metadata: pt.DataFrame[SubstationMetadata],
    start_date: date,
    end_date: date,
    config: DataConfig | None = None,
    selection: EnsembleSelection = EnsembleSelection.MEAN,
    use_lags: bool = True,
) -> pt.DataFrame[SubstationFeatures]:
    """Prepare training data for multiple substations.

    Args:
        substation_numbers: List of substation numbers.
        metadata: Substation metadata.
        start_date: Start date for weather data.
        end_date: End date for weather data.
        config: Data configuration.
        selection: Ensemble selection method.
        use_lags: Whether to add power lags.

    Returns:
        A Patito DataFrame containing the training data for all substations.
    """
    config = config or DataConfig()

    # 1. Extract all unique h3_indices
    relevant_metadata = metadata.filter(pl.col("substation_number").is_in(substation_numbers))
    h3_indices = relevant_metadata["h3_res_5"].unique().to_list()

    # 2. Load historical weather once for all indices
    log.info(f"Loading historical weather for {len(h3_indices)} H3 indices...")
    raw_weather = construct_historical_weather(start_date, end_date, h3_indices, config)

    # 3. Process weather once
    log.info("Processing weather data...")
    processed_weather = process_weather_data(raw_weather, selection, config)

    # 4. Loop over substations and join
    all_subs_data = []
    for sub_num in substation_numbers:
        log.info(f"Preparing data for substation {sub_num}...")
        try:
            sub_meta = relevant_metadata.filter(pl.col("substation_number") == sub_num)
            if sub_meta.is_empty():
                log.warning(f"Substation {sub_num} not found in metadata. Skipping.")
                continue

            h3_idx = sub_meta["h3_res_5"][0]
            power = load_substation_power(sub_num, config)

            # Filter processed weather for this substation's H3 index
            sub_weather = processed_weather.filter(pl.col("h3_index") == h3_idx)
            print(
                f"DEBUG: Substation {sub_num}: h3_idx={h3_idx}, sub_weather shape={sub_weather.shape}"
            )

            sub_data = join_features(power, sub_weather, sub_num, use_lags, is_training=True)

            all_subs_data.append(sub_data)
        except Exception as e:
            log.error(f"Failed to prepare data for substation {sub_num}: {e}")
            raise

    if not all_subs_data:
        raise RuntimeError("No data prepared for any substation")

    return cast(pt.DataFrame[SubstationFeatures], pl.concat(all_subs_data, how="diagonal"))


def prepare_inference_data(
    substation_number: int,
    init_time: datetime,
    metadata: pt.DataFrame[SubstationMetadata],
    config: DataConfig | None = None,
    use_lags: bool = True,
) -> pt.DataFrame[SubstationFeatures]:
    """Prepare data for inference for a single substation.

    Args:
        substation_number: The substation number.
        init_time: The NWP initialization time to use.
        metadata: Substation metadata.
        config: Data configuration.
        use_lags: Whether to add power lags.

    Returns:
        A Patito DataFrame containing the inference data.
    """
    config = config or DataConfig()

    sub_meta = metadata.filter(pl.col("substation_number") == substation_number)
    if sub_meta.is_empty():
        raise RuntimeError(f"Substation {substation_number} not found in metadata")

    h3_idx = sub_meta["h3_res_5"][0]

    # 1. Load NWP run
    raw_weather = load_nwp_run(init_time, [h3_idx], config)

    # 2. Process weather (use ALL for probabilistic inference)
    processed_weather = process_weather_data(raw_weather, EnsembleSelection.ALL, config)

    # 3. Load power data (for lags)
    power = load_substation_power(substation_number, config)

    # 4. Join and add features
    return join_features(power, processed_weather, substation_number, use_lags, is_training=False)
