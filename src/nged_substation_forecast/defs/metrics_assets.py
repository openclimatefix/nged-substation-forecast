import logging
from datetime import timedelta
from typing import cast

import dagster as dg
import polars as pl
from dagster import ResourceParam

from contracts.settings import Settings
from xgboost_forecaster import DataConfig, get_substation_metadata
from .partitions import model_partitions
from .xgb_assets import XGBoostConfig, _apply_config_overrides, load_hydra_config

log = logging.getLogger(__name__)


@dg.asset(deps=["live_primary_flows"])
def combined_actuals(
    context: dg.AssetExecutionContext, settings: ResourceParam[Settings]
) -> pl.LazyFrame:
    """Combines all live primary flows into a single lazy dataframe."""
    actuals_path = settings.nged_data_path / "delta" / "live_primary_flows"
    if not actuals_path.exists():
        return pl.LazyFrame()

    df = pl.scan_delta(str(actuals_path))

    # Get metadata to map filenames to substation IDs
    data_config = DataConfig(
        base_power_path=actuals_path,
        base_weather_path=settings.nwp_data_path / "ECMWF" / "ENS",
    )
    metadata = get_substation_metadata(data_config)

    # Join metadata to get substation_number
    df = df.join(metadata.select(["substation_number"]).lazy(), on="substation_number", how="inner")

    # Some actuals might have 'MW', others 'MVA'.
    df = df.with_columns(MW_or_MVA=pl.coalesce(["MW", "MVA"]))

    return df


@dg.asset_check(asset=combined_actuals)
def check_all_zeros(
    context: dg.AssetCheckExecutionContext, combined_actuals: pl.LazyFrame
) -> dg.AssetCheckResult:
    """Check if any substations have all zero values."""
    if combined_actuals.collect_schema().names() == []:
        return dg.AssetCheckResult(passed=True, description="No data to check.")

    # First, query the maximum timestamp from combined_actuals to define the 30-day window.
    # We limit the check to the last 30 days to prevent unbounded memory/compute growth
    # as the historical dataset expands.
    max_timestamp = cast(
        pl.DataFrame, combined_actuals.select(pl.col("timestamp").max()).collect()
    ).item()
    if max_timestamp is None:
        return dg.AssetCheckResult(passed=True, description="No data to check.")

    # Filter combined_actuals to only include rows where timestamp is within 30 days of the maximum timestamp.
    df_recent = combined_actuals.filter(pl.col("timestamp") >= max_timestamp - timedelta(days=30))

    # Calculate max absolute value per substation on this filtered 30-day window.
    stats = cast(
        pl.DataFrame,
        df_recent.group_by("substation_number")
        .agg(max_abs=pl.col("MW_or_MVA").abs().max())
        .collect(),
    )

    # Use a small epsilon threshold instead of exact float equality to account for
    # floating-point inaccuracies and minor sensor noise in dead substations.
    zero_subs = stats.filter(pl.col("max_abs") < 1e-3).get_column("substation_number").to_list()

    if zero_subs:
        return dg.AssetCheckResult(
            passed=True,
            severity=dg.AssetCheckSeverity.WARN,
            description=f"Found {len(zero_subs)} substations with all zero values in the last 30 days.",
            metadata={"zero_substations": dg.MetadataValue.json(zero_subs[:100])},
        )

    return dg.AssetCheckResult(
        passed=True, description="No all-zero substations found in the last 30 days."
    )


@dg.asset
def healthy_substations(
    context: dg.AssetExecutionContext, config: XGBoostConfig, combined_actuals: pl.LazyFrame
) -> list[int]:
    """Filters for substations with healthy telemetry."""
    if combined_actuals.collect_schema().names() == []:
        return []

    # Load the hydra config for "xgboost" and apply overrides from `config` to determine
    # the training period. This ensures substations are evaluated for health only during
    # the training period, preventing temporal data leakage and lookahead bias.
    hydra_config = load_hydra_config("xgboost")
    hydra_config = _apply_config_overrides(hydra_config, config)
    train_start = hydra_config.data_split.train_start
    train_end = hydra_config.data_split.train_end

    # Filter combined_actuals to only include data between train_start and train_end
    # before computing health statistics.
    df = combined_actuals.filter(
        (pl.col("timestamp").dt.date() >= train_start)
        & (pl.col("timestamp").dt.date() <= train_end)
    )

    # Add date column for daily grouping
    df = df.with_columns(date=pl.col("timestamp").dt.date())

    # Calculate daily stats per substation
    daily_stats = df.group_by(["substation_number", "date"]).agg(
        max_abs=pl.col("MW_or_MVA").abs().max(),
        std=pl.col("MW_or_MVA").std(),
        count=pl.col("MW_or_MVA").count(),
    )

    # Identify bad days: days with low activity or low variance despite having enough data points.
    daily_health = daily_stats.with_columns(
        is_bad_day=(pl.col("count") > 50)
        & ((pl.col("max_abs") < 0.5) | (pl.col("std").fill_null(0.0) < 0.01))
    )

    # Calculate the percentage of bad days relative to the total number of days the substation
    # has data in the training period. We use a 5% tolerance threshold to tolerate real-world
    # telemetry noise without discarding valuable training data.
    substation_health = (
        daily_health.group_by("substation_number")
        .agg(
            total_days=pl.len(),
            bad_days_count=pl.col("is_bad_day").sum(),
        )
        .with_columns(bad_day_ratio=pl.col("bad_days_count") / pl.col("total_days"))
    )

    # Filter for healthy substations
    healthy_stats = cast(
        pl.DataFrame,
        substation_health.filter(pl.col("bad_day_ratio") <= 0.05).collect(),
    )
    healthy_ids = healthy_stats.get_column("substation_number").to_list()

    # Get all unique substations to identify discarded ones
    all_substations = (
        cast(pl.DataFrame, combined_actuals.select("substation_number").unique().collect())
        .get_column("substation_number")
        .to_list()
    )
    substations_with_bad_data = [s for s in all_substations if s not in healthy_ids]

    context.log.info(
        f"Found {len(healthy_ids)} healthy substations out of {len(all_substations)} total. "
        f"Discarded {len(substations_with_bad_data)} due to bad telemetry."
    )

    context.add_output_metadata(
        {
            "num_healthy": len(healthy_ids),
            "num_discarded": len(substations_with_bad_data),
            "discarded_ids": dg.MetadataValue.json(substations_with_bad_data[:100]),
        }
    )

    return healthy_ids


@dg.asset(
    partitions_def=model_partitions,
    auto_materialize_policy=dg.AutoMaterializePolicy.eager(),
)
def metrics(
    context: dg.AssetExecutionContext,
    combined_actuals: pl.LazyFrame,
    settings: ResourceParam[Settings],
) -> pl.DataFrame:
    """Computes MAE/RMSE per substation for a specific model partition."""
    model_name = context.partition_key
    log.info(f"Computing metrics for model: {model_name}")

    # TODO: Implement Delta Lake read for evaluation_results.delta
    # results_df = pl.read_delta("data/evaluation_results.delta").filter(pl.col("power_fcst_model_name") == model_name)

    # For now, we'll just return an empty DataFrame as a placeholder
    # since we don't have the actual Delta Lake logic implemented yet.
    return pl.DataFrame()
