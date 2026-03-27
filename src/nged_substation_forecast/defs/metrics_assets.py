import logging

import dagster as dg
import polars as pl
from dagster import ResourceParam

from contracts.settings import Settings
from xgboost_forecaster import DataConfig, get_substation_metadata
from .partitions import model_partitions

log = logging.getLogger(__name__)


@dg.asset(deps=["live_primary_flows"])
def combined_actuals(
    context: dg.AssetExecutionContext, settings: ResourceParam[Settings]
) -> pl.DataFrame:
    """Combines all live primary flows into a single dataframe."""
    actuals_path = settings.nged_data_path / "delta" / "live_primary_flows"
    if not actuals_path.exists():
        return pl.DataFrame()

    df = pl.read_delta(str(actuals_path))
    if df.is_empty():
        return pl.DataFrame()

    # Get metadata to map filenames to substation IDs
    data_config = DataConfig(
        base_power_path=actuals_path,
        base_weather_path=settings.nwp_data_path / "ECMWF" / "ENS",
    )
    metadata = get_substation_metadata(data_config)

    # Join metadata to get substation_number
    df = df.join(metadata.select(["substation_number"]), on="substation_number", how="inner")

    # Some actuals might have 'MW', others 'MVA'.
    df = df.with_columns(MW_or_MVA=pl.coalesce(["MW", "MVA"]))

    return df


@dg.asset
def healthy_substations(
    context: dg.AssetExecutionContext, combined_actuals: pl.DataFrame
) -> list[int]:
    """Filters for substations with healthy telemetry."""
    if combined_actuals.is_empty():
        return []

    # Add date column for daily grouping
    df = combined_actuals.with_columns(date=pl.col("timestamp").dt.date())

    # Calculate daily stats per substation
    daily_stats = df.group_by(["substation_number", "date"]).agg(
        max_abs=pl.col("MW_or_MVA").abs().max(),
        std=pl.col("MW_or_MVA").std(),
        count=pl.col("MW_or_MVA").count(),
    )

    # Identify bad days
    bad_days = daily_stats.filter(
        (pl.col("count") > 50) & ((pl.col("max_abs") < 0.5) | (pl.col("std").fill_null(0.0) < 0.01))
    )

    # Get substations with ANY bad days
    substations_with_bad_data = bad_days.select("substation_number").unique().to_series().to_list()

    # Get all unique substations and filter out the bad ones
    all_substations = combined_actuals.select("substation_number").unique().to_series().to_list()
    healthy_ids = [s for s in all_substations if s not in substations_with_bad_data]

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
    combined_actuals: pl.DataFrame,
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
