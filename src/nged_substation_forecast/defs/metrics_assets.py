import dagster as dg
import polars as pl
from dagster import ResourceParam

from contracts.data_schemas import PowerForecast
from contracts.settings import Settings
from xgboost_forecaster import DataConfig, get_substation_metadata


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
    """Filters for substations with healthy telemetry.

    A substation is considered healthy if it has no "bad" days in the dataset.
    A day is bad if the peak absolute power is < 0.5 MW/MVA or the standard
    deviation is < 0.01 (indicating stuck telemetry).

    TODO: Ideally, `healthy_substations` would return a list of substations and the
    date ranges of healthy data for each substation (so we can still train on a
    substation that occasionally has bad data, we'd just ignore that bad data).
    """
    if combined_actuals.is_empty():
        return []

    # 1. Add date column for daily grouping
    df = combined_actuals.with_columns(date=pl.col("timestamp").dt.date())

    # 2. Calculate daily stats per substation
    daily_stats = df.group_by(["substation_number", "date"]).agg(
        max_abs=pl.col("MW_or_MVA").abs().max(),
        std=pl.col("MW_or_MVA").std(),
    )

    # 3. Identify bad days
    # Peak < 0.5 MW or Std < 0.01
    bad_days = daily_stats.filter((pl.col("max_abs") < 0.5) | (pl.col("std") < 0.01))

    # 4. Get substations with ANY bad days
    substations_with_bad_data = bad_days.select("substation_number").unique().to_series().to_list()

    # 5. Get all unique substations and filter out the bad ones
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


@dg.asset
def metrics_asset(
    context: dg.AssetExecutionContext,
    xgb_forecasts: pl.DataFrame,
    combined_actuals: pl.DataFrame,
    settings: ResourceParam[Settings],
) -> pl.DataFrame:
    """Computes MAE/RMSE per substation."""
    if xgb_forecasts.is_empty() or combined_actuals.is_empty():
        context.log.warning("Forecast or actuals are empty.")
        return pl.DataFrame()

    # Validate forecasts against contract
    PowerForecast.validate(xgb_forecasts)

    # Join on substation_number and time
    comparison = xgb_forecasts.join(
        combined_actuals,
        left_on=["substation_number", "valid_time"],
        right_on=["substation_number", "timestamp"],
        suffix="_actual",
    )

    if comparison.is_empty():
        context.log.warning("No overlapping data between forecast and actuals.")
        return pl.DataFrame()

    # Compute metrics per substation
    metrics = comparison.group_by("substation_number").agg(
        mae=(pl.col("MW_or_MVA") - pl.col("MW_or_MVA_actual")).abs().mean(),
        rmse=((pl.col("MW_or_MVA") - pl.col("MW_or_MVA_actual")) ** 2).mean().sqrt(),
    )

    output_path = settings.forecast_metrics_data_path
    output_path.mkdir(parents=True, exist_ok=True)
    metrics.write_parquet(output_path / "metrics.parquet")

    context.log.info(f"Saved metrics to {output_path}")

    return metrics
