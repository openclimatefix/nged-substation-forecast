import dagster as dg
import polars as pl
from nged_substation_forecast.config_resource import NgedConfig
from contracts.data_schemas import PowerForecast
from xgboost_forecaster import get_substation_metadata, DataConfig


@dg.asset(deps=["live_primary_flows"])
def combined_actuals(context: dg.AssetExecutionContext, nged_config: NgedConfig) -> pl.DataFrame:
    """Combines all live primary flows into a single dataframe."""
    settings = nged_config.to_settings()
    actuals_path = settings.NGED_DATA_PATH / "delta" / "live_primary_flows"
    if not actuals_path.exists():
        return pl.DataFrame()

    df = pl.read_delta(str(actuals_path))
    if df.is_empty():
        return pl.DataFrame()

    # Get metadata to map filenames to substation IDs
    data_config = DataConfig(
        base_power_path=actuals_path,
        base_weather_path=settings.NWP_DATA_PATH / "ECMWF" / "ENS",
    )
    metadata = get_substation_metadata(data_config)

    # Join metadata to get substation_id
    metadata = metadata.with_columns(
        pl.col("parquet_filename").str.replace(".parquet", "").alias("substation_name")
    )

    df = df.join(
        metadata.select(["substation_name", "substation_number"]), on="substation_name", how="inner"
    ).rename({"substation_number": "substation_id"})

    # Some actuals might have 'MW', others 'MVA'.
    df = df.with_columns(pl.coalesce(["MW", "MVA"]).alias("power_mw"))

    return df


@dg.asset
def metrics_asset(
    context: dg.AssetExecutionContext,
    xgb_forecasts: pl.DataFrame,
    combined_actuals: pl.DataFrame,
    nged_config: NgedConfig,
) -> pl.DataFrame:
    """Computes MAE/RMSE per substation."""
    settings = nged_config.to_settings()
    if xgb_forecasts.is_empty() or combined_actuals.is_empty():
        context.log.warning("Forecast or actuals are empty.")
        return pl.DataFrame()

    # Validate forecasts against contract
    PowerForecast.validate(xgb_forecasts)

    # Join on substation_id and time
    comparison = xgb_forecasts.join(
        combined_actuals,
        left_on=["substation_id", "valid_time"],
        right_on=["substation_id", "timestamp"],
        suffix="_actual",
    )

    if comparison.is_empty():
        context.log.warning("No overlapping data between forecast and actuals.")
        return pl.DataFrame()

    # Compute metrics per substation
    metrics = comparison.group_by("substation_id").agg(
        [
            (pl.col("power_mw") - pl.col("power_mw_actual")).abs().mean().alias("mae"),
            ((pl.col("power_mw") - pl.col("power_mw_actual")) ** 2).mean().sqrt().alias("rmse"),
        ]
    )

    output_path = settings.FORECAST_METRICS_DATA_PATH
    output_path.mkdir(parents=True, exist_ok=True)
    metrics.write_parquet(output_path / "metrics.parquet")

    context.log.info(f"Saved metrics to {output_path}")

    return metrics
