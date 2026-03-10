import dagster as dg
import polars as pl
from nged_substation_forecast.config_resource import NgedConfig
from contracts.data_schemas import PowerForecast
from xgboost_forecaster import get_substation_metadata, DataConfig


@dg.asset(deps=["live_primary_flows"])
def combined_actuals(context: dg.AssetExecutionContext, config: NgedConfig) -> pl.DataFrame:
    """Combines all live primary parquet files into a single dataframe."""
    settings = config.to_settings()
    actuals_path = settings.NGED_DATA_PATH / "parquet/live_primary_flows"
    if not actuals_path.exists():
        return pl.DataFrame()

    parquets = list(actuals_path.glob("*.parquet"))
    if not parquets:
        return pl.DataFrame()

    # Get metadata to map filenames to substation IDs
    data_config = DataConfig(
        base_power_path=settings.NGED_DATA_PATH / "parquet" / "live_primary_flows",
        base_weather_path=settings.NWP_DATA_PATH / "ECMWF" / "ENS",
    )
    metadata = get_substation_metadata(data_config)

    dfs = []
    for p in parquets:
        df = pl.read_parquet(p)

        # Get substation ID for this file
        sub_meta = metadata.filter(pl.col("parquet_filename") == p.name)
        if not sub_meta.is_empty():
            sub_id = sub_meta["substation_number"][0]
            df = df.with_columns(pl.lit(sub_id).alias("substation_id").cast(pl.Int32))
        else:
            continue  # Skip if we don't know the substation ID

        # Some actuals might have 'MW', others 'MVA'.
        if "MW" in df.columns:
            df = df.rename({"MW": "power_mw"})
        elif "MVA" in df.columns:
            df = df.rename({"MVA": "power_mw"})

        dfs.append(df)

    if not dfs:
        return pl.DataFrame()

    return pl.concat(dfs, how="diagonal")


@dg.asset
def metrics_asset(
    context: dg.AssetExecutionContext,
    xgb_forecasts: pl.DataFrame,
    combined_actuals: pl.DataFrame,
    config: NgedConfig,
) -> pl.DataFrame:
    """Computes MAE/RMSE per substation."""
    settings = config.to_settings()
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
