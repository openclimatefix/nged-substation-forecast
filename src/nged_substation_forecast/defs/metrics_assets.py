import polars as pl
import dagster as dg
from pathlib import Path
from xgboost_forecaster import get_substation_metadata

from contracts.data_schemas import Forecast
from contracts.config import METRICS_DATA_PATH


@dg.asset(deps=["live_primary_parquet"])
def combined_actuals() -> pl.DataFrame:
    """Combines all live primary parquet files into a single dataframe."""
    actuals_path = Path("data/NGED/parquet/live_primary_flows")
    if not actuals_path.exists():
        return pl.DataFrame()

    parquets = list(actuals_path.glob("*.parquet"))
    if not parquets:
        return pl.DataFrame()

    # Get metadata to map filenames to substation IDs
    metadata = get_substation_metadata()

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
        # Rename to power_mw for consistency with Forecast.
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
    xgb_forecast: dict[str, pl.DataFrame],
    combined_actuals: pl.DataFrame,
) -> pl.DataFrame:
    """Computes MAE/RMSE per substation."""
    if not xgb_forecast:
        context.log.warning("No forecasts provided.")
        return pl.DataFrame()

    all_forecasts = pl.concat(list(xgb_forecast.values()))

    if all_forecasts.is_empty() or combined_actuals.is_empty():
        context.log.warning("Forecast or actuals are empty.")
        return pl.DataFrame()

    # Validate forecasts against contract
    Forecast.validate(all_forecasts)

    # Ensure combined_actuals has substation_id for joining
    # If not present, we might need to skip or error.
    if "substation_id" not in combined_actuals.columns:
        context.log.error("combined_actuals missing 'substation_id' column.")
        return pl.DataFrame()

    # Join on substation_id and time
    # Forecast: valid_time, power_mw
    # Actuals: timestamp, power_mw
    comparison = all_forecasts.join(
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

    # Save results to a Parquet file in METRICS_DATA_PATH
    METRICS_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_parquet(METRICS_DATA_PATH)

    context.log.info(f"Saved metrics to {METRICS_DATA_PATH}")

    return metrics
