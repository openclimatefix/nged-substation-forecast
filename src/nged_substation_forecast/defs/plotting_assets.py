import dagster as dg
import polars as pl
from dagster import ResourceParam

from contracts.settings import Settings
from .partitions import model_partitions


class PlotConfig(dg.Config):
    """Configuration for the forecast vs actual plot asset."""

    substation_ids: list[int] = []


@dg.asset(
    partitions_def=model_partitions,
    auto_materialize_policy=dg.AutoMaterializePolicy.eager(),
)
def forecast_vs_actual_plot(
    context: dg.AssetExecutionContext,
    combined_actuals: pl.DataFrame,
    substation_metadata: pl.DataFrame,
    config: PlotConfig,
    settings: ResourceParam[Settings],
):
    """Generates an Altair plot comparing forecast vs actuals for a specific model partition."""
    model_name = context.partition_key
    context.log.info(f"Generating plot for model: {model_name}")

    # TODO: Implement Delta Lake read for evaluation_results.delta
    # forecasts = pl.read_delta("data/evaluation_results.delta").filter(pl.col("power_fcst_model_name") == model_name)

    # For now, we'll just return as a placeholder
    return
