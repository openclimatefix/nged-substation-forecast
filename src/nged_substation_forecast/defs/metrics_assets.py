"""Asset definitions for computing forecasting metrics.

This module defines the `metrics` asset, which computes evaluation metrics (MAE, RMSE, etc.)
for trained models by reading evaluation results from Delta Lake storage.
"""

import logging

import dagster as dg
import polars as pl
from dagster import ResourceParam

from contracts.settings import Settings

log = logging.getLogger(__name__)


@dg.asset(
    partitions_def=dg.DailyPartitionsDefinition(start_date="2026-03-10"),
    auto_materialize_policy=dg.AutoMaterializePolicy.eager(),
)
def metrics(
    context: dg.AssetExecutionContext,
    cleaned_actuals: pl.DataFrame,
    settings: ResourceParam[Settings],
) -> pl.DataFrame:
    """Computes MAE/RMSE per substation for a specific model partition.

    This asset reads evaluation results from Delta Lake storage and computes
    performance metrics against cleaned actuals data. It uses the cleaned_actuals
    asset which applies data quality checks and replaces problematic values with null.

    Args:
        context: Dagster asset execution context.
        cleaned_actuals: Cleaned actuals data with problematic values replaced by null.
        settings: Global settings containing data paths.

    Returns:
        Polars DataFrame containing computed metrics per substation.
        Returns empty DataFrame if no evaluation results are found.

    Notes:
        - The metrics computation reads from `evaluation_results.delta` table.
        - Uses `cleaned_actuals` (not `combined_actuals`) to ensure evaluation
          is performed on physically plausible data.
        - The cleaned_actuals DataFrame is eager (not lazy), so no .collect() needed.
    """
    model_name = context.partition_key
    log.info(f"Computing metrics for model: {model_name}")

    # TODO: Implement Delta Lake read for evaluation_results.delta
    # results_df = pl.read_delta("data/evaluation_results.delta").filter(
    #     pl.col("power_fcst_model_name") == model_name
    # )

    # For now, we'll just return an empty DataFrame as a placeholder
    # since we don't have the actual Delta Lake logic implemented yet.
    return pl.DataFrame()
