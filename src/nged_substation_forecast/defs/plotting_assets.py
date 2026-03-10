import random

import altair as alt
import dagster as dg
import polars as pl

from nged_substation_forecast.config_resource import NgedConfig


class PlotConfig(dg.Config):
    """Configuration for the forecast vs actual plot asset."""

    substation_ids: list[int] = []


@dg.asset
def forecast_vs_actual_plot(
    context: dg.AssetExecutionContext,
    xgb_forecasts: pl.DataFrame,
    combined_actuals: pl.DataFrame,
    config: PlotConfig,
    nged_config: NgedConfig,
):
    """Generates an Altair plot comparing forecast vs actuals.

    Args:
        context: Asset execution context.
        xgb_forecasts: Combined forecast dataframe.
        combined_actuals: Combined actual power data.
        config: PlotConfig.
        nged_config: NgedConfig.
    """
    if xgb_forecasts.is_empty() or combined_actuals.is_empty():
        context.log.warning("Forecast or actuals are empty for plotting.")
        return

    substation_ids = config.substation_ids
    if not substation_ids:
        # Pick 5 random substations if none specified
        unique_ids = xgb_forecasts.select("substation_id").unique()["substation_id"].to_list()
        substation_ids = random.sample(unique_ids, min(5, len(unique_ids)))

    # Filter data
    plot_forecast = xgb_forecasts.filter(pl.col("substation_id").is_in(substation_ids))
    plot_actuals = combined_actuals.filter(pl.col("substation_id").is_in(substation_ids))

    if plot_forecast.is_empty() or plot_actuals.is_empty():
        context.log.warning("No data for selected substations in plotting.")
        return

    # Prepare for plotting
    # TODO: Don't use `to_pandas()`!
    df_f = plot_forecast.to_pandas().rename(columns={"valid_time": "time", "power_mw": "value"})
    df_f["type"] = "Forecast"

    # TODO: Don't use `to_pandas()`!
    df_a = plot_actuals.to_pandas().rename(columns={"timestamp": "time", "power_mw": "value"})
    df_a["type"] = "Actual"

    import pandas as pd  # TODO: Remove!

    combined_df = pd.concat([df_f, df_a])

    # Create Altair chart
    (
        alt.Chart(combined_df)
        .mark_line()
        .encode(
            x="time:T",
            y="value:Q",
            color="substation_id:N",
            strokeDash="type:N",
            tooltip=["time", "value", "substation_id", "type"],
        )
        .properties(
            title=f"Forecast vs Actuals for Substation(s): {substation_ids}", width=800, height=400
        )
        .interactive()
    )

    # Log chart and metadata
    context.add_output_metadata(
        {
            "plot_metadata": dg.MetadataValue.text("Altair plot generated"),
            "substation_ids_plotted": dg.MetadataValue.json(substation_ids),
        }
    )

    return
