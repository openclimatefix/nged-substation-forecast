import polars as pl
import dagster as dg
import altair as alt
import random


@dg.asset(
    config_schema={"substation_ids": dg.Field(dg.Array(int), is_required=False, default_value=[])}
)
def forecast_vs_actual_plot(
    context: dg.AssetExecutionContext,
    xgb_forecast: dict[str, pl.DataFrame],
    combined_actuals: pl.DataFrame,
):
    """Generates an Altair plot comparing forecast vs actuals.

    Args:
        context: Asset execution context.
        xgb_forecast: Dictionary of forecast dataframes per partition.
        combined_actuals: Combined actual power data.
    """
    if not xgb_forecast:
        context.log.warning("No forecasts provided for plotting.")
        return

    all_forecasts = pl.concat(list(xgb_forecast.values()))

    if all_forecasts.is_empty() or combined_actuals.is_empty():
        context.log.warning("Forecast or actuals are empty for plotting.")
        return

    substation_ids = context.op_config.get("substation_ids", [])
    if not substation_ids:
        # Pick 5 random substations if none specified
        unique_ids = all_forecasts.select("substation_id").unique()["substation_id"].to_list()
        substation_ids = random.sample(unique_ids, min(5, len(unique_ids)))

    # Filter data
    plot_forecast = all_forecasts.filter(pl.col("substation_id").is_in(substation_ids))
    plot_actuals = combined_actuals.filter(pl.col("substation_id").is_in(substation_ids))

    if plot_forecast.is_empty() or plot_actuals.is_empty():
        context.log.warning("No data for selected substations in plotting.")
        return

    # Prepare for plotting
    df_f = plot_forecast.to_pandas().rename(columns={"valid_time": "time", "power_mw": "value"})
    df_f["type"] = "Forecast"

    df_a = plot_actuals.to_pandas().rename(columns={"timestamp": "time", "power_mw": "value"})
    df_a["type"] = "Actual"

    import pandas as pd

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
