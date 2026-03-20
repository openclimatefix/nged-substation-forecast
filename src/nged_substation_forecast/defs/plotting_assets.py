import random
from datetime import datetime, timedelta
from typing import cast

import altair as alt
import dagster as dg
import polars as pl
from dagster import ResourceParam

from contracts.settings import Settings


class PlotConfig(dg.Config):
    """Configuration for the forecast vs actual plot asset."""

    substation_ids: list[int] = []


@dg.asset
def forecast_vs_actual_plot(
    context: dg.AssetExecutionContext,
    xgb_forecasts: pl.DataFrame,
    combined_actuals: pl.DataFrame,
    substation_metadata: pl.DataFrame,
    config: PlotConfig,
    settings: ResourceParam[Settings],
):
    """Generates an Altair plot comparing forecast vs actuals.

    Args:
        context: Asset execution context.
        xgb_forecasts: Combined forecast dataframe.
        combined_actuals: Combined actual power data.
        substation_metadata: Substation metadata.
        config: PlotConfig.
        settings: Settings.
    """
    if xgb_forecasts.is_empty() or combined_actuals.is_empty():
        context.log.warning("Forecast or actuals are empty for plotting.")
        return

    substation_ids = config.substation_ids
    if not substation_ids:
        # Pick 5 random substations if none specified
        unique_ids = (
            xgb_forecasts.select("substation_number").unique()["substation_number"].to_list()
        )
        substation_ids = random.sample(unique_ids, min(5, len(unique_ids)))

    # Filter data
    plot_forecast = xgb_forecasts.filter(pl.col("substation_number").is_in(substation_ids))

    if plot_forecast.is_empty():
        context.log.warning("No forecast data for selected substations.")
        return

    # Get the start of the forecast to filter actuals
    forecast_start = cast(datetime, plot_forecast["valid_time"].min())
    history_start = forecast_start - timedelta(days=2)

    plot_actuals = combined_actuals.filter(
        (pl.col("substation_number").is_in(substation_ids))
        & (pl.col("timestamp") >= history_start)
        & (pl.col("timestamp") <= plot_forecast["valid_time"].max())
    )

    if plot_actuals.is_empty():
        context.log.warning("No actuals data for selected substations in the plotting window.")
        # We can still plot the forecast though, but the goal is comparison.

    # Join with metadata to get names
    metadata_subset = substation_metadata.select(
        ["substation_number", "substation_name_in_location_table"]
    ).rename({"substation_name_in_location_table": "substation_name"})

    # Prepare for plotting using Polars
    df_f = (
        plot_forecast.join(metadata_subset, on="substation_number", how="left")
        .with_columns(
            type=pl.lit("Forecast"),
            display_name=pl.format(
                "{} ({})", pl.col("substation_name"), pl.col("substation_number")
            ),
        )
        .select(
            time=pl.col("valid_time"),
            value=pl.col("MW_or_MVA"),
            type=pl.col("type"),
            display_name=pl.col("display_name"),
            ensemble_member=pl.col("ensemble_member"),
        )
    )

    df_a = (
        plot_actuals.join(metadata_subset, on="substation_number", how="left")
        .with_columns(
            type=pl.lit("Actual"),
            display_name=pl.format(
                "{} ({})", pl.col("substation_name"), pl.col("substation_number")
            ),
            ensemble_member=pl.lit(0).cast(pl.UInt8),
        )
        .select(
            time=pl.col("timestamp"),
            value=pl.col("MW_or_MVA"),
            type=pl.col("type"),
            display_name=pl.col("display_name"),
            ensemble_member=pl.col("ensemble_member"),
        )
    )

    combined_df = pl.concat([df_f, df_a])

    # Create Altair chart with subplots (facets)
    # We use a layered approach for the spaghetti plot
    # Note: For faceting layered charts, the data must be at the top level
    base = alt.Chart(combined_df).encode(
        x=alt.X("time:T", title="Time (UTC)"),
        y=alt.Y("value:Q", title="Power (MW/MVA)"),
    )

    forecast_layer = (
        base.transform_filter(alt.datum.type == "Forecast")
        .mark_line(strokeWidth=1, opacity=0.15, color="#F18536")
        .encode(
            detail="ensemble_member:N",
            tooltip=[
                alt.Tooltip("time:T", format="%Y-%m-%d %H:%M"),
                alt.Tooltip("value:Q", format=".2f"),
                alt.Tooltip("display_name:N"),
                alt.Tooltip("ensemble_member:N"),
                alt.Tooltip("type:N"),
            ],
        )
    )

    actuals_layer = (
        base.transform_filter(alt.datum.type == "Actual")
        .mark_line(strokeWidth=2, opacity=1, color="#5276A7")
        .encode(
            tooltip=[
                alt.Tooltip("time:T", format="%Y-%m-%d %H:%M"),
                alt.Tooltip("value:Q", format=".2f"),
                alt.Tooltip("display_name:N"),
                alt.Tooltip("type:N"),
            ],
        )
    )

    chart = (
        alt.layer(forecast_layer, actuals_layer)
        .properties(width=800, height=250)
        .facet(row=alt.Row("display_name:N", title="Substation"), spacing=40)
        .resolve_scale(y="independent")
        .configure_title(fontSize=16, anchor="start", color="gray")
        .configure_axis(labelFontSize=12, titleFontSize=14)
        .configure_legend(labelFontSize=12, titleFontSize=14)
        .interactive()
    )

    # Save chart to HTML for easy viewing
    plot_path = settings.nged_data_path / "plots" / "forecast_vs_actual.html"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    chart.save(str(plot_path))

    # Log chart and metadata
    context.add_output_metadata(
        {
            "plot_metadata": dg.MetadataValue.text("Altair plot generated"),
            "substation_ids_plotted": dg.MetadataValue.json(substation_ids),
            "plot_url": dg.MetadataValue.path(str(plot_path)),
        }
    )

    return
