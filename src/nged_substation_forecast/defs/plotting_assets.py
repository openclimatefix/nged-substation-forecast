from datetime import datetime, timedelta
from typing import cast

import altair as alt
import dagster as dg
import polars as pl
from contracts.settings import Settings
from dagster import ResourceParam
from ml_core.data import downsample_power_flows
from nged_data import clean_substation_flows


class PlotConfig(dg.Config):
    """Configuration for the forecast vs actual plot asset."""

    output_path: str = "tests/xgboost_integration_plot.html"


@dg.asset(
    ins={
        "predictions": dg.AssetIn("evaluate_xgboost"),
        "substation_metadata": dg.AssetIn("substation_metadata"),
    },
    deps=["live_primary_flows"],
    compute_kind="python",
    group_name="plots",
)
def forecast_vs_actual_plot(
    context: dg.AssetExecutionContext,
    predictions: pl.DataFrame,
    substation_metadata: pl.DataFrame,
    config: PlotConfig,
    settings: ResourceParam[Settings],
):
    """Generates an Altair plot comparing forecast vs actuals."""

    # Empty Data Guard: Before performing any timestamp arithmetic, check if data is present.
    # We read directly from the live_primary_flows Delta table instead of using the
    # cleaned_actuals asset. This is because the cleaned_actuals asset is partitioned
    # and might only provide a single day of data if the job is run for a single partition.
    # By reading the Delta table directly, we ensure we have the full 14-day history
    # required for the plot.
    delta_path = str(settings.nged_data_path / "delta" / "live_primary_flows")
    raw_flows = pl.scan_delta(delta_path)

    # We apply the same cleaning logic as the production pipeline to ensure consistency.
    # This replaces stuck sensors and insane values with nulls.
    cleaned_actuals = clean_substation_flows(cast(pl.DataFrame, raw_flows.collect()), settings)

    if predictions.is_empty() or cleaned_actuals.is_empty():
        context.log.warning("Empty predictions or actuals, skipping plot.")
        return

    # Extract unique substation numbers from predictions
    pred_substations = predictions.get_column("substation_number").unique().to_list()

    # Downsample actuals to 30m to match predictions, filtering by substation first.
    actuals_30m = cast(
        pl.DataFrame,
        downsample_power_flows(
            cleaned_actuals.filter(pl.col("substation_number").is_in(pred_substations))
        ).collect(),
    )

    if actuals_30m.is_empty():
        context.log.warning("No actuals found for the predicted substations, skipping plot.")
        return

    # Calculate target_init_time = max_actual_time - 14 days.
    # We select the latest nwp_init_time that is <= max_actual_time - 14 days to guarantee
    # that the entire 14-day forecast horizon has corresponding actuals for comparison.
    max_actual_time = cast(datetime, actuals_30m.get_column("timestamp").max())
    target_init_time = max_actual_time - timedelta(days=14)

    # Select chosen_init_time: max nwp_init_time <= target_init_time
    nwp_init_times = predictions.get_column("nwp_init_time").unique().sort()
    if nwp_init_times.is_empty():
        context.log.warning("No nwp_init_time found in predictions, skipping plot.")
        return

    valid_init_times = nwp_init_times.filter(nwp_init_times <= target_init_time)

    if not valid_init_times.is_empty():
        chosen_init_time = cast(datetime, valid_init_times.max())
    else:
        chosen_init_time = cast(datetime, nwp_init_times.min())
        context.log.warning(
            f"No NWP init time found before {target_init_time}. "
            f"Falling back to earliest available: {chosen_init_time}"
        )

    latest_predictions = predictions.filter(pl.col("nwp_init_time") == chosen_init_time)

    # Join predictions with actuals. We use a 'left' join with latest_predictions
    # on the left to ensure that all 14 days of the forecast trajectory are preserved
    # in the plot, even if actuals are missing for the later days.
    eval_df = pl.DataFrame(latest_predictions).join(
        actuals_30m.rename({"timestamp": "valid_time", "MW_or_MVA": "actual"}),
        on=["valid_time", "substation_number"],
        how="left",
    )

    # Overlap Check: Because a left join guarantees rows exist, we must check if the
    # 'actual' column consists entirely of nulls to detect when there are no actuals
    # to plot against.
    if eval_df.is_empty() or eval_df.get_column("actual").null_count() == len(eval_df):
        context.log.warning("No overlapping data for plotting, skipping.")
        return

    # Filter plot to 14-day horizon starting from chosen_init_time
    horizon_end = chosen_init_time + timedelta(days=14)
    plot_df = eval_df.filter(
        (pl.col("valid_time") >= chosen_init_time) & (pl.col("valid_time") <= horizon_end)
    )

    # Empty Plot Window Guard: Ensure we have data in the 14-day window.
    if plot_df.is_empty():
        context.log.warning(
            f"No data found in the 14-day window starting {chosen_init_time}, skipping."
        )
        return

    # Join with substation_metadata to get names. Joining after filtering minimizes DF size.
    # We convert to plain Polars DataFrames to avoid Patito subclass join type mismatches.
    plot_df = (
        pl.DataFrame(plot_df)
        .join(
            pl.DataFrame(substation_metadata).select(
                ["substation_number", "substation_name_in_location_table"]
            ),
            on="substation_number",
            how="inner",
        )
        .with_columns(
            substation_name_with_id=pl.format(
                "{} ({})",
                pl.col("substation_name_in_location_table"),
                pl.col("substation_number"),
            )
        )
        .sort("valid_time")
    )

    # Prepare a single dataframe for Altair to avoid faceting issues and duplication
    # Forecasts: 51 members
    preds_df = (
        plot_df.select(["valid_time", "substation_name_with_id", "ensemble_member", "MW_or_MVA"])
        .rename({"MW_or_MVA": "power"})
        .with_columns(
            type=pl.lit("Forecast"),
            ensemble_member=pl.col("ensemble_member").cast(pl.Int32),
        )
    )

    # Actuals: single line per substation (avoiding 51x duplication)
    actuals_df = (
        plot_df.select(["valid_time", "substation_name_with_id", "actual"])
        .unique()
        .rename({"actual": "power"})
        .with_columns(
            type=pl.lit("Actual"),
            ensemble_member=pl.lit(0, dtype=pl.Int32),
        )
    )

    combined_plot_df = pl.concat([preds_df, actuals_df], how="diagonal").to_pandas()

    # Generate Altair Chart using layers
    # We use a shared color scale to ensure the legend is consistent
    color_scale = alt.Scale(domain=["Forecast", "Actual"], range=["blue", "black"])

    base = alt.Chart(combined_plot_df).encode(
        x=alt.X("valid_time:T", title="Time"),
        y=alt.Y("power:Q", title="Power (MW/MVA)"),
        color=alt.Color("type:N", scale=color_scale, title="Type"),
    )

    # Layer 1: Ensemble predictions (51 members) with thin, semi-transparent lines
    preds_layer = (
        base.transform_filter(alt.datum.type == "Forecast")
        .mark_line(strokeWidth=0.5, opacity=0.3)
        .encode(detail="ensemble_member:N")
    )

    # Layer 2: Actuals with a single thick, solid line
    actuals_layer = base.transform_filter(alt.datum.type == "Actual").mark_line(
        strokeWidth=2, opacity=1.0
    )

    # Different substations have vastly different power capacities, so independent y-axes
    # are necessary to visualize the forecast accuracy for smaller substations.
    # Explicitly showing the chosen_init_time as a subtitle ensures users understand
    # the exact forecast initialization time being visualized.
    chart = (
        alt.layer(preds_layer, actuals_layer)
        .properties(width=400, height=200)
        .facet(facet="substation_name_with_id:N", columns=2)
        .resolve_scale(y="independent")
        .properties(
            title=alt.TitleParams(
                text="Actuals vs Predictions", subtitle=f"NWP Init Time: {chosen_init_time}"
            )
        )
    )

    chart.save(config.output_path)
    context.log.info(f"Plot saved to {config.output_path}")

    return dg.MaterializeResult(
        metadata={
            "plot_path": dg.MetadataValue.path(config.output_path),
            "num_points": len(plot_df),
            "chosen_init_time": str(chosen_init_time),
        }
    )
