import dagster as dg
import polars as pl
import altair as alt
from typing import cast
from datetime import timedelta, datetime
from dagster import ResourceParam

from contracts.settings import Settings
from ml_core.data import downsample_power_flows


class PlotConfig(dg.Config):
    """Configuration for the forecast vs actual plot asset."""

    substation_ids: list[int] = []
    output_path: str = "tests/xgboost_integration_plot.html"


@dg.asset(
    ins={
        "predictions": dg.AssetIn("evaluate_xgboost"),
        "combined_actuals": dg.AssetIn("combined_actuals"),
    },
    compute_kind="python",
    group_name="plots",
)
def forecast_vs_actual_plot(
    context: dg.AssetExecutionContext,
    predictions: pl.DataFrame,
    combined_actuals: pl.LazyFrame,
    config: PlotConfig,
    settings: ResourceParam[Settings],
):
    """Generates an Altair plot comparing forecast vs actuals."""
    if predictions.is_empty() or combined_actuals.collect_schema().names() == []:
        context.log.warning("Empty predictions or actuals, skipping plot.")
        return

    # 1. Extract unique substation numbers from predictions
    pred_substations = predictions.get_column("substation_number").unique().to_list()

    # 2. Downsample actuals to 30m to match predictions, filtering by substation first
    actuals_30m = cast(
        pl.DataFrame,
        downsample_power_flows(
            combined_actuals.filter(pl.col("substation_number").is_in(pred_substations))
        ).collect(),
    )

    # 3. Filter predictions to the latest NWP run available in the set
    max_init = predictions.get_column("nwp_init_time").max()
    if max_init is None:
        context.log.warning("No nwp_init_time found in predictions, skipping plot.")
        return

    latest_predictions = predictions.filter(pl.col("nwp_init_time") == max_init)

    # 4. Join predictions with actuals
    eval_df = latest_predictions.join(
        actuals_30m.rename({"timestamp": "valid_time", "MW_or_MVA": "actual"}),
        on=["valid_time", "substation_number"],
        how="inner",
    )

    if eval_df.is_empty():
        context.log.warning("No overlapping data for plotting, skipping.")
        return

    # 5. Filter plot to only show the last 2 weeks of the test set
    max_date = cast(datetime, eval_df.get_column("valid_time").max())
    test_start = max_date - timedelta(days=14)
    plot_df = eval_df.filter(pl.col("valid_time") >= test_start)

    # Prepare a single dataframe for Altair to avoid faceting issues and duplication
    # Forecasts: 51 members
    preds_df = (
        plot_df.select(["valid_time", "substation_number", "ensemble_member", "MW_or_MVA"])
        .rename({"MW_or_MVA": "power"})
        .with_columns(
            type=pl.lit("Forecast"),
            ensemble_member=pl.col("ensemble_member").cast(pl.Int32),
        )
    )

    # Actuals: single line per substation (avoiding 51x duplication)
    actuals_df = (
        plot_df.select(["valid_time", "substation_number", "actual"])
        .unique()
        .rename({"actual": "power"})
        .with_columns(
            type=pl.lit("Actual"),
            ensemble_member=pl.lit(0, dtype=pl.Int32),
        )
    )

    combined_plot_df = pl.concat([preds_df, actuals_df], how="diagonal").to_pandas()

    # 6. Generate Altair Chart using layers
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

    chart = (
        alt.layer(preds_layer, actuals_layer)
        .properties(width=400, height=200)
        .facet(facet="substation_number:N", columns=2)
        .properties(title=f"Actuals vs Predictions (Latest NWP: {max_init})")
    )

    chart.save(config.output_path)
    context.log.info(f"Plot saved to {config.output_path}")

    return dg.MaterializeResult(
        metadata={
            "plot_path": dg.MetadataValue.path(config.output_path),
            "num_points": len(plot_df),
            "latest_nwp": str(max_init),
        }
    )
