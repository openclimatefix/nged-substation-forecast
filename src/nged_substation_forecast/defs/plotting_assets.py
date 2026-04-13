from datetime import datetime, timedelta
from typing import cast

import altair as alt
import dagster as dg
import polars as pl
from contracts.settings import Settings
from dagster import ResourceParam
from ..utils import scan_delta_table


class PlotConfig(dg.Config):
    """Configuration for the forecast vs actual plot asset."""

    output_path: str = "tests/xgboost_integration_plot.html"
    max_time_series: int = 6


@dg.asset(
    ins={
        "predictions": dg.AssetIn("evaluate_xgboost"),
    },
    deps=["cleaned_power_time_series"],
    compute_kind="python",
    group_name="plots",
)
def forecast_vs_actual_plot(
    context: dg.AssetExecutionContext,
    predictions: pl.DataFrame,
    config: PlotConfig,
    settings: ResourceParam[Settings],
):
    """Generates an Altair plot comparing forecast vs actuals."""

    # Empty Data Guard: Before performing any timestamp arithmetic, check if data is present.
    # We use get_cleaned_power_time_series_lazy to ensure we have the full history required for the plot.
    # This function serves as the single source of truth for accessing cleaned power time series.
    if predictions.is_empty():
        context.log.warning("Empty predictions, skipping plot.")
        return

    # Load time series metadata
    time_series_metadata = pl.read_parquet(
        settings.nged_data_path / "parquet" / "time_series_metadata.parquet"
    )

    # Extract unique substation numbers from predictions and limit for plotting.
    pred_time_series_ids = (
        predictions.get_column("time_series_id").unique().to_list()[: config.max_time_series]
    )

    # Keep actuals lazy and filter by substation first to avoid eager collection.
    # We need to ensure we have enough data for the 14-day plot.
    # We calculate the required start time based on the predictions.
    max_pred_time = predictions.get_column("valid_time").max()

    # We need data from at least 14 days before the max_pred_time to ensure we have
    # enough actuals for the comparison.
    required_start_time = cast(datetime, max_pred_time) - timedelta(days=14)

    cleaned_power_time_series_lazy = scan_delta_table(
        str(settings.nged_data_path / "delta" / "cleaned_power_time_series")
    ).filter(pl.col("period_end_time") >= required_start_time)

    raw_power_lazy = scan_delta_table(
        str(settings.nged_data_path / "delta" / "raw_power_time_series")
    ).filter(pl.col("period_end_time") >= required_start_time)

    # Filter actuals and raw by substation.
    actuals_30m = cast(
        pl.DataFrame,
        cleaned_power_time_series_lazy.filter(
            pl.col("time_series_id").is_in(pred_time_series_ids)
        ).collect(),
    )
    raw_30m = cast(
        pl.DataFrame,
        raw_power_lazy.filter(pl.col("time_series_id").is_in(pred_time_series_ids)).collect(),
    )

    if actuals_30m.is_empty():
        context.log.warning("No actuals found for the predicted time series, skipping plot.")
        return

    # Calculate target_init_time = max_actual_time - 14 days.
    # We select the latest nwp_init_time that is <= max_actual_time - 14 days to guarantee
    # that the entire 14-day forecast horizon has corresponding actuals for comparison.
    max_actual_time = cast(datetime, actuals_30m.get_column("period_end_time").max())
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

    latest_predictions = predictions.filter(pl.col("nwp_init_time") == chosen_init_time).rename(
        {"valid_time": "period_end_time"}
    )

    # Join predictions with actuals and raw. We use a 'left' join with latest_predictions
    # on the left to ensure that all 14 days of the forecast trajectory are preserved
    # in the plot, even if actuals/raw are missing for the later days.
    eval_df = latest_predictions.join(
        actuals_30m.rename({"power": "actual"}),
        on=["period_end_time", "time_series_id"],
        how="left",
    ).join(
        raw_30m.rename({"power": "raw"}),
        on=["period_end_time", "time_series_id"],
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
        (pl.col("period_end_time") >= chosen_init_time) & (pl.col("period_end_time") <= horizon_end)
    )

    # Empty Plot Window Guard: Ensure we have data in the 14-day window.
    if plot_df.is_empty():
        context.log.warning(
            f"No data found in the 14-day window starting {chosen_init_time}, skipping."
        )
        return

    # Join with time_series_metadata to get names. Joining after filtering minimizes DF size.
    # We convert to plain Polars DataFrames to avoid Patito subclass join type mismatches.
    plot_df = (
        pl.DataFrame(plot_df)
        .join(
            pl.DataFrame(time_series_metadata).select(["time_series_id", "time_series_name"]),
            on="time_series_id",
            how="inner",
        )
        .with_columns(
            time_series_name_with_id=pl.format(
                "{} ({})",
                pl.col("time_series_name"),
                pl.col("time_series_id"),
            )
        )
        .sort("period_end_time")
    )

    # Generate Altair Chart using layers
    # We use a shared color scale to ensure the legend is consistent
    color_scale = alt.Scale(domain=["Forecast", "Actual", "Raw"], range=["blue", "red", "green"])

    charts = []
    time_series_names_with_ids = plot_df.get_column("time_series_name_with_id").unique().to_list()

    for ts_name_with_id in time_series_names_with_ids:
        ts_df = plot_df.filter(pl.col("time_series_name_with_id") == ts_name_with_id)

        # Forecasts: 51 members
        preds_df = (
            ts_df.select(["period_end_time", "ensemble_member", "power_fcst"])
            .rename({"power_fcst": "power"})
            .with_columns(
                type=pl.lit("Forecast"),
                ensemble_member=pl.col("ensemble_member").cast(pl.Int32),
                power=pl.col("power").cast(pl.Float64).round(2),
            )
            .unique()
        )

        # Actuals: single line per substation
        actuals_df = (
            ts_df.select(["period_end_time", "actual"])
            .unique()
            .rename({"actual": "power"})
            .with_columns(
                type=pl.lit("Actual"),
                ensemble_member=pl.lit(0, dtype=pl.Int32),
                power=pl.col("power").cast(pl.Float64).round(2),
            )
        )

        # Raw: single line per substation
        raw_df = (
            ts_df.select(["period_end_time", "raw"])
            .unique()
            .rename({"raw": "power"})
            .with_columns(
                type=pl.lit("Raw"),
                ensemble_member=pl.lit(0, dtype=pl.Int32),
                power=pl.col("power").cast(pl.Float64).round(2),
            )
        )

        # Combine and convert to CSV
        ts_data_df = (
            pl.concat([preds_df, actuals_df, raw_df], how="diagonal")
            .select(["period_end_time", "power", "ensemble_member", "type"])
            .with_columns(period_end_time=pl.col("period_end_time").dt.replace_time_zone(None))
            .unique()
        )

        # Embed data directly
        csv_string = ts_data_df.write_csv()
        data = alt.Data(values=csv_string, format=alt.DataFormat(type="csv"))

        # Generate Altair Chart for this substation
        base = alt.Chart(data).encode(
            x=alt.X("period_end_time:T", title="Time"),
            y=alt.Y("power:Q", title="Power (MW/MVA)"),
            color=alt.Color("type:N", scale=color_scale, title="Type"),
        )

        preds_layer = (
            base.transform_filter(alt.datum.type == "Forecast")
            .mark_line(strokeWidth=0.5, opacity=0.7)
            .encode(detail="ensemble_member:N")
        )

        actuals_layer = base.transform_filter(alt.datum.type == "Actual").mark_line(
            strokeWidth=1.0, opacity=0.7
        )

        raw_layer = base.transform_filter(alt.datum.type == "Raw").mark_line(
            strokeWidth=1.0, opacity=0.7, strokeDash=[5, 5]
        )

        chart = alt.layer(preds_layer, actuals_layer, raw_layer).properties(
            width=400, height=200, title=ts_name_with_id
        )

        charts.append(chart)

    # Combine charts
    final_chart = (
        alt.concat(*charts, columns=2)
        .resolve_scale(y="independent")
        .properties(
            title=alt.TitleParams(
                text="Actuals vs Predictions", subtitle=f"NWP Init Time: {chosen_init_time}"
            )
        )
    )

    final_chart.save(config.output_path)
    context.log.info(f"Plot saved to {config.output_path}")

    return dg.MaterializeResult(
        metadata={
            "plot_path": dg.MetadataValue.path(config.output_path),
            "num_points": len(plot_df),
            "chosen_init_time": str(chosen_init_time),
        }
    )
