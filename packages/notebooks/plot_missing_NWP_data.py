import marimo

__generated_with = "0.23.6"
app = marimo.App()

with app.setup:
    from datetime import UTC, datetime

    import altair as alt
    import plotting.ocf_theme  # noqa: F401 — registers OCF Altair theme as side effect
    import polars as pl
    from contracts.weather_schemas import Nwp
    from plotting.ocf_theme import GRID, ORANGE_RED


@app.cell
def _():
    df = Nwp.scan_delta().drop("nwp_model_id")
    return (df,)


@app.cell
def _():
    nwp_vars = [
        "geopotential_height_500hpa",
        "pressure_reduced_to_mean_sea_level",
        "categorical_precipitation_type_surface",
        "downward_long_wave_radiation_flux_surface",
        "pressure_surface",
        "temperature_2m",
        "precipitation_surface",
        "wind_speed_100m",
        "wind_direction_100m",
        "wind_speed_10m",
        "wind_direction_10m",
        "downward_short_wave_radiation_flux_surface",
        "dew_point_temperature_2m",
    ]
    return (nwp_vars,)


@app.cell
def _():
    def plot_null_distribution(df, target_init_time, target_h3_index, nwp_vars):
        """Chart where `nwp_vars` is missing, one row per ensemble member.

        Args:
            df: Lazy scan of the NWP Delta table.
            target_init_time: The single NWP run to plot.
            target_h3_index: The single H3 cell to plot.
            nwp_vars: The NWP variable name (or list of names) to plot.
        """
        # 1. Filter down to the specific init_time and h3_index
        filtered = df.filter(
            (pl.col("init_time") == target_init_time) & (pl.col("h3_index") == target_h3_index)
        )

        # 2. Unpivot (melt) the data so we can plot all variables on a single Y-axis
        melted = filtered.unpivot(
            on=nwp_vars,
            index=["valid_time", "ensemble_member"],
            variable_name="variable",
            value_name="value",
        )

        # 3. Create a boolean flag for missing data and a combined label for the Y-axis. Altair
        # only accepts materialised data, so collect once the filter has cut the scan down to a
        # single NWP run and a single H3 cell.
        plot_df = melted.with_columns(
            # Check for both database Nulls and float NaNs
            is_missing=pl.col("value").is_null() | pl.col("value").is_nan(),
            # Create a string like "temperature_2m (Member 0)":
            # row_label=(
            #     pl.col("variable") + " (Member " + pl.col("ensemble_member").cast(pl.Utf8) + ")"
            # ),
            row_label=pl.col("ensemble_member"),
        ).collect()

        # 5. Build the Altair Chart
        base = alt.Chart(plot_df).encode(
            y=alt.Y("row_label:N", title="Ensemble Member", sort="ascending")
        )

        # Layer 1: A light gray background line showing the full time series extent
        background_line = base.mark_line(color=GRID, strokeWidth=1)
        background_lines = background_line.encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("valid_time:T", title="Valid Time"),
            detail="row_label:N",  # Ensures lines don't connect across different rows
        )

        # Layer 2: Red ticks superimposed exactly where the data is missing
        missing_marks = (
            base.transform_filter(alt.datum.is_missing)
            .mark_tick(
                color=ORANGE_RED,
                thickness=3,  # Make the red mark stand out
                size=12,  # Height of the tick mark
            )
            .encode(x="valid_time:T")  # ty: ignore[unresolved-attribute]  # astral-sh/ty#2520
        )

        # Combine the layers and configure the chart size
        return (
            (background_lines + missing_marks)
            .properties(
                title=(
                    f"Missing NWP Data | init_time: {target_init_time.strftime('%Y-%m-%d')}"
                    f" | {nwp_vars} | H3: {target_h3_index}"
                ),
                width=800,
                # Dynamically scales chart height based on the number of rows
                height=alt.Step(10),
            )
            .configure_axis(labelFontSize=11, titleFontSize=13)
        )

    return (plot_null_distribution,)


@app.cell
def _(df, nwp_vars, plot_null_distribution):
    chart = plot_null_distribution(
        df,
        target_init_time=datetime(2026, 5, 1, tzinfo=UTC),
        target_h3_index=599148110664433663,
        nwp_vars=nwp_vars[0],  # ["downward_short_wave_radiation_flux_surface"], #
    )
    chart.show()
    return


@app.cell
def _(plot_null_distribution):
    def test_plot_null_distribution_flags_nulls_and_nans():
        init_time = datetime(2026, 5, 1, tzinfo=UTC)
        h3_index = 599148110664433663
        readings = pl.LazyFrame(
            {
                "init_time": [init_time] * 4,
                "valid_time": [datetime(2026, 5, 1, hour=h, tzinfo=UTC) for h in range(4)],
                "ensemble_member": [0, 0, 0, 0],
                "h3_index": [h3_index] * 4,
                "temperature_2m": pl.Series([1.0, None, float("nan"), 4.0], dtype=pl.Float32),
            }
        )

        chart = plot_null_distribution(
            readings,
            target_init_time=init_time,
            target_h3_index=h3_index,
            nwp_vars="temperature_2m",
        )

        # Altair inlines the chart's data as its single named dataset.
        (rows,) = chart.to_dict()["datasets"].values()
        assert [row["is_missing"] for row in rows] == [False, True, True, False]

    return


if __name__ == "__main__":
    app.run()
