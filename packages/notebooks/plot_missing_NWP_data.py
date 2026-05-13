import marimo

__generated_with = "0.23.6"
app = marimo.App()

with app.setup:
    import marimo as mo
    from datetime import datetime, timezone
    import altair as alt
    import polars as pl


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
def _(df, nwp_vars):
    def plot_null_distribution(df, target_init_time, target_h3_index, nwp_vars):
        # 1. Filter down to the specific init_time and h3_index
        filtered = df.filter((pl.col("init_time") == target_init_time) & (pl.col("h3_index") == target_h3_index))

        # 2. Unpivot (melt) the data so we can plot all variables on a single Y-axis
        melted = filtered.unpivot(
            on=nwp_vars, index=["valid_time", "ensemble_member"], variable_name="variable", value_name="value"
        )

        # 3. Create a boolean flag for missing data and a combined label for the Y-axis
        plot_df = melted.with_columns(
            # Check for both database Nulls and float NaNs
            is_missing=pl.col("value").is_null() | pl.col("value").is_nan(),
            # Create a string like "temperature_2m (Member 0)"
            # row_label=(pl.col("variable") + " (Member " + pl.col("ensemble_member").cast(pl.Utf8) + ")"),
            row_label=pl.col("ensemble_member"),
        )

        # 5. Build the Altair Chart
        base = alt.Chart(plot_df).encode(y=alt.Y("row_label:N", title="Ensemble Member", sort="ascending"))

        # Layer 1: A light gray background line showing the full time series extent
        background_lines = base.mark_line(color="lightgray", strokeWidth=1).encode(
            x=alt.X("valid_time:T", title="Valid Time"),
            detail="row_label:N",  # Ensures lines don't connect across different rows
        )

        # Layer 2: Red ticks superimposed exactly where the data is missing
        missing_marks = (
            base.transform_filter(alt.datum.is_missing == True)
            .mark_tick(
                color="red",
                thickness=3,  # Make the red mark stand out
                size=12,  # Height of the tick mark
            )
            .encode(x="valid_time:T")
        )

        # Combine the layers and configure the chart size
        chart = (
            (background_lines + missing_marks)
            .properties(
                title=f"Missing NWP Data | init_time: {target_init_time.strftime('%Y-%m-%d')} | {nwp_vars} | H3: {target_h3_index}",
                width=800,
                height=alt.Step(10),  # Dynamically scales chart height based on the number of rows
            )
            .configure_axis(labelFontSize=11, titleFontSize=13)
        )

        return chart


    chart = plot_null_distribution(
        df,
        target_init_time=datetime(2026, 5, 1, tzinfo=timezone.utc),
        target_h3_index=599148110664433663,
        nwp_vars=nwp_vars[0],  # ["downward_short_wave_radiation_flux_surface"], #
    )
    chart.show()
    return


if __name__ == "__main__":
    app.run()
