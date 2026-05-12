import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    from datetime import datetime, timezone
    import polars as pl
    from lonboard import Map, H3HexagonLayer
    import altair as alt
    import numpy as np

    from contracts.settings import Settings, PROJECT_ROOT
    import dynamical_data.ecmwf_ens.download
    import dynamical_data.ecmwf_ens.convert_to_polars

    SETTINGS = Settings()


@app.cell
def _():
    h3_grid = pl.read_parquet(PROJECT_ROOT / SETTINGS.h3_grid_weights_path)
    h3_grid
    return (h3_grid,)


@app.cell
def _(h3_grid):
    ds = dynamical_data.ecmwf_ens.download.download_ecmwf_ens_run(
        nwp_init_time=datetime(2026, 5, 1, tzinfo=timezone.utc),
        h3_grid=h3_grid,
    )
    return (ds,)


@app.cell
def _(ds):
    ds
    return


@app.cell
def _(ds, h3_grid):
    df = dynamical_data.ecmwf_ens.convert_to_polars.convert_nwp_xarray_dataset_to_polars_dataframe(ds=ds, h3_grid=h3_grid)
    df
    return (df,)


@app.cell
def _(df):
    _df = df.filter([pl.col("ensemble_member") == 0, pl.col("h3_index") == 599148110664433663])

    # replace _df with your data source
    _chart = (
        alt.Chart(_df)
        .mark_line()
        .encode(
            x=alt.X(field="valid_time", type="temporal"),
            y=alt.Y(field="wind_direction_100m", type="quantitative"),
        )
        .properties(height=290, width="container", config={"axis": {"grid": False}})
    )
    _chart
    return


@app.cell
def _(df):
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

    _df = df.filter([pl.col("ensemble_member") == 0, pl.col("h3_index") == 599148110664433663]).with_columns(
        #    pl.col(nwp_vars).interpolate().rolling(index_column="valid_time", period="2h", closed="both")
    )


    def interpolate_single_nulls(col_name: str | list[str], groups: list[str]) -> pl.Expr:
        """Compute the full interpolation first, and then use pl.when().then().otherwise() with a
        Boolean mask to keep only interpolated values where the gap is exactly 1 null."""

        # Check if: (current is null) AND (previous is NOT null) AND (next is NOT null)
        is_null = pl.col(col_name).is_null()
        prev_valid = pl.col(col_name).shift(1).over(groups).is_not_null()
        next_valid = pl.col(col_name).shift(-1).over(groups).is_not_null()

        single_gap_mask = is_null & prev_valid & next_valid

        # Interpolate the whole series
        interpolated = pl.col(col_name).interpolate().over(groups)

        # Only keep the interpolated value if it was a single gap;
        # otherwise keep original (which preserves larger null gaps)
        return pl.when(single_gap_mask).then(interpolated).otherwise(pl.col(col_name))


    _chart = (
        alt.Chart(_df.with_columns(interpolate_single_nulls(nwp_vars, ["ensemble_member"])))
        .mark_line()
        .encode(
            x=alt.X(field="valid_time", type="temporal"),
            y=alt.Y(field="wind_direction_100m", type="quantitative"),
        )
        .properties(height=290, width="container", config={"axis": {"grid": False}})
    )
    _chart
    return interpolate_single_nulls, nwp_vars


@app.cell
def _(df):
    df_w_nan = df.filter([pl.col("ensemble_member") == 0, pl.col("h3_index") == 599148110664433663])
    df_w_nan.select(["valid_time", "wind_direction_100m"])
    return (df_w_nan,)


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
        nwp_vars=nwp_vars[13],  # ["downward_short_wave_radiation_flux_surface"], #
    )
    chart.show()
    return


@app.cell
def _(df, interpolate_single_nulls, nwp_vars):
    df_interp = df.with_columns(interpolate_single_nulls(nwp_vars, ["init_time", "ensemble_member", "h3_index"]))
    df_interp
    return (df_interp,)


@app.cell
def _(df_interp):
    from contracts import weather_schemas
    import patito as pt

    weather_schemas.NwpInMemory.validate(
        df_interp.cast({"categorical_precipitation_type_surface": pl.UInt8}),
        # allow_missing_columns=True,
    )
    return


@app.cell
def _(df_w_nan):
    groups = ["init_time", "ensemble_member", "h3_index"]

    is_null = df_w_nan.select(pl.col("temperature_2m").fill_nan(None).is_null())

    run_id = is_null.rle_id()
    gap_size = is_null.group_by(groups + [run_id]).sum()

    gap_size
    return


@app.cell
def _(ds):
    valid_time = datetime(2026, 5, 3, 9)
    ds["temperature_2m"].sel(valid_time=valid_time, ensemble_member=0).plot.imshow()
    return (valid_time,)


@app.cell
def _(df, valid_time):
    filtered_df = df.filter(ensemble_member=0, valid_time=valid_time.replace(tzinfo=timezone.utc))
    filtered_df
    return (filtered_df,)


@app.cell
def _(filtered_df):
    from palettable.matplotlib import Viridis_20
    from lonboard.colormap import apply_continuous_cmap

    temperature = filtered_df["categorical_precipitation_type_surface"]
    min_bound = temperature.min()
    max_bound = temperature.max() - min_bound

    normalized = (temperature - min_bound) / max_bound
    normalized
    return Viridis_20, apply_continuous_cmap, normalized


@app.cell
def _(Viridis_20, apply_continuous_cmap, filtered_df, normalized):
    Map(
        H3HexagonLayer(
            filtered_df,
            get_hexagon=filtered_df["h3_index"],
            get_fill_color=apply_continuous_cmap(normalized, Viridis_20, alpha=1),
            opacity=1,
        )
    )
    return


if __name__ == "__main__":
    app.run()
