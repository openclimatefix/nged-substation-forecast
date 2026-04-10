import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")

with app.setup:
    from typing import cast

    from contracts.settings import Settings

    settings = Settings()

    from datetime import datetime
    from pathlib import Path

    import altair as alt
    import geoarrow.pyarrow as geo_pyarrow
    import lonboard
    import marimo as mo
    import patito as pt
    import polars as pl
    import pyarrow
    from contracts.data_schemas import TimeSeriesMetadata, PowerTimeSeries

    BASE_PATH = Path("~/dev/python/nged-substation-forecast").expanduser()

    BASE_DELTA_PATH = BASE_PATH / settings.nged_data_path / "delta" / "power_time_series"


@app.cell
def _():
    metadata_path = BASE_PATH / settings.nged_data_path / "parquet" / "time_series_metadata.parquet"
    df = TimeSeriesMetadata.validate(pl.read_parquet(metadata_path))
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    # Create arrow table
    geo_array = (
        geo_pyarrow.point()
        .with_crs("epsg:4326")
        .from_geobuffers(
            None,
            df["longitude"].cast(pl.Float64).to_numpy(),
            df["latitude"].cast(pl.Float64).to_numpy(),
        )
    )

    attributes_to_include = [
        "time_series_id",
        "time_series_name",
        "time_series_type",
        "units",
        "licence_area",
        "substation_number",
        "substation_type",
    ]

    dict_for_table = {key: df[key] for key in attributes_to_include}
    dict_for_table["geometry"] = geo_array

    arrow_table = pyarrow.table(dict_for_table)
    return (arrow_table,)


@app.cell
def _(arrow_table):
    layer = lonboard.ScatterplotLayer(
        arrow_table,
        pickable=True,
        # Styling
        auto_highlight=True,
        get_fill_color=[0, 128, 255],
        get_radius=1000,
        radius_units="meters",
        stroked=False,  # No outline
    )
    map = lonboard.Map(layers=[layer])

    # Enable reactivity in Marimo:
    layer_widget = mo.ui.anywidget(layer)
    return layer_widget, map


@app.cell
def _(df, layer_widget, map):
    delta_df = pl.scan_delta(str(BASE_DELTA_PATH)).filter(
        pl.col("period_end_time") > pl.lit(datetime(2026, 3, 1)).cast(pl.Datetime("us", "UTC"))
    )

    if layer_widget.selected_index is None:
        right_pane = mo.md(
            """
            ### Select a Substation
            *Click a dot on the map to view the demand profile.*
            """
        )
    else:
        selected_df = df[layer_widget.selected_index]
        time_series_id = selected_df["time_series_id"].item()

        try:
            filtered_demand = cast(
                pt.DataFrame[PowerTimeSeries],
                delta_df.filter(pl.col("time_series_id") == time_series_id).collect(),
            )
        except Exception as e:
            right_pane = mo.md(f"{e}")
        else:
            if filtered_demand.height == 0:
                right_pane = mo.md("No data")
            else:
                right_pane = (
                    alt.Chart(filtered_demand)
                    .mark_line()
                    .encode(
                        x=alt.X(
                            "period_end_time:T",
                            axis=alt.Axis(format="%H:%M %b %d"),
                        ),
                        y=alt.Y("power:Q", title=f"Power ({selected_df['units'].item()})"),
                        color=alt.value("teal"),
                        tooltip=["period_end_time", "power"],
                    )
                    .properties(
                        title=f"{selected_df['time_series_name'].item()} - {selected_df['substation_type'].item()} - {selected_df['time_series_type'].item()}",
                        height=300,
                        width="container",  # Fill available width
                    )
                    .interactive()
                )

    mo.vstack([map, right_pane])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
