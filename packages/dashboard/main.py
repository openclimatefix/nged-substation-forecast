import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")

with app.setup:
    from pathlib import Path, PurePosixPath
    from typing import Final

    import altair as alt
    import geoarrow.pyarrow as geo_pyarrow
    import pyarrow
    import lonboard
    import marimo as mo
    import polars as pl
    from nged_data import ckan
    from nged_data.substation_names.align import join_location_table_to_live_primaries

    # TODO(Jack): This path should be configured once for the entire uv workspace.
    BASE_PARQUET_PATH: Final[Path] = Path(
        "~/dev/python/nged-substation-forecast/data/NGED/parquet/live_primary_flows"
    ).expanduser()


@app.cell
def _():
    # TODO: Dagster should grab the latest locations (only when it updates)
    #       and store the locations locally.
    _locations = ckan.get_primary_substation_locations()
    _live_primaries = ckan.get_csv_resources_for_live_primary_substation_flows()

    df = join_location_table_to_live_primaries(live_primaries=_live_primaries, locations=_locations)
    df = df.with_columns(
        parquet_filename=pl.col("url").map_elements(
            lambda url: PurePosixPath(url.path).with_suffix(".parquet").name, return_dtype=pl.String
        )
    )
    return (df,)


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

    arrow_table = pyarrow.table(
        {
            "geometry": geo_array,
            "name": df["substation_name_in_location_table"],
            "number": df["substation_number"],
        },
    )
    return (arrow_table,)


@app.cell
def _(arrow_table):
    layer = lonboard.ScatterplotLayer(
        arrow_table,
        pickable=True,
        auto_highlight=True,
        # Styling
        get_fill_color=[0, 128, 255],
        get_radius=1000,
        radius_units="meters",
        stroked=False,  # No outline.
    )

    map = lonboard.Map(layers=[layer])

    # Enable reactivity in Marimo:
    layer_widget = mo.ui.anywidget(layer)  # type: ignore[invalid-argument-type]
    return layer_widget, map


@app.cell
def _(df, layer_widget, map):
    if layer_widget.selected_index is None:
        right_pane = mo.md(
            """
            ### Select a Substation
            *Click a dot on the map to view the demand profile.*
            """
        )
    else:
        selected_df = df[layer_widget.selected_index]
        parquet_filename = selected_df["parquet_filename"].item()

        try:
            filtered_demand = pl.read_parquet(BASE_PARQUET_PATH / parquet_filename)
        except Exception as e:
            right_pane = mo.md(f"{e}")
        else:
            power_column = "MW" if "MW" in filtered_demand else "MVA"
            right_pane = (
                alt.Chart(filtered_demand)
                .mark_line()
                .encode(
                    x=alt.X(
                        "timestamp:T",
                        axis=alt.Axis(format="%H:%M %b %d"),
                    ),
                    y=alt.Y(f"{power_column}:Q", title=f"Demand ({power_column})"),
                    color=alt.value("teal"),
                    tooltip=["timestamp", power_column],
                )
                .properties(
                    title=selected_df["substation_name_in_location_table"].item(),
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
