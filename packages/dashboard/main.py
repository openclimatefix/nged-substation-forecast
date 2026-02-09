import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import polars as pl
    import altair as alt
    from typing import Final
    from pathlib import PurePosixPath, Path

    import lonboard
    from spatial_polars import SpatialFrame

    from nged_data import ckan
    from nged_data.substation_names.align import join_location_table_to_live_primaries

    BASE_PARQUET_PATH: Final[Path] = Path(
        "~/dev/python/nged-substation-forecast/data/NGED/parquet/live_primary_flows"
    ).expanduser()


@app.cell
def _():
    # TODO: Dagster should grab the latest locations (only when it updates)
    #       and store the locations locally.
    _locations = ckan.get_primary_substation_locations()
    _live_primaries = ckan.get_csv_resources_for_live_primary_substation_flows()

    joined = join_location_table_to_live_primaries(live_primaries=_live_primaries, locations=_locations)
    joined = joined.with_columns(
        parquet_filename=pl.col("url").map_elements(
            lambda url: PurePosixPath(url.path).with_suffix(".parquet").name, return_dtype=pl.String
        )
    )
    return (joined,)


@app.cell
def _(joined):
    sdf = SpatialFrame.from_point_coords(joined, x_col="longitude", y_col="latitude", crs="EPSG:4326")
    return (sdf,)


@app.cell
def _(sdf):
    # Docs: https://atl2001.github.io/spatial_polars/SpatialFrame/#spatial_polars.spatialframe.SpatialFrame.to_scatterplotlayer
    layer = sdf.spatial.to_scatterplotlayer(
        pickable=True,
        auto_highlight=True,
        # Styling
        radius=1000,
        radius_units="meters",
        stroked=False,
    )

    # https://developmentseed.org/lonboard/latest/api/layers/scatterplot-layer/
    layer.get_fill_color = [0, 128, 255]

    map = lonboard.Map(layers=[layer])
    layer_widget = mo.ui.anywidget(layer)
    return layer_widget, map


@app.cell
def _(joined, layer_widget, map):
    if layer_widget.selected_index is None:
        right_pane = mo.md(
            """
            ### Select a Substation
            *Click a dot on the map to view the demand profile.*
            """
        )
    else:
        selected_df = joined[layer_widget.selected_index]
        parquet_filename = selected_df["parquet_filename"].item()

        try:
            filtered_demand = pl.read_parquet(BASE_PARQUET_PATH / parquet_filename)
        except Exception:
            right_pane = mo.md("e")
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
            )

    dashboard = mo.vstack([map, right_pane], heights=[4, 4])
    dashboard
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
