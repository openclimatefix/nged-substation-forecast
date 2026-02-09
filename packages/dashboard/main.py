import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import polars as pl
    import altair as alt
    from typing import Final
    from pathlib import PurePosixPath, Path

    from ipydeck import Deck, Layer, ViewState
    from spatial_polars import SpatialFrame

    from nged_data import ckan
    from nged_data.substation_names.align import join_location_table_to_live_primaries

    BASE_PARQUET_PATH: Final[Path] = Path(
        "~/dev/python/nged-substation-forecast/data/NGED/parquet/live_primary_flows"
    ).expanduser()


@app.cell
def _():
    _locations = ckan.get_primary_substation_locations()
    _live_primaries = ckan.get_csv_resources_for_live_primary_substation_flows()

    joined = join_location_table_to_live_primaries(
        live_primaries=_live_primaries, locations=_locations
    )
    # joined = joined.filter(pl.col("simple_name").is_in(["Albrighton", "Alderton", "Alveston", "Bayston Hill", "Bearstone"]))
    joined = joined.with_columns(
        parquet_filename=pl.col("url").map_elements(
            lambda url: PurePosixPath(url.path).with_suffix(".parquet").name, return_dtype=pl.String
        )
    )
    return (joined,)


@app.cell
def _():
    return


@app.cell
def _(joined):
    sdf = SpatialFrame.from_point_coords(
        joined, x_col="longitude", y_col="latitude", crs="EPSG:4326"
    )
    return (sdf,)


@app.cell
def _(sdf):
    layers = [
        Layer(
            type="ScatterPlotLayer",
            data=sdf,
        )
    ]

    view_state = ViewState(
        latitude=49.254, longitude=-123.13, zoom=11, max_zoom=16, pitch=45, bearing=0
    )

    deck = Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="light",
    )

    deck
    return


@app.cell
def _():
    get_selected_index, set_selected_index = mo.state(None)
    return get_selected_index, set_selected_index


@app.cell
def _(set_selected_index):
    # --- THE BRIDGE ---
    # Define a callback that runs whenever the layer's 'selected_index' changes
    def on_map_click(change: dict):
        # change['new'] contains the integer index of the clicked point
        new_index = change.get("new")
        if new_index is not None:
            set_selected_index(new_index.get("selected_index"))

    return (on_map_click,)


@app.cell
def _(lonboard, on_map_click, sdf):
    layer = sdf.spatial.to_scatterplotlayer(
        pickable=True,  # enables the selection events
        auto_highlight=True,  # provides immediate visual feedback on hover
        # Styling
        fill_color=[0, 128, 255],
        radius=1000,
        radius_units="meters",
    )

    # Attach the callback to the layer
    layer.observe(on_map_click)

    # Create and display the map
    m = lonboard.Map(layers=[layer])
    return (m,)


@app.cell
def _(get_selected_index, joined, m, refresh):
    # Retrieve the current selection
    selected_idx = get_selected_index()

    if selected_idx is None:
        right_pane = mo.md(
            """
            ### Select a Substation
            *Click a dot on the map to view the demand profile.*
            """
        )
    else:
        selected_df = joined[selected_idx]
        parquet_filename = selected_df["parquet_filename"].item()

        try:
            filtered_demand = pl.read_parquet(BASE_PARQUET_PATH / parquet_filename)
        except Exception:
            right_pane = mo.md("e")
        else:
            power_column = "MW" if "MW" in filtered_demand else "MVA"

            # Create Time Series Chart
            right_pane = (
                alt.Chart(filtered_demand)
                .mark_line()
                .encode(
                    x=alt.X(
                        "timestamp:T",
                        axis=alt.Axis(
                            format="%H:%M %b %d",
                            # labelAngle=-45,  # Tilting labels often helps clarity
                        ),
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

    dashboard = mo.vstack([m, right_pane, refresh], heights=[4, 4, 1])  # , gap="2rem")
    dashboard
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
