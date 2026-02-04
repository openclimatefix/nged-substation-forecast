import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    from typing import Final
    from pathlib import PurePosixPath, Path

    import lonboard
    import geopandas as gpd

    from nged_data import ckan
    from nged_data.substation_names.align import join_location_table_to_live_primaries
    return (
        Final,
        Path,
        PurePosixPath,
        alt,
        ckan,
        gpd,
        join_location_table_to_live_primaries,
        lonboard,
        mo,
        pl,
    )


@app.cell
def _(Final, Path):
    BASE_PARQUET_PATH: Final[Path] = Path(
        "~/dev/python/nged-substation-forecast/data/NGED/parquet/live_primary_flows"
    ).expanduser()
    return (BASE_PARQUET_PATH,)


@app.cell
def _(PurePosixPath, ckan, join_location_table_to_live_primaries, pl):
    _locations = ckan.get_primary_substation_locations()
    _live_primaries = ckan.get_csv_resources_for_live_primary_substation_flows()

    joined = join_location_table_to_live_primaries(live_primaries=_live_primaries, locations=_locations)
    # joined = joined.filter(pl.col("simple_name").is_in(["Albrighton", "Alderton", "Alveston", "Bayston Hill", "Bearstone"]))
    joined = joined.with_columns(
        parquet_filename=pl.col("url").map_elements(
            lambda url: PurePosixPath(url.path).with_suffix(".parquet").name, return_dtype=pl.String
        )
    )
    joined
    return (joined,)


@app.cell
def _(gpd, joined):
    pandas_df = joined.select(
        ["substation_number", "latitude", "longitude", "parquet_filename", "substation_name_in_location_table"]
    ).to_pandas()

    gdf = gpd.GeoDataFrame(
        pandas_df,
        geometry=gpd.points_from_xy(pandas_df["longitude"], pandas_df["latitude"]),
        crs="EPSG:4326",  # Defines the coordinate system as standard Lat/Lon
    )
    return (gdf,)


@app.cell
def _(mo):
    get_selected_index, set_selected_index = mo.state(None)
    return get_selected_index, set_selected_index


@app.cell
def _(mo):
    refresh = mo.ui.refresh(default_interval="1s")
    return (refresh,)


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
def _(gdf, lonboard, on_map_click):
    layer = lonboard.ScatterplotLayer.from_geopandas(
        gdf,
        pickable=True,  # enables the selection events
        auto_highlight=True,  # provides immediate visual feedback on hover
        # Styling
        get_fill_color=[0, 128, 255],
        get_radius=1000,
        radius_units="meters",
    )

    # Attach the callback to the layer
    layer.observe(on_map_click)

    # Create and display the map
    m = lonboard.Map(layers=[layer])
    return (m,)


@app.cell
def _(
    BASE_PARQUET_PATH: "Final[Path]",
    alt,
    get_selected_index,
    joined,
    m,
    mo,
    pl,
    refresh,
):
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

        filtered_demand = pl.read_parquet(BASE_PARQUET_PATH / parquet_filename)

        power_column = "MW" if "MW" in filtered_demand else "MVA"

        # Create Time Series Chart
        right_pane = (
            alt.Chart(filtered_demand)
            .mark_line()
            .encode(
                x="timestamp:T",
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


    dashboard = mo.hstack([m, right_pane, refresh], widths=[4, 4, 1])  # , gap="2rem")
    dashboard
    return


if __name__ == "__main__":
    app.run()
