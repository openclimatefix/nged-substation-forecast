import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    from vega_datasets import data
    from typing import Final
    from pathlib import PurePosixPath, Path

    from nged_data import ckan
    from nged_data.substation_names.align import join_location_table_to_live_primaries
    return (
        Final,
        Path,
        PurePosixPath,
        alt,
        ckan,
        data,
        join_location_table_to_live_primaries,
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
    joined = joined.filter(pl.col("simple_name").is_in(["Albrighton", "Alderton", "Alveston", "Bayston Hill", "Bearstone"]))
    joined = joined.with_columns(
        parquet_filename=pl.col("url").map_elements(
            lambda url: PurePosixPath(url.path).with_suffix(".parquet").name, return_dtype=pl.String
        )
    )
    joined
    return (joined,)


@app.cell
def _(Final, alt, data, joined, mo):
    # Define the Base Map (Optional but recommended for context)
    # We use a simple background of the world or specific region
    countries = alt.topo_feature(data.world_110m.url, "countries")
    map_background = alt.Chart(countries).mark_geoshape(fill="#f0f0f0", stroke="white")

    SUBSTATION_NAME_COL: Final[str] = "simple_name"

    select_substation = alt.selection_point(fields=[SUBSTATION_NAME_COL])

    substation_points = (
        alt.Chart(joined)
        .mark_circle(size=100, color="teal")
        .encode(
            longitude="longitude:Q",
            latitude="latitude:Q",
            tooltip=[SUBSTATION_NAME_COL, "latitude", "longitude"],
            opacity=alt.condition(select_substation, alt.value(1), alt.value(0.2)),
            color=alt.condition(select_substation, alt.value("teal"), alt.value("lightgray")),
        )
    ).add_params(select_substation)

    final_map = (map_background + substation_points).project(
        "mercator",
        scale=3000,  # Zoom level (high for local data)
        center=[-1, 52.5],  # Center on your data
    )

    # Create the Marimo UI Element
    # This renders the chart and makes it reactive
    map_widget = mo.ui.altair_chart(final_map)
    return (map_widget,)


@app.cell
def _(BASE_PARQUET_PATH: "Final[Path]", alt, joined, map_widget, mo, pl):
    selected_df = map_widget.apply_selection(joined)

    if selected_df.height == 0:
        right_pane = mo.md(
            """
            ### Select a Substation
            *Click a dot on the map to view the demand profile.*
            """
        )
    else:
        # Extract the name (handle multiple selections if needed, here we take the first)
        parquet_filename = selected_df["parquet_filename"][0]

        # Filter the demand data using Polars
        # Note: Altair v6 + Polars is very fast here
        filtered_demand = pl.read_parquet(BASE_PARQUET_PATH / parquet_filename)

        # Create Time Series Chart
        ts_chart = (
            alt.Chart(filtered_demand)
            .mark_line()
            .encode(
                x="timestamp:T", y=alt.Y("MW:Q", title="Demand (MW)"), color=alt.value("teal"), tooltip=["timestamp", "MW"]
            )
            .properties(
                title=f"{parquet_filename}",
                height=300,
                width="container",  # Fill available width
            )
        )

        right_pane = mo.ui.altair_chart(ts_chart)


    dashboard = mo.hstack([map_widget, right_pane], widths=[2, 3], gap="2rem")
    dashboard
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
