import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import PurePosixPath
    from typing import cast

    from contracts.settings import Settings

    settings = Settings()

    import altair as alt
    import geoarrow.pyarrow as geo_pyarrow
    import lonboard
    import marimo as mo
    from datetime import datetime
    from pathlib import Path
    import polars as pl
    import pyarrow
    from contracts.data_schemas import SubstationLocationsWithH3
    from nged_data import ckan
    from nged_data.substation_names.align import join_location_table_to_live_primaries

    BASE_PATH = Path("~/dev/python/nged-substation-forecast").expanduser()

    BASE_DELTA_PATH = BASE_PATH / settings.nged_data_path / "delta" / "live_primary_flows"
    return (
        BASE_DELTA_PATH,
        BASE_PATH,
        PurePosixPath,
        SubstationLocationsWithH3,
        alt,
        cast,
        ckan,
        datetime,
        geo_pyarrow,
        join_location_table_to_live_primaries,
        lonboard,
        mo,
        pl,
        pyarrow,
        settings,
    )


@app.cell
def _(
    BASE_PATH,
    PurePosixPath,
    SubstationLocationsWithH3,
    ckan,
    join_location_table_to_live_primaries,
    pl,
    settings,
):
    locations_path = (
        BASE_PATH / settings.nged_data_path / "parquet" / "substation_locations.parquet"
    )
    _locations = SubstationLocationsWithH3.validate(pl.read_parquet(locations_path))

    _live_primaries = ckan.get_csv_resources_for_live_primary_substation_flows(
        api_key=settings.nged_ckan_token
    )

    df = join_location_table_to_live_primaries(
        live_primaries=_live_primaries,
        locations=_locations,  # type: ignore[arg-type]
    )
    df = df.with_columns(
        substation_name=pl.col("url").map_elements(
            lambda url: PurePosixPath(url.path).stem, return_dtype=pl.String
        )
    )
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df, geo_pyarrow, pl, pyarrow):
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
            "csv_stem": df["substation_name"],
        },
    )
    return (arrow_table,)


@app.cell
def _(arrow_table, lonboard, mo):
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
    layer_widget = mo.ui.anywidget(layer)  # type: ignore[invalid-argument-type]
    return layer_widget, map


@app.cell
def _(BASE_DELTA_PATH, alt, cast, datetime, df, layer_widget, map, mo, pl):
    delta_df = pl.scan_delta(str(BASE_DELTA_PATH)).filter(
        pl.col("timestamp") > pl.lit(datetime(2026, 3, 1)).cast(pl.Datetime("us", "UTC"))
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
        substation_name = selected_df["substation_name"].item()

        try:
            filtered_demand = cast(
                pl.DataFrame,
                delta_df.filter(pl.col("substation_name") == substation_name).collect(),
            )
        except Exception as e:
            right_pane = mo.md(f"{e}")
        else:
            if filtered_demand.height == 0:
                right_pane = mo.md("No data")
            else:
                power_column = "MW" if filtered_demand["MW"].is_not_null().all() else "MVA"
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


if __name__ == "__main__":
    app.run()
