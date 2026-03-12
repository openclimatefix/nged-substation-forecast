import marimo

__generated_with = "0.20.4"
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
    import polars as pl
    import pyarrow
    from contracts.data_schemas import SubstationMetadata, SubstationFlows

    BASE_PATH = Path("~/dev/python/nged-substation-forecast").expanduser()

    BASE_DELTA_PATH = BASE_PATH / settings.nged_data_path / "delta" / "live_primary_flows"


@app.cell
def _():
    metadata_path = BASE_PATH / settings.nged_data_path / "parquet" / "substation_metadata.parquet"
    df = SubstationMetadata.validate(pl.read_parquet(metadata_path))

    # Filter for substations with live telemetry
    df = df.filter(pl.col("url").is_not_null())
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
def _(df, layer_widget, map):
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
        substation_number = selected_df["substation_number"].item()

        try:
            filtered_demand = cast(
                pl.DataFrame,
                delta_df.filter(pl.col("substation_number") == substation_number).collect(),
            )
        except Exception as e:
            right_pane = mo.md(f"{e}")
        else:
            if filtered_demand.height == 0:
                right_pane = mo.md("No data")
            else:
                power_column = SubstationFlows.choose_power_column(filtered_demand)
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
