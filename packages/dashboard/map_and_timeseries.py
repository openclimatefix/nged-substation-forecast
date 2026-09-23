import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")

with app.setup:
    from datetime import UTC, datetime, timedelta
    from typing import Final, cast

    import altair as alt
    import geoarrow.pyarrow as geo_pyarrow
    import lonboard
    import marimo as mo
    import patito as pt
    import plotting.ocf_theme  # noqa: F401 — registers OCF Altair theme as side effect
    import polars as pl
    import pyarrow
    from anywidget import AnyWidget
    from contracts.common import UTC_DATETIME_DTYPE
    from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
    from contracts.typing_utils import typeddict_to_dict
    from dashboard.data_source import settings_for_source, source_status_message
    from plotting.ocf_theme import DATA_BLUE, hex_to_rgb

    RECENT_WINDOW: Final[timedelta] = timedelta(days=21)
    """How far back the power chart looks, measured from the selected series' newest row.

    NGED telemetry is half-hourly, 48 rows a day, and Altair's default row-count guard rejects a
    frame over 5,000 rows — so the window has to stay well under about 104 days. 21 days is
    roughly 1,000 rows, comfortably inside that limit, and covers the most recent few weeks the
    chart is meant to show.
    """


@app.cell
def _():
    source = mo.ui.radio(
        options=["local", "s3"],
        value="local",
        label="Data source",
        inline=True,
    )
    source
    return (source,)


@app.cell
def _(source):
    settings = settings_for_source(source.value)
    status_message, is_warning = source_status_message(source.value, settings)
    mo.callout(mo.md(status_message), kind="warn") if is_warning else mo.md(status_message)
    return (settings,)


@app.cell
def _(settings):
    metadata_path = settings.metadata_path
    df = TimeSeriesMetadata.validate(
        pl.read_parquet(metadata_path, storage_options=typeddict_to_dict(settings.storage_options))
    )
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
        get_fill_color=hex_to_rgb(DATA_BLUE),
        get_radius=1000,
        radius_units="meters",
        stroked=False,  # No outline
    )
    map = lonboard.Map(layers=[layer])

    # Enable reactivity in Marimo:
    layer_widget = mo.ui.anywidget(cast(AnyWidget, layer))
    return layer_widget, map


@app.cell
def _(settings):
    delta_df = pl.scan_delta(
        settings.power_time_series_data_path,
        storage_options=typeddict_to_dict(settings.storage_options),
    )
    return (delta_df,)


@app.cell
def _(delta_df, df, layer_widget, map):
    if layer_widget.selected_index is None:
        right_pane = mo.md(
            """
            ### Select a site
            *Click a dot on the map to view that site's power time series. The map shows
            substations, generation sites, and storage sites together, so the line is not
            always demand.*
            """
        )
    else:
        selected_df = df[layer_widget.selected_index]
        time_series_id = selected_df["time_series_id"].item()

        try:
            # Both reads below are scoped to the selected series first. `power_time_series` is
            # partitioned by `time_series_id`, so scoping first leaves Delta reading one
            # partition instead of all of them, and it defers both reads until a chart is
            # actually about to be drawn.
            series_lf = delta_df.filter(pl.col("time_series_id") == time_series_id)

            # Anchor the rolling window on this series' own newest row, rather than on
            # wall-clock time, so a series that stopped reporting still shows its last
            # RECENT_WINDOW of telemetry instead of an empty chart. No engine on this stack
            # answers a `max` from Parquet statistics, so this reads the series' `time` column
            # in full; the streaming engine keeps the peak memory of that read bounded.
            newest_time: datetime | None = (
                series_lf.select(pl.col("time").max()).collect(engine="streaming").item()
            )
            anchor = newest_time if newest_time is not None else datetime.now(UTC)

            # `anchor - RECENT_WINDOW` is a plain Python datetime, so the filter stays a single
            # comparison against a literal, which Delta and Parquet can push down to the scan.
            filtered_demand = cast(
                pt.DataFrame[PowerTimeSeries],
                series_lf.filter(
                    pl.col("time") > pl.lit(anchor - RECENT_WINDOW).cast(UTC_DATETIME_DTYPE)
                ).collect(),
            )
        except Exception as e:  # noqa: BLE001 — surface any read failure in the pane, never crash.
            right_pane = mo.md(f"{e}")
        else:
            if filtered_demand.height == 0:
                right_pane = mo.md("No data")
            else:
                right_pane = (
                    alt.Chart(filtered_demand)
                    .mark_line()
                    .encode(  # ty: ignore[unresolved-attribute]  # astral-sh/ty#2520
                        x=alt.X(
                            "time:T",
                            axis=alt.Axis(format="%H:%M %b %d"),
                        ),
                        y=alt.Y("power:Q", title=f"Power ({selected_df['units'].item()})"),
                        color=alt.value(DATA_BLUE),
                        tooltip=["time", "power"],
                    )
                    .properties(
                        title=(
                            f"{selected_df['time_series_name'].item()}"
                            f" - {selected_df['substation_type'].item()}"
                            f" - {selected_df['time_series_type'].item()}"
                        ),
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
