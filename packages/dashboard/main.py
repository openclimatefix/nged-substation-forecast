import marimo
from anywidget import AnyWidget
from contracts.common import UTC_DATETIME_DTYPE

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    from pathlib import Path
    from typing import cast

    from contracts.settings import PROJECT_ROOT, Settings

    from datetime import datetime

    import altair as alt
    import geoarrow.pyarrow as geo_pyarrow
    import lonboard
    import marimo as mo
    import patito as pt
    import plotting.ocf_theme  # noqa: F401 — registers OCF Altair theme as side effect
    import polars as pl
    import pyarrow
    from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
    from contracts.typing_utils import typeddict_to_dict
    from plotting.ocf_theme import BLUE

    ROOT_ENV: Path = PROJECT_ROOT / ".env"
    """Root .env — the local-pipeline config shared with the rest of the app."""
    DASHBOARD_S3_ENV: Path = PROJECT_ROOT / "packages" / "dashboard" / ".env.s3"
    """Git-ignored S3-mode overrides, layered on top of ROOT_ENV when the toggle is 's3'."""

    def _settings_for_source(source: str) -> Settings:
        """Instantiate Settings for the dashboard's selected data source.

        "local" reads only the root .env (the local pipeline, same as the rest of the app).
        "s3" layers packages/dashboard/.env.s3 on top of the root .env, overriding the
        data-path roots and object-store credentials to point at the real S3 buckets, so
        production data can be viewed without restarting marimo.

        Only the data tables follow the toggle: .env.s3 sets DATA_PATH_INTERNAL,
        DATA_PATH_DELIVERY and the DATA_STORE_* credentials. It deliberately does not set
        LOCAL_ARTIFACTS_PATH, so the model cache and production model stay laptop-local in
        both modes. A missing .env.s3 is silently skipped by pydantic-settings, so "s3"
        then falls back to the root .env's local paths (the UI flags this).
        """
        if source == "s3":
            # _env_file is a pydantic-settings builtin kwarg not modelled by ty's synthesised
            # BaseModel __init__; the list layers .env.s3 over the root .env (later file wins).
            return Settings(_env_file=[ROOT_ENV, DASHBOARD_S3_ENV])  # ty: ignore[unknown-argument]
        return Settings()


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
    settings = _settings_for_source(source.value)
    if source.value == "s3" and not DASHBOARD_S3_ENV.exists():
        status = mo.callout(
            mo.md(
                f"No `{DASHBOARD_S3_ENV.name}` found next to `main.py`. Copy "
                f"`{DASHBOARD_S3_ENV.name}.example` to `{DASHBOARD_S3_ENV.name}` and fill in "
                "the S3 buckets and credentials. Falling back to local data."
            ),
            kind="warn",
        )
    else:
        status = mo.md(f"Reading **{source.value}** data from `{settings.nged_data_path}`.")
    status
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
        get_fill_color=[0, 128, 255],
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
    ).filter(
        # Filter to only show recent data. Altair crashes if you try to show too much data.
        pl.col("time") > pl.lit(datetime(2026, 3, 1)).cast(UTC_DATETIME_DTYPE)
    )
    return (delta_df,)


@app.cell
def _(delta_df, df, layer_widget, map):
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
                            "time:T",
                            axis=alt.Axis(format="%H:%M %b %d"),
                        ),
                        y=alt.Y("power:Q", title=f"Power ({selected_df['units'].item()})"),
                        color=alt.value(BLUE),
                        tooltip=["time", "power"],
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
