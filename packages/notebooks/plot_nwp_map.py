import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    from datetime import datetime, timezone
    import polars as pl
    from lonboard import Map, H3HexagonLayer
    import altair as alt
    import numpy as np

    from contracts.settings import Settings, PROJECT_ROOT
    import dynamical_data.ecmwf_ens.download
    import dynamical_data.ecmwf_ens.convert_to_polars

    SETTINGS = Settings()


@app.cell
def _():
    h3_grid = pl.read_parquet(PROJECT_ROOT / SETTINGS.h3_grid_weights_path)
    h3_grid
    return (h3_grid,)


@app.cell
def _():
    NWP_INIT_TIME = datetime(2026, 5, 13, tzinfo=timezone.utc)
    return (NWP_INIT_TIME,)


@app.cell
def _(NWP_INIT_TIME, h3_grid):
    ds = dynamical_data.ecmwf_ens.download.download_ecmwf_ens_run(
        nwp_init_time=NWP_INIT_TIME,
        h3_grid=h3_grid,
    )
    return (ds,)


@app.cell
def _(ds):
    ds
    return


@app.cell
def _(ds, h3_grid):
    df = dynamical_data.ecmwf_ens.convert_to_polars.convert_nwp_xarray_dataset_to_polars_dataframe(ds=ds, h3_grid=h3_grid)
    df
    return (df,)


@app.cell
def _(df):
    NWP_VAR_TO_PLOT = "temperature_2m"

    _df = df.filter([pl.col("h3_index") == 599148110664433663])

    _chart = (
        alt.Chart(_df)
        .mark_line(
            strokeWidth=1,
            opacity=0.3,
        )
        .encode(
            x=alt.X(field="valid_time", type="temporal"),
            y=alt.Y(field=NWP_VAR_TO_PLOT, type="quantitative"),
            detail="ensemble_member:N",
        )
        .properties(height=290, width="container")
    )
    _chart
    return (NWP_VAR_TO_PLOT,)


@app.cell
def _(NWP_VAR_TO_PLOT, ds):
    VALID_TIME_TO_PLOT = datetime(2026, 5, 13, hour=12, tzinfo=timezone.utc)
    ENS_MEMBER_TO_PLOT = 0

    ds[NWP_VAR_TO_PLOT].sel(
        valid_time=VALID_TIME_TO_PLOT.replace(tzinfo=None), ensemble_member=ENS_MEMBER_TO_PLOT
    ).plot.imshow()
    return ENS_MEMBER_TO_PLOT, VALID_TIME_TO_PLOT


@app.cell
def _(ENS_MEMBER_TO_PLOT, VALID_TIME_TO_PLOT, df):
    filtered_df = df.filter(
        [
            pl.col("ensemble_member") == ENS_MEMBER_TO_PLOT,
            pl.col("valid_time") == VALID_TIME_TO_PLOT,
        ]
    )
    filtered_df
    return (filtered_df,)


@app.cell
def _(NWP_VAR_TO_PLOT, filtered_df):
    from palettable.matplotlib import Viridis_20
    from lonboard.colormap import apply_continuous_cmap

    values = filtered_df[NWP_VAR_TO_PLOT]
    min_bound = values.min()
    max_bound = values.max() - min_bound

    normalized = (values - min_bound) / max_bound
    normalized
    return Viridis_20, apply_continuous_cmap, normalized


@app.cell
def _(Viridis_20, apply_continuous_cmap, filtered_df, normalized):
    Map(
        H3HexagonLayer(
            filtered_df.select("h3_index"),
            get_hexagon=filtered_df["h3_index"],
            get_fill_color=apply_continuous_cmap(normalized, Viridis_20, alpha=1),
            opacity=1,
        )
    )
    return


if __name__ == "__main__":
    app.run()
