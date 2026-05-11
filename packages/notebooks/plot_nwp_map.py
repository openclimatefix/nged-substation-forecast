import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    from datetime import datetime, timezone
    import polars as pl
    from lonboard import Map, H3HexagonLayer

    from contracts.settings import Settings, PROJECT_ROOT
    import dynamical_data.ecmwf_ens

    SETTINGS = Settings()


@app.cell
def _():
    h3_grid = pl.read_parquet(PROJECT_ROOT / SETTINGS.h3_grid_weights_path)
    h3_grid
    return (h3_grid,)


@app.cell
def _(h3_grid):
    ds = dynamical_data.ecmwf_ens.download_ecmwf_ens_run(
        nwp_init_time=datetime(2026, 5, 1, tzinfo=timezone.utc),
        h3_grid=h3_grid,
    )
    return (ds,)


@app.cell
def _(ds):
    ds
    return


@app.cell
def _(ds, h3_grid):
    df = dynamical_data.ecmwf_ens.convert_nwp_xarray_dataset_to_polars_dataframe(ds=ds, h3_grid=h3_grid)
    df
    return (df,)


@app.cell
def _(ds):
    valid_time = datetime(2026, 5, 3, 12)
    ds["temperature_2m"].sel(valid_time=valid_time, ensemble_member=0).plot.imshow()
    return (valid_time,)


@app.cell
def _(df, valid_time):
    filtered_df = df.filter(ensemble_member=0, valid_time=valid_time.replace(tzinfo=timezone.utc))
    filtered_df
    return (filtered_df,)


@app.cell
def _(filtered_df):
    from palettable.matplotlib import Viridis_20
    from lonboard.colormap import apply_continuous_cmap

    temperature = filtered_df["temperature_2m"]
    min_bound = temperature.min()
    max_bound = temperature.max() - min_bound

    normalized = (temperature - min_bound) / max_bound
    normalized
    return Viridis_20, apply_continuous_cmap, normalized


@app.cell
def _(Viridis_20, apply_continuous_cmap, filtered_df, normalized):
    Map(
        H3HexagonLayer(
            filtered_df,
            get_hexagon=filtered_df["h3_index"],
            get_fill_color=apply_continuous_cmap(normalized, Viridis_20, alpha=1),
            opacity=1,
        )
    )
    return


@app.cell
def _():
    import numpy as np
    ndt = np.datetime64(datetime(2024, 1, 1))
    return (ndt,)


@app.cell
def _(ndt):
    pl.DataFrame(
        {"a": [0, 1, 2], "b": 10, "c": ndt.astype("datetime64[s]").astype(int), "e": [10, 20, 30]},
        schema_overrides={"b": pl.UInt8},
    ).with_columns(
        pl.from_epoch(pl.col(["c"])).cast(pl.Datetime(time_zone="utc"))
    )
    return


@app.cell
def _(ndt):
    ndt.astype("datetime64[s]")
    return


@app.cell
def _(ds):
    ds.sel(ensemble_member=0)["ensemble_member"].values.item()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
