import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    from datetime import datetime, timezone
    import polars as pl
    from lonboard import Map, H3HexagonLayer

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
def _(h3_grid):
    ds = dynamical_data.ecmwf_ens.download.download_ecmwf_ens_run(
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
    df = dynamical_data.ecmwf_ens.convert_to_polars.convert_nwp_xarray_dataset_to_polars_dataframe(ds=ds, h3_grid=h3_grid)
    df
    return (df,)


@app.cell
def _(df):
    import altair as alt

    _df = df.filter([pl.col("ensemble_member")==0, pl.col("h3_index")==599148110664433663])

    # replace _df with your data source
    _chart = (
        alt.Chart(_df)
        .mark_line()
        .encode(
            x=alt.X(field='valid_time', type='temporal'),
            y=alt.Y(field='wind_speed_10m', type='quantitative'),
        )
        .properties(
            height=290,
            width='container',
            config={
                'axis': {
                    'grid': False
                }
            }
        )
    )
    _chart
    return


@app.cell
def _(ds):
    valid_time = datetime(2026, 5, 6, 12)
    ds["categorical_precipitation_type_surface"].sel(valid_time=valid_time, ensemble_member=0).plot.imshow()
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

    temperature = filtered_df["categorical_precipitation_type_surface"]
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


if __name__ == "__main__":
    app.run()
