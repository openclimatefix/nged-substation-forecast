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


@app.cell
def _(ds, h3_grid):
    import xarray as xr
    import numpy as np

    ds_chunk = ds.isel(ensemble_member=0, lead_time=0)

    lat_grid, lon_grid = np.meshgrid(
        ds_chunk.latitude.values.astype(np.float32),
        ds_chunk.longitude.values.astype(np.float32),
        indexing="ij",
    )

    # Prepare data dictionary
    data_dict = {"latitude": lat_grid.ravel(), "longitude": lon_grid.ravel()}

    # Add data variables
    for var_name in ds_chunk.data_vars:
        data_dict[str(var_name)] = ds_chunk[var_name].values.ravel()

    # Create Polars DataFrame.
    nwp_df = pl.DataFrame(data_dict)

    joined = h3_grid.join(nwp_df, left_on=["nwp_lon", "nwp_lat"], right_on=["longitude", "latitude"], how="left")

    # Aggregate to H3 resolution 5.


    # TODO: I think a lot of this aggregation logic is over-complex, and possibly is filling things
    # with zeros when they shouldn't be.
    def weight_sum_expr(x: str) -> pl.Expr:
        return (
            pl.when(pl.col(x).fill_nan(None).is_not_null())
            .then(pl.col("proportion"))
            .otherwise(0.0)  # TODO: This zero makes me nervous!
            .sum()
        )


    # Define variables for aggregation
    all_nwp_vars = [str(v) for v in ds_chunk.data_vars]
    numeric_vars = [v for v in all_nwp_vars if v not in ["categorical_precipitation_type_surface"]]

    # Aggregate numeric variables
    agg_exprs = [
        pl.when(weight_sum_expr(x) > 0)
        .then((pl.col(x).fill_nan(None) * pl.col("proportion")).sum() / weight_sum_expr(x))
        .otherwise(None)
        .cast(pl.Float32)
        .alias(x)
        for x in numeric_vars
    ]

    processed = joined.group_by("h3_index").agg(agg_exprs)

    # WEIGHTED CATEGORICAL AGGREGATION:
    for categorical_var_name in ["categorical_precipitation_type_surface"]:
        df_cat = (
            joined.group_by(["h3_index", categorical_var_name])
            .agg(pl.col("proportion").sum().alias("weight"))
            .sort(["h3_index", "weight"])
            .group_by("h3_index")
            .agg(pl.col(categorical_var_name).last().cast(pl.UInt8))
        )
        processed = processed.join(df_cat, on="h3_index", how="left")
    return (processed,)


@app.cell
def _(processed):
    processed.sort("h3_index")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
