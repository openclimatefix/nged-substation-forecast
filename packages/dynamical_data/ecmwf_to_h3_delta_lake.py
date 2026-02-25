import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")

with app.setup:
    from typing import Final

    import h3.api.numpy_int as h3
    import icechunk
    import marimo as mo
    import numpy as np
    import polars as pl
    import polars_h3 as plh3
    import shapely.geometry
    import shapely.wkt
    import xarray as xr
    from lonboard import H3HexagonLayer, Map


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Fetch geometry of GB
    """)
    return


@app.cell
def _():
    # TODO: Move all this geo code into a new `package/geo_utils` package.

    # Downloaded from https://onsdigital.github.io/uk-topojson/ for the year 2025.
    with open("england_scotland_wales.geojson") as f:
        file_contents = f.read()
    return (file_contents,)


@app.cell
def _(file_contents):
    shape = shapely.from_geojson(file_contents)

    # Buffer by 0.25 degrees (approx 20km) to catch islands/coasts.
    # This turns the "rough" map into a "safe" container.
    # This takes about 20 seconds to compute!
    shape = shape.buffer(0.25)

    shape
    return (shape,)


@app.cell
def _(shape):
    # Which H3 resolution to use?
    # ECMWF ENS has a horizontal resolutions of 0.25°.
    # For GB, a 0.25° grid box is approx 28 km north-south (lat) x 16 km east-west (lon) ~= 450 km².
    # H3 average hexagon areas (from https://h3geo.org/docs/core-library/restable/#average-area-in-km2):
    #   res 4 = 1,770 km² (far too coarse).
    #   res 5 =   253 km² (the right choice).
    H3_RES: Final[int] = 5

    cells = h3.geo_to_cells(shape, res=H3_RES)

    df = pl.DataFrame({"h3_index": list(cells)}, schema={"h3_index": pl.UInt64}).sort("h3_index")

    # Verify we caught the Isles of Scilly:
    _scilly_hex = h3.latlng_to_cell(lat=49.9, lng=-6.3, res=H3_RES)
    assert _scilly_hex in df["h3_index"]

    df
    return (df,)


@app.cell
def _(df):
    layer = H3HexagonLayer(
        df,
        get_hexagon=df["h3_index"],
        opacity=0.2,
    )
    Map(layer)
    return


@app.cell
def _(df):
    df_with_children = df.with_columns(h3_res7=plh3.cell_to_children("h3_index", 7)).explode(
        "h3_res7"
    )
    return (df_with_children,)


@app.cell
def _(df_with_children):
    # Instead of spatial join, we compute the NWP Key mathematically
    GRID_SIZE = 0.25

    # TODO: Need to think if ECMWF's grid boxes are centered on these coords, and how that interacts with our code below:

    df_with_grid_x_y = df_with_children.with_columns(
        # Bin every child into an NWP box by snapping to the nearest 0.25 degree
        nwp_grid_box=pl.struct(
            nwp_lat=(plh3.cell_to_lat("h3_res7") / GRID_SIZE).floor() * GRID_SIZE,
            nwp_lng=(plh3.cell_to_lng("h3_res7") / GRID_SIZE).floor() * GRID_SIZE,
        ),
    )

    df_with_grid_x_y
    return (df_with_grid_x_y,)


@app.cell
def _(df_with_grid_x_y):
    df_with_counts = (
        df_with_grid_x_y.group_by("h3_index")
        .agg(grid_cell_counts=pl.col("nwp_grid_box").value_counts())
        .with_columns(
            total=pl.col("grid_cell_counts").list.agg(pl.element().struct.field("count").sum())
        )
        .explode("grid_cell_counts")
        .unnest("grid_cell_counts")
        .unnest("nwp_grid_box")
        .with_columns(proportion=pl.col.count / pl.col.total)
    )
    df_with_counts
    return (df_with_counts,)


@app.cell
def _():
    storage = icechunk.s3_storage(
        bucket="dynamical-ecmwf-ifs-ens",
        prefix="ecmwf-ifs-ens-forecast-15-day-0-25-degree/v0.1.0.icechunk/",
        region="us-west-2",
        anonymous=True,
    )
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")

    ds = xr.open_zarr(
        session.store,
        chunks=None,  # Don't use dask.
    )

    ds
    return (ds,)


@app.cell
def _(df_with_counts):
    min_lat, max_lat, min_lng, max_lng = df_with_counts.select(
        min_lat=pl.col("nwp_lat").min(),
        max_lat=pl.col("nwp_lat").max(),
        min_lng=pl.col("nwp_lng").min(),
        max_lng=pl.col("nwp_lng").max(),
    )
    return max_lat, max_lng, min_lat, min_lng


@app.cell
def _(df_with_counts, ds, max_lat, max_lng, min_lat, min_lng):
    import concurrent.futures

    def download_array(var_name: str) -> dict[str, xr.DataArray]:
        return {var_name: ds_cropped[var_name].compute()}

    for init_time in ds.init_time.values:
        print(init_time)

        # Crop the NWP data spatially using the min & max lats and lngs from the Polars H3 dataframe.
        ds_cropped = ds.sel(
            # Latitude coords are in _descending_ order or the Northern hemisphere!
            latitude=slice(max_lat.item(), min_lat.item()),
            longitude=slice(min_lng.item(), max_lng.item()),
            init_time=init_time,
        )

        data_arrays: dict[str, xr.DataArray] = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(download_array, name) for name in ds_cropped.data_vars]
            for future in concurrent.futures.as_completed(futures):
                data_arrays.update(future.result())

        loaded_ds = xr.Dataset(data_arrays)
        del data_arrays
        loaded_ds

        # TODO: Don't compute this every loop
        lat_grid, lon_grid = np.meshgrid(
            loaded_ds.latitude.values, loaded_ds.longitude.values, indexing="ij"
        )

        dfs = []

        for lead_time in loaded_ds.lead_time.values:
            for ensemble_member in loaded_ds.ensemble_member.values:
                loaded_cropped_ds = loaded_ds.sel(
                    lead_time=lead_time, ensemble_member=ensemble_member
                )
                nwp_data = {
                    var_name: var_array.values.ravel()
                    for var_name, var_array in loaded_cropped_ds.items()
                }
                nwp_data.update(
                    {
                        "longitude": lon_grid.ravel(),
                        "latitude": lat_grid.ravel(),
                    }
                )
                _nwp_df = pl.DataFrame(nwp_data)

                # - Join h3_res7_grid_cell with the actual NWP data, to end up with a dataframe that has `proportion` and the raw NWP value
                joined = df_with_counts.join(
                    _nwp_df,
                    left_on=["nwp_lng", "nwp_lat"],
                    right_on=["longitude", "latitude"],
                )

                all_nwp_vars: list[str] = list(loaded_cropped_ds.data_vars.keys())  # type: ignore[invalid-assignment]

                # We need to handle categorical values differently:
                categorical_nwp_vars = ["categorical_precipitation_type_surface"]
                numeric_nwp_vars = [var for var in all_nwp_vars if var not in categorical_nwp_vars]

                dtypes = {numeric_nwp_var: pl.Float32 for numeric_nwp_var in numeric_nwp_vars}
                dtypes.update(
                    {categorical_nwp_var: pl.UInt8 for categorical_nwp_var in categorical_nwp_vars}
                )

                joined = (
                    joined.with_columns(pl.col(numeric_nwp_vars) * pl.col("proportion"))
                    .group_by("h3_index")
                    .agg(pl.col(numeric_nwp_vars).sum(), pl.col(categorical_nwp_vars).mode())
                    .cast(dtypes)
                    .with_columns(
                        lead_time=pl.duration(seconds=lead_time / np.timedelta64(1, "s")),
                        ensemble_member=pl.lit(ensemble_member, dtype=pl.UInt8),
                        init_time=pl.lit(init_time),
                    )
                )

                dfs.append(joined)

        nwp_df = pl.concat(dfs)
        nwp_df = nwp_df.sort(by=["ensemble_member", "h3_index", "lead_time"])

        break  # TODO(Jack) REMOVE THIS!

        nwp_df.write_parquet(
            f"data/{np.datetime_as_string(init_time, unit='h')}.parquet",
            compression="zstd",
            compression_level=10,
            statistics="full",
        )
    return (nwp_df,)


@app.cell
def _():
    d = {"a": 1}
    d.update({"b": 2})
    d
    return


@app.cell
def _(nwp_var_names):
    nwp_var_names.remove("categorical_precipitation_type_surface")
    return


@app.cell
def _(nwp_var_names):
    nwp_var_names
    return


@app.cell
def _():
    from lonboard.colormap import apply_continuous_cmap

    return (apply_continuous_cmap,)


@app.cell
def _(nwp_df):
    selection = nwp_df.filter(
        pl.col.ensemble_member == 0, pl.col.lead_time == pl.duration(days=4.5)
    )

    temperature = selection["temperature_2m"]
    min_bound = temperature.min()
    max_bound = temperature.max() - min_bound

    normalized = (temperature - min_bound) / max_bound
    return normalized, selection


@app.cell
def _():
    from palettable.matplotlib import Viridis_20  # type: ignore[unresolved-import]

    return (Viridis_20,)


@app.cell
def _(Viridis_20, apply_continuous_cmap, normalized, selection):
    Map(
        H3HexagonLayer(
            selection,
            get_hexagon=selection["h3_index"],
            get_fill_color=apply_continuous_cmap(normalized, Viridis_20, alpha=0.7),
            opacity=0.8,
        )
    )
    return


@app.cell
def _(selection):
    selection
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
