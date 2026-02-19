import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import polars as pl
    import h3.api.numpy_int as h3
    import polars_h3 as plh3
    import shapely.geometry
    import shapely.wkt
    from typing import Final
    from lonboard import Map, H3HexagonLayer
    import numpy as np

    import xarray as xr
    import icechunk


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
    # ECMWF ENS has a res of 0.25°.
    # For GB, a 0.25° grid box is approx 28 km north-south (lat) x 16 km east-west (lon) ~= 450 km².
    # H3 average hexagon areas (from https://h3geo.org/docs/core-library/restable/#average-area-in-km2):
    #   res 4 = 1,770 km² (far too coarse).
    #   res 5 =   253 km² (the right choice).
    H3_RES: Final[int] = 5

    cells = h3.geo_to_cells(shape, res=H3_RES)

    df = pl.DataFrame({"h3_index": list(cells)}, schema={"h3_index": pl.UInt64}).sort("h3_index")
    df
    return H3_RES, df


@app.cell
def _(H3_RES: Final[int], df):
    # Verify we caught the Isles of Scilly:
    _scilly_hex = h3.latlng_to_cell(lat=49.9, lng=-6.3, res=H3_RES)
    assert _scilly_hex in df["h3_index"]
    return


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

    # Load dataset lazily (metadata only)
    ds = xr.open_zarr(session.store, chunks=None)

    ds
    return (ds,)


@app.cell
def _():
    # TODO:
    # - Optimise: Load a Zarr chunk at a time, and then convert that chunk to a DataFrame. Each ECMWF chunk contains 1 init time, all lead times, and all ens members.
    return


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
def _(ds, max_lat, max_lng, min_lat, min_lng):
    # - Crop the NWP data spatially using the min & max lats and lngs from the Polars H3 dataframe.
    ds_cropped = ds.sel(
        # Latitude coords are in _descending_ order or the Northern hemisphere!
        latitude=slice(max_lat.item(), min_lat.item()),
        longitude=slice(min_lng.item(), max_lng.item()),
    )

    ds_cropped
    return (ds_cropped,)


@app.cell
def _(ds_cropped):
    # - Convert to numpy array
    # - Flatten numpy array
    flat_array = (
        ds_cropped["temperature_2m"]
        .isel(init_time=-1, lead_time=0, ensemble_member=0)
        .values.ravel()
    )
    flat_array
    return (flat_array,)


@app.cell
def _(ds_cropped):
    # - Put into Polars DataFrame with appropriate lat and lng (perhaps by flattening the cropped xarray coords arrays?)
    lat_grid, lon_grid = np.meshgrid(
        ds_cropped.latitude.values, ds_cropped.longitude.values, indexing="ij"
    )
    lat_grid
    return lat_grid, lon_grid


@app.cell
def _(flat_array, lat_grid, lon_grid):
    df_nwp = pl.DataFrame(
        {"temperature_2m": flat_array, "longitude": lon_grid.ravel(), "latitude": lat_grid.ravel()}
    )
    df_nwp
    return (df_nwp,)


@app.cell
def _(df_nwp, df_with_counts):
    # - Join h3_res7_grid_cell with the actual NWP data, to end up with a dataframe that has `proportion` and the raw NWP value
    joined = df_with_counts.join(
        df_nwp,
        left_on=["nwp_lng", "nwp_lat"],
        right_on=["longitude", "latitude"],
    )
    joined
    return (joined,)


@app.cell
def _(joined):
    # - Multiply `proportion` with each NWP value.
    # - Groupby h3_index, and sum each NWP value column
    df_of_h3_and_temperature = (
        joined.with_columns(temperature_2m=pl.col("temperature_2m") * pl.col("proportion"))
        .group_by("h3_index")
        .agg(pl.col("temperature_2m").sum())
    )
    df_of_h3_and_temperature
    return (df_of_h3_and_temperature,)


@app.cell
def _():
    from lonboard.colormap import apply_continuous_cmap

    return (apply_continuous_cmap,)


@app.cell
def _(df_of_h3_and_temperature):
    temperature = df_of_h3_and_temperature["temperature_2m"]
    min_bound = temperature.min()
    max_bound = temperature.max() - min_bound

    normalized = (temperature - min_bound) / max_bound
    normalized
    return (normalized,)


@app.cell
def _():
    from palettable.matplotlib import Viridis_20

    return (Viridis_20,)


@app.cell
def _(Viridis_20, apply_continuous_cmap, df_of_h3_and_temperature, normalized):
    Map(
        H3HexagonLayer(
            df_of_h3_and_temperature,
            get_hexagon=df_of_h3_and_temperature["h3_index"],
            get_fill_color=apply_continuous_cmap(normalized, Viridis_20, alpha=0.7),
            opacity=0.8,
        )
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
