import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import dynamical_catalog
    import numpy as np

    import geo.h3
    import h3.api.basic_int as h3

    import polars as pl
    import polars_h3 as plh3


@app.cell
def _():
    ds = dynamical_catalog.open("ecmwf-ifs-ens-forecast-15-day-0-25-degree", chunks=None)
    ds
    return (ds,)


@app.cell
def _(ds):
    lat_grid, lon_grid = np.meshgrid(
        ds.latitude.values.astype(np.float32), ds.longitude.values.astype(np.float32), indexing="ij"
    )
    return (lon_grid,)


@app.cell
def _(lon_grid):
    lon_grid.ravel()
    return


@app.cell
def _(lon_grid):
    lon_grid
    return


@app.cell
def _():
    boundary = geo.h3.uk_boundary()
    return (boundary,)


@app.cell
def _(boundary):
    boundary
    return


@app.cell
def _(boundary):
    cells = h3.geo_to_cells(boundary, res=5)
    return (cells,)


@app.cell
def _(cells):
    type(cells)
    return


@app.cell
def _(cells):
    df = pl.DataFrame({"h3_index": cells}, schema={"h3_index": pl.UInt64}).sort("h3_index")
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    grid_size = 0.25
    child_h3_res = 7

    half_grid_size = grid_size / 2

    weights_df = (
        df.with_columns(child_h3=plh3.cell_to_children("h3_index", child_h3_res))
        .explode("child_h3")
        .with_columns(
            # GRID SNAPPING FORMULA:
            # The half-grid offset binning `((lat + grid_size/2) / grid_size).floor() * grid_size`
            # ensures that points are snapped to the *closest* grid center rather than
            # the bottom-left corner of the grid cell. Adding `grid_size/2` before
            # flooring shifts the bin boundaries so that the grid points (0, 0.25, 0.5, etc.)
            # are at the center of each bin.
            nwp_lat=((plh3.cell_to_lat("child_h3") + half_grid_size) / grid_size).floor() * grid_size,
            nwp_lon=((plh3.cell_to_lng("child_h3") + half_grid_size) / grid_size).floor() * grid_size,
        )
        .group_by(["h3_index", "nwp_lat", "nwp_lon"])
        .len(name="n_h3_children_in_nwp_grid_cell")
        .with_columns(
            total_children_per_h3_hexagon=pl.col("n_h3_children_in_nwp_grid_cell").sum().over("h3_index"),
        )
        .with_columns(
            proportion=pl.col("n_h3_children_in_nwp_grid_cell") / pl.col("total_children_per_h3_hexagon"),
        )
    )
    return child_h3_res, weights_df


@app.cell
def _(weights_df):
    weights_df
    return


@app.cell
def _(child_h3_res, df):
    (df.with_columns(child_h3=plh3.cell_to_children("h3_index", child_h3_res)).explode("child_h3"))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
