import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import dynamical_catalog
    import numpy as np


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
    return


if __name__ == "__main__":
    app.run()
