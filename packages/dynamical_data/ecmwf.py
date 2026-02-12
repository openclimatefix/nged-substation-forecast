import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import xarray as xr
    import icechunk

    import matplotlib.pyplot as plt


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Lazily open ECMWF Icechunk dataset hosted by Dynamical.org
    """)
    return


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

    # Define Great Britain Bounding Box
    # Lat: North-to-South (61 down to 49)
    # Lon: West-to-East (-9 to 2)
    ds = ds.sel(
        latitude=slice(59, 49.5),  # Note: Max -> Min order for ECMWF
        longitude=slice(-7.5, 2),
    )

    ds
    return (ds,)


@app.cell
def _(ds):
    ds["temperature_2m"].sel(init_time="2026-02-12T00", lead_time="12h", ensemble_member=1).plot()
    return


@app.cell
def _(ds):
    ds_for_london = (
        ds["wind_v_100m"]
        .sel(init_time="2026-02-12T00")
        .sel(latitude=51.51, longitude=0.13, method="nearest")  # London
    ).load()
    return (ds_for_london,)


@app.cell
def _(ds_for_london):
    ds_for_london
    return


@app.cell
def _(ds_for_london):
    smoothed = (
        ds_for_london.swap_dims({"lead_time": "valid_time"})
        .resample(valid_time="1h")
        .interpolate("cubic")
    )

    fig, ax = plt.subplots(figsize=(15, 6))

    smoothed.plot.line(
        ax=ax,
        x="valid_time",
        hue="ensemble_member",
        add_legend=False,
        color="gray",
        alpha=0.3,
        linewidth=1,
    )

    ax.set_title("ECMWF ENS: Wind for London")  # , loc='left', fontweight='bold')
    ax.set_ylabel("wind")
    ax.set_xlabel("Valid time (day of month)")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
