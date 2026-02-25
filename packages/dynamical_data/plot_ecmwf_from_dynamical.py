import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import xarray as xr
    import icechunk
    from datetime import datetime, time, date

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
def _():
    INIT_TIME = datetime.combine(date.today(), time())
    INIT_TIME
    return (INIT_TIME,)


@app.cell
def _(ds):
    ds["temperature_2m"].isel(init_time=-1, lead_time=0, ensemble_member=0).plot(cmap="viridis")
    return


@app.cell
def _(INIT_TIME, ds):
    var_name = "downward_short_wave_radiation_flux_surface"
    ds_for_london = (
        ds[var_name]
        .sel(init_time=INIT_TIME)
        .sel(latitude=51.51, longitude=0.13, method="nearest")  # London
    ).load()
    return ds_for_london, var_name


@app.cell
def _(ds_for_london):
    ds_for_london
    return


@app.cell
def _(ds_for_london, var_name):
    smoothed = (
        ds_for_london.swap_dims({"lead_time": "valid_time"})
        .resample(valid_time="1h")
        .interpolate("cubic")
    )

    fig, ax = plt.subplots(figsize=(15, 6))

    ds_for_london.plot.line(
        ax=ax,
        x="valid_time",
        hue="ensemble_member",
        add_legend=False,
        color="gray",
        alpha=0.3,
        linewidth=1,
    )

    ax.set_title(f"ECMWF ENS: {var_name} for London")  # , loc='left', fontweight='bold')
    ax.set_ylabel(var_name)
    ax.set_xlabel("Valid time (day of month)")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig
    return (smoothed,)


@app.cell
def _(smoothed):
    smoothed
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
