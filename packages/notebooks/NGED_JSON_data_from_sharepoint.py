import marimo

__generated_with = "0.22.5"
app = marimo.App(width="full")

with app.setup:
    import polars as pl
    from pathlib import Path
    from datetime import datetime, timezone


@app.cell
def _():
    archive_path = Path(
        "/home/jack/dev/python/nged-substation-forecast/data/from_NGED_sharepoint/OneDrive_1_4-8-2026/1451606400000_1774512000000/"
    )
    file_listing = list(archive_path.glob("*.json"))
    file_listing
    return (file_listing,)


@app.cell
def _(file_listing):
    df = pl.read_json(file_listing[0])
    return (df,)


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df):
    metadata_columns = [
        "TimeSeriesID",
        "TimeSeriesName",
        "TimeSeriesType",
        "Units",
        "LicenceArea",
        "SubstationNumber",
        "SubstationType",
        "Latitude",
        "Longitude",
        "Information",
        "Area",
    ]

    df[metadata_columns]
    return


@app.cell
def _(df):
    timeseries = (
        df["data"]
        .explode()
        .struct.unnest()
        .select(["value", "endTime"])
        .with_columns(endTime=pl.col("endTime").str.to_datetime(time_zone="UTC"))
    )
    timeseries
    return (timeseries,)


@app.cell
def _(timeseries):
    filtered = timeseries.filter(pl.col("endTime") > datetime(2026, 3, 20, tzinfo=timezone.utc))
    filtered
    return (filtered,)


@app.cell
def _(filtered):
    filtered.plot.line(x="endTime", y="value")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
