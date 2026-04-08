import marimo

__generated_with = "0.22.5"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    return (Path,)


@app.cell
def _(Path):
    archive_path = Path(
        "/home/jack/dev/python/nged-substation-forecast/data/from_NGED_sharepoint/OneDrive_1_4-8-2026/1451606400000_1774512000000/"
    )
    file_listing = list(archive_path.glob("*.json"))
    file_listing
    return


app._unparsable_cell(
    r"""
    df = pl.read_json(file_listing[0])s
    """,
    name="_",
)


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
    df["data"].explode().struct.unnest()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
