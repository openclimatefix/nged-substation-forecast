import marimo

__generated_with = "0.22.5"
app = marimo.App(width="full")

with app.setup:
    import obstore
    from contracts.settings import Settings, PROJECT_ROOT
    import polars as pl
    from io import BytesIO

    assert (PROJECT_ROOT / ".env").exists()

    settings = Settings()


@app.cell
def _():
    store = obstore.store.S3Store.from_url(
        url=settings.nged_s3_bucket_url,
        config={
            "aws_access_key_id": settings.nged_s3_bucket_access_key,
            "aws_secret_access_key": settings.nged_s3_bucket_secret,
        },
    )
    store
    return (store,)


@app.cell
def _(store):
    list(store.list())
    return


@app.cell
def _(store):
    _result = store.get(
        "curtailment/1772610803000_1772632403000/Curtailment1_1772610803000_1772632403000.parquet"
    )
    curtailment_df = pl.read_parquet(BytesIO(_result.bytes()))
    curtailment_df
    return (curtailment_df,)


@app.cell
def _(curtailment_df):
    curtailment_df.select(pl.col("data").explode().struct.unnest())
    return


@app.cell
def _(store):
    _result = store.get(
        "timeseries/1772610682000_1772632282000/TimeSeries_37_1772610682000_1772632282000.parquet"
    )
    timeseries_df = pl.read_parquet(BytesIO(_result.bytes()))
    timeseries_df
    return (timeseries_df,)


@app.cell
def _(timeseries_df):
    timeseries_df.explode("data")
    return


@app.cell
def _(timeseries_df):
    timeseries_df.select(pl.col("data").explode().struct.unnest())
    return


@app.cell
def _(files):
    json_files = [f.path for f in files if f.path.endswith(".json")]
    print(f"Found {len(json_files)} JSON files.")
    for json_file in json_files[:5]:
        print(json_file)
    return


app._unparsable_cell(
    r"""
    if json_files:
        # Read the first JSON file
        _result = store.get(json_files[0])
        # Assuming it's a standard JSON file
        json_df = pl.read_json(BytesIO(_result.bytes()))
        print(f"Successfully read {json_files[0]}")
        print(json_df.head())
        return (json_df,)
    """,
    name="_",
)


if __name__ == "__main__":
    app.run()
