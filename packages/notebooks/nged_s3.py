import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")

with app.setup:
    import obstore
    from contracts.config import Settings, PROJECT_ROOT
    import polars as pl
    from io import BytesIO

    assert (PROJECT_ROOT / ".env").exists()

    settings = Settings()


@app.cell
def _():
    store = obstore.store.S3Store.from_url(
        url=str(settings.NGED_S3_BUCKET_URL),
        config={
            "aws_access_key_id": settings.NGED_S3_BUCKET_ACCESS_KEY.get_secret_value(),
            "aws_secret_access_key": settings.NGED_S3_BUCKET_SECRET.get_secret_value(),
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
def _():
    return


if __name__ == "__main__":
    app.run()
