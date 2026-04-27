import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")

with app.setup:
    import obstore
    import polars as pl

    from contracts.settings import Settings, PROJECT_ROOT
    from nged_data.read_nged_json import nged_json_to_metadata_df_and_time_series_df


@app.function
def get_nged_s3_store() -> obstore.store.S3Store:

    assert (PROJECT_ROOT / ".env").exists()

    settings = Settings()

    return obstore.store.S3Store.from_url(
        url=settings.nged_s3_bucket_url,
        config={
            "aws_access_key_id": settings.nged_s3_bucket_access_key,
            "aws_secret_access_key": settings.nged_s3_bucket_secret,
        },
    )


@app.cell
def _():
    paths = []
    for listing in get_nged_s3_store().list(prefix="timeseries"):
        paths_for_chunk = [item["path"] for item in listing if item["path"].endswith(".json")]
        paths.extend(paths_for_chunk)

    paths
    return (paths,)


@app.cell
def _(paths):

    df = pl.DataFrame({"path": paths}).with_columns(
        end_time=(
            pl.col("path")
            .str.extract(r"/(\d+)_(\d+)/", 2)  # Capture group 2: the digits after the underscore
            .cast(pl.Int64)
            .cast(pl.Datetime("ms", time_zone="UTC"))
        ),
        time_series_id=(pl.col("path").str.extract(r"TimeSeries_(\d+)", 1).cast(pl.Int32)),
    )
    df
    return


@app.cell
def _(paths_df2):
    (
        paths_df2.group_by(["start_time", "end_time"], maintain_order=True)
        .agg(pl.len().alias("n_timeseries"))
        .with_columns(
            duration=pl.col("end_time") - pl.col("start_time"),
            diff_start_time=pl.col("start_time").diff(),
            overlap=pl.col("end_time").shift(1) - pl.col("start_time"),
        )
    )
    return


@app.cell
def _(store):
    # _path = paths[100]
    _path = "timeseries/1774512000000_1774533600000/TimeSeries_23_20260326T080000Z_20260326T140000Z.json"
    print(_path)
    _result = store.get(_path)

    _bytes = bytes(_result.bytes())
    print("RAW:", str(_bytes[:1000]))
    _bytes = _bytes.replace(b": NaN", b": null")

    _metadata_df, _time_series_df = nged_json_to_metadata_df_and_time_series_df(_bytes)
    return


@app.cell
def _(paths, store):
    from collections import defaultdict

    all_metadata = defaultdict(list)
    all_time_series = defaultdict(list)
    failures = []

    for i, _path in enumerate(paths):
        print(f"{i:03d}", end="\r")
        if not _path.endswith(".json"):
            continue

        _result = store.get(_path)
        _bytes = bytes(_result.bytes())

        try:
            _metadata_df, _time_series_df = nged_json_to_metadata_df_and_time_series_df(_bytes)
        except Exception as e:
            failures.append((_path, e))
            continue

        _time_series_id = _metadata_df["time_series_id"].item()

        all_metadata[_time_series_id].append(_metadata_df)
        all_time_series[_time_series_id].append(_time_series_df)
    return all_metadata, all_time_series


@app.cell
def _():
    # failures
    return


@app.cell
def _(all_time_series):
    ts_list = []
    for _lst in all_time_series.values():
        ts_list.extend(_lst)

    ts_df = pl.concat(ts_list).sort(["time_series_id", "time"])
    return (ts_df,)


@app.cell
def _(ts_df):
    ts_df
    return


@app.cell
def _(all_metadata, ts_df):
    from datetime import datetime, timezone, timedelta

    time_series_id = 23
    start_date = datetime(2026, 3, 7, tzinfo=timezone.utc)
    duration = timedelta(weeks=10)

    _metadata = all_metadata[time_series_id][-1]

    chart = (
        ts_df.filter(
            pl.col("time_series_id") == time_series_id,
            start_date < pl.col("time"),
            pl.col("time") < start_date + duration,
        )
        .plot.line(x="time", y="power")
        .properties(
            width=800,
            title=f"{time_series_id=} | "
            + _metadata["time_series_name"].item()
            + " | "
            + _metadata["time_series_type"].item()
            + " | "
            + _metadata["substation_type"].item(),
        )
    )

    chart.encoding.y.title = _metadata["units"].item()

    chart
    return


@app.cell
def _(ts_df):
    ts_df.filter(pl.col("time_series_id") == 23)
    return


@app.cell
def _(ts_df):
    ts_df.filter(pl.col("time_series_id") == 23, pl.col("power") > 1).sort("time")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
