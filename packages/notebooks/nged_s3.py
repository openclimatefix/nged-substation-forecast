import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")

with app.setup:
    import obstore
    import polars as pl

    from contracts.settings import Settings, PROJECT_ROOT
    from nged_data.read_nged_json import nged_json_to_metadata_df_and_time_series_df

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
    paths = []
    for lst in list(store.list(prefix="timeseries")):
        paths.extend([d["path"] for d in lst])

    paths
    return (paths,)


@app.cell
def _(paths):
    paths_df = (
        pl.DataFrame({"path": paths})
        .with_columns(
            split=pl.col("path")
            .str.split_exact(by="/", n=2)
            .struct.rename_fields(["part0_root", "part1_epochs", "part2_filename"])
        )
        .unnest("split")
    )

    paths_df
    return (paths_df,)


@app.cell
def _(paths_df):
    # Extract the start and end times from the filenames of the form
    # "TimeSeries_3_20160101T003000Z_20260326T083000Z.json"
    # Regex Pattern: Captures two blocks of 8 digits + T + 6 digits + Z
    pattern = r"_(\d{8}T\d{6}Z)_(\d{8}T\d{6}Z)\."

    paths_df2 = (
        # Extract the datetimes in the filenames
        paths_df.with_columns(
            pl.col("part2_filename")
            .str.extract_groups(pattern)
            .struct.rename_fields(["start_time", "end_time"])
            .alias("datetimes"),
        )
        .unnest("datetimes")
        .with_columns(
            pl.col("start_time", "end_time").str.to_datetime("%Y%m%dT%H%M%SZ", time_zone="UTC")
        )
        .sort(["start_time"])
        #
        # Extract the unix epochs from the middle part of the path
        .with_columns(
            pl.col("part1_epochs")
            .str.split("_")
            .list.to_struct(fields=["epoch_start", "epoch_end"])
            .alias("epochs")
        )
        .unnest("epochs")
        .with_columns(
            pl.col("epoch_start", "epoch_end")
            .cast(pl.Int64)
            .cast(pl.Datetime("ms", time_zone="Europe/London"))
        )
    )

    paths_df2
    return (paths_df2,)


@app.cell
def _(paths_df2):
    paths_df2.select((pl.col("epoch_start").dt.replace_time_zone("UTC") == pl.col("start_time")))
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

    time_series_id = 21
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
