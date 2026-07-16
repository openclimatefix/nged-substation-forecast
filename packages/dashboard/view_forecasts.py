import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import polars as pl
    from contracts.power_schemas import TimeSeriesMetadata
    from contracts.typing_utils import typeddict_to_dict
    from dashboard.data_source import settings_for_source, source_status_message
    from dashboard.forecast_chart import PLOT_HISTORY, PLOT_HORIZON, build_view_forecast_chart
    from deltalake import DeltaTable


@app.cell
def _():
    mo.md("""
    # View forecasts

    Pick a time series and a forecast run; the plot shows every forecast ensemble member
    (thin grey lines) against the observed power (thick blue line), from 24 hours before
    the forecast init time to 14 days after it.
    """)
    return


@app.cell
def _():
    source = mo.ui.radio(
        options=["local", "s3"],
        value="local",
        label="Data source",
        inline=True,
    )
    source
    return (source,)


@app.cell
def _(source):
    settings = settings_for_source(source.value)
    status_message, is_warning = source_status_message(source.value, settings)
    mo.callout(mo.md(status_message), kind="warn") if is_warning else mo.md(status_message)
    return (settings,)


@app.cell
def _(settings):
    metadata_df = TimeSeriesMetadata.validate(
        pl.read_parquet(
            settings.metadata_path, storage_options=typeddict_to_dict(settings.storage_options)
        )
    )
    # Sorting by type first clusters e.g. all the PV sites / all the primaries together, so one
    # kind of series is easy to find in the dropdown.
    series_options = {
        f"{row['time_series_type']} · {row['time_series_name']} ({row['units']}) — id {row['time_series_id']}": row[
            "time_series_id"
        ]
        for row in metadata_df.sort("time_series_type", "time_series_name").iter_rows(named=True)
    }
    series_picker = mo.ui.dropdown(
        options=series_options,
        value=next(iter(series_options)),
        label="Time series",
        searchable=True,
    )
    return metadata_df, series_picker


@app.cell
def _(settings):
    # Delta partition metadata (no data scan) lists the available (experiment_name, fold_id)
    # pairs. If the table is missing entirely (e.g. a fresh local checkout), stop here — the
    # data-source radio above stays usable so the user can switch to S3.
    try:
        forecast_partitions = DeltaTable(
            settings.power_forecasts_data_path,
            storage_options=typeddict_to_dict(settings.storage_options),
        ).partitions()
    except Exception as error:
        forecast_partitions = []
        _load_error = f": `{error}`"
    else:
        _load_error = "."
    mo.stop(
        not forecast_partitions,
        mo.callout(
            mo.md(f"No forecasts found at `{settings.power_forecasts_data_path}`{_load_error}"),
            kind="warn",
        ),
    )
    return (forecast_partitions,)


@app.cell
def _(forecast_partitions):
    fold_ids = sorted({p["fold_id"] for p in forecast_partitions})
    fold_picker = mo.ui.dropdown(
        options=fold_ids,
        value="live" if "live" in fold_ids else fold_ids[0],
        label="Fold",
    )
    return (fold_picker,)


@app.cell
def _(fold_picker, forecast_partitions):
    experiment_names = sorted(
        {p["experiment_name"] for p in forecast_partitions if p["fold_id"] == fold_picker.value}
    )
    # Auto-pick when the fold has a single experiment; the dropdown is only *shown* (in the
    # controls cell below) when there is a genuine choice to make.
    experiment_picker = mo.ui.dropdown(
        options=experiment_names,
        value=experiment_names[-1],
        label="Experiment",
    )
    return experiment_names, experiment_picker


@app.cell
def _(experiment_picker, fold_picker, settings):
    available_init_times = (
        pl.scan_delta(
            settings.power_forecasts_data_path,
            storage_options=typeddict_to_dict(settings.storage_options),
        )
        .filter(
            pl.col("experiment_name") == experiment_picker.value,
            pl.col("fold_id") == fold_picker.value,
        )
        .select("power_fcst_init_time")
        .unique()
        .collect()
        .get_column("power_fcst_init_time")
        .sort()
        .to_list()
    )
    return (available_init_times,)


@app.cell
def _(available_init_times):
    available_dates = sorted({t.date() for t in available_init_times})
    date_picker = mo.ui.date(
        start=available_dates[0],
        stop=available_dates[-1],
        value=available_dates[-1],
        label="Forecast date (UTC)",
    )
    return available_dates, date_picker


@app.cell
def _(available_dates, available_init_times, date_picker):
    runs_on_date = [t for t in available_init_times if t.date() == date_picker.value]
    if runs_on_date:
        run_picker = mo.ui.radio(
            options={t.strftime("%H:%M UTC"): t for t in runs_on_date},
            value=runs_on_date[-1].strftime("%H:%M UTC"),
            label="Forecast run",
            inline=True,
        )
        no_runs_message = None
    else:
        _nearest = min(available_dates, key=lambda d: abs(d - date_picker.value))
        run_picker = None
        no_runs_message = mo.callout(
            mo.md(
                f"No forecast runs on {date_picker.value} for this experiment and fold. "
                f"The nearest date with forecasts is **{_nearest}**."
            ),
            kind="warn",
        )
    return no_runs_message, run_picker


@app.cell
def _():
    weekend_shading = mo.ui.checkbox(value=True, label="Shade weekends")
    return (weekend_shading,)


@app.cell
def _(
    date_picker,
    experiment_names,
    experiment_picker,
    fold_picker,
    no_runs_message,
    run_picker,
    series_picker,
    weekend_shading,
):
    _pickers = [series_picker, fold_picker]
    if len(experiment_names) > 1:
        _pickers.append(experiment_picker)
    _pickers.append(date_picker)
    if run_picker is not None:
        _pickers.append(run_picker)
    _pickers.append(weekend_shading)
    _rows = [mo.hstack(_pickers, justify="start", gap=2, wrap=True)]
    if no_runs_message is not None:
        _rows.append(no_runs_message)
    mo.vstack(_rows)
    return


@app.cell
def _(experiment_picker, fold_picker, run_picker, series_picker, settings):
    mo.stop(run_picker is None or run_picker.value is None, mo.md(""))
    init_time = run_picker.value
    _storage = typeddict_to_dict(settings.storage_options)

    forecasts = (
        pl.scan_delta(settings.power_forecasts_data_path, storage_options=_storage)
        .filter(
            pl.col("experiment_name") == experiment_picker.value,
            pl.col("fold_id") == fold_picker.value,
            pl.col("time_series_id") == series_picker.value,
            pl.col("power_fcst_init_time") == init_time,
            pl.col("valid_time") <= init_time + PLOT_HORIZON,
        )
        .select("valid_time", "power_fcst", "ensemble_member")
        .collect()
    )
    actuals = (
        pl.scan_delta(settings.power_time_series_data_path, storage_options=_storage)
        .filter(
            pl.col("time_series_id") == series_picker.value,
            pl.col("time").is_between(init_time - PLOT_HISTORY, init_time + PLOT_HORIZON),
        )
        .select("time", "power")
        .collect()
    )
    mo.stop(
        forecasts.height == 0,
        mo.callout(
            mo.md("No forecast rows found for this selection — try another run or series."),
            kind="warn",
        ),
    )
    return actuals, forecasts, init_time


@app.cell
def _(
    actuals,
    experiment_picker,
    fold_picker,
    forecasts,
    init_time,
    metadata_df,
    series_picker,
    weekend_shading,
):
    _meta = metadata_df.filter(pl.col("time_series_id") == series_picker.value)
    chart = build_view_forecast_chart(
        forecasts.lazy(),
        actuals.lazy(),
        power_fcst_init_time=init_time,
        units=_meta["units"].item(),
        title=(
            f"{_meta['time_series_name'].item()} — {_meta['time_series_type'].item()}"
            f" — id {series_picker.value}"
        ),
        subtitle=(
            f"Forecast init {init_time:%a %d %b %Y %H:%M} UTC"
            f" · experiment {experiment_picker.value} · fold {fold_picker.value}"
        ),
        shade_weekends=weekend_shading.value,
    )
    # mo.ui.altair_chart serves the ~34k data rows as a virtual file instead of inlining them in
    # the cell output, which would blow marimo's max-output-size guard. Selections are disabled —
    # the chart's own scale-bound zoom/pan is the intended interaction.
    mo.ui.altair_chart(chart, chart_selection=False, legend_selection=False)
    return


if __name__ == "__main__":
    app.run()
