"""Inspect the three baseline-export parquets produced by ``scripts/export_forecasts_for_alex.py``.

For a chosen time series it overlays observed power against the forecast (ensemble mean, p50, and
the p10-p90 band), plots the observed-minus-expected residual (the raw material for switching-event
detection), and draws the raw ensemble members over a selected window. All three exported parquets
(``*_full_ensemble``, ``*_ensemble_mean``, ``*_quantiles``) are used. Every chart is pan/zoom
interactive; titles carry the series id, type, and name; the y-axis is labelled in the series' own
unit (MW or MVA).

    uv run marimo edit packages/notebooks/view_baseline_export.py
"""

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")

with app.setup:
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl
    from contracts.settings import PROJECT_ROOT, Settings
    from contracts.typing_utils import typeddict_to_dict

    DEFAULT_EXPORT_DIR = PROJECT_ROOT / "data" / "exports"

    # A single series spans ~17k half-hourly rows; mo.ui.altair_chart serves them as a virtual file
    # rather than inlining, so lift altair's default 5000-row guard (mirrors the dashboard).
    alt.data_transformers.disable_max_rows()

    SETTINGS = Settings()
    # Per-series descriptive metadata (name, type, unit) keyed by time_series_id, loaded once.
    SERIES_META: dict[int, dict[str, str]] = {
        row["time_series_id"]: {
            "time_series_name": row["time_series_name"],
            "time_series_type": row["time_series_type"],
            "units": row["units"],
        }
        for row in pl.read_parquet(
            SETTINGS.metadata_path,
            storage_options=typeddict_to_dict(SETTINGS.storage_options),
        )
        .select("time_series_id", "time_series_name", "time_series_type", "units")
        .iter_rows(named=True)
    }

    def available_stems(export_dir: Path) -> list[str]:
        """Return the export stems (``{experiment}__{fold}``) found in ``export_dir``.

        A stem is present when its ``*_ensemble_mean.parquet`` file exists.

        Args:
            export_dir: Directory holding the exported parquet files.

        Returns:
            Sorted list of stems, empty if the directory has no exports.
        """
        if not export_dir.is_dir():
            return []
        return sorted(
            p.name.removesuffix("_ensemble_mean.parquet")
            for p in export_dir.glob("*_ensemble_mean.parquet")
        )

    def load_frame(export_dir: Path, stem: str, kind: str) -> pl.DataFrame:
        """Read one exported parquet (``kind`` in {full_ensemble, ensemble_mean, quantiles})."""
        return pl.read_parquet(export_dir / f"{stem}_{kind}.parquet")

    def series_descr(series_id: int) -> tuple[str, str]:
        """Return ``(title_prefix, unit)`` for a series from its metadata.

        Args:
            series_id: The ``time_series_id`` to describe.

        Returns:
            A ``(title_prefix, unit)`` tuple — e.g. ``("id 7 · pv · Foo Primary", "MW")``.
            Falls back to ``("id {series_id}", "MW/MVA")`` when the series is absent from metadata.
        """
        meta = SERIES_META.get(series_id)
        if meta is None:
            return f"id {series_id}", "MW/MVA"
        return (
            f"id {series_id} · {meta['time_series_type']} · {meta['time_series_name']}",
            meta["units"],
        )

    def date_x() -> alt.X:
        """An x-axis channel over ``valid_time`` whose tick labels always show the date."""
        return alt.X("valid_time:T", axis=alt.Axis(title="date", format="%d %b %Y"))


@app.cell
def _():
    mo.md("""
    # Baseline export viewer

    Weather/calendar-only XGBoost baseline (no power lags) — issue #179. Pick an export and a
    time series to compare the forecast against observed power and inspect the residual. Drag to
    pan and scroll to zoom on any chart.
    """)
    return


@app.cell
def _():
    export_dir_ui = mo.ui.text(
        value=str(DEFAULT_EXPORT_DIR),
        label="Export directory",
        full_width=True,
    )
    export_dir_ui
    return (export_dir_ui,)


@app.cell
def _(export_dir_ui):
    export_dir = Path(export_dir_ui.value).expanduser()
    stems = available_stems(export_dir)
    stem_ui = mo.ui.dropdown(
        options=stems,
        value=stems[0] if stems else None,
        label="Export (experiment__fold)",
    )
    mo.stop(
        not stems,
        mo.md(f"No `*_ensemble_mean.parquet` files found in `{export_dir}`."),
    )
    stem_ui
    return export_dir, stem_ui


@app.cell
def _(export_dir, stem_ui):
    mean_df = load_frame(export_dir, stem_ui.value, "ensemble_mean")
    quantiles_df = load_frame(export_dir, stem_ui.value, "quantiles")

    series_ids = mean_df["time_series_id"].unique().sort().to_list()
    series_ui = mo.ui.dropdown(
        options=[str(s) for s in series_ids],
        value=str(series_ids[0]),
        label="Time series id",
    )
    series_ui
    return mean_df, quantiles_df, series_ui


@app.cell
def _(mean_df, quantiles_df, series_ui):
    series_id = int(series_ui.value)
    title_prefix, unit = series_descr(series_id)
    y_title = f"power ({unit})"
    mean_one = mean_df.filter(pl.col("time_series_id") == series_id).sort("valid_time")
    quant_one = quantiles_df.filter(pl.col("time_series_id") == series_id).sort("valid_time")

    band = (
        alt.Chart(quant_one)
        .mark_area(opacity=0.25, color="#4c78a8")
        .encode(x=date_x(), y=alt.Y("power_fcst_p10:Q", title=y_title), y2="power_fcst_p90:Q")
    )
    p50_line = (
        alt.Chart(quant_one)
        .mark_line(strokeWidth=1, color="#4c78a8")
        .encode(x=date_x(), y="power_fcst_p50:Q")
    )
    mean_line = (
        alt.Chart(mean_one)
        .mark_line(strokeWidth=1, strokeDash=[4, 2], color="#f58518")
        .encode(x=date_x(), y="power_fcst_mean:Q")
    )
    observed_line = (
        alt.Chart(mean_one)
        .mark_line(strokeWidth=1.2, color="#333333")
        .encode(
            x=date_x(),
            y=alt.Y("observed_power:Q", title=y_title),
            tooltip=[
                alt.Tooltip("valid_time:T", format="%Y-%m-%d %H:%M", title="valid time"),
                alt.Tooltip("observed_power:Q", format=".2f", title="observed"),
                alt.Tooltip("power_fcst_mean:Q", format=".2f", title="ens mean"),
            ],
        )
    )
    forecast_chart = mo.ui.altair_chart(
        (band + p50_line + mean_line + observed_line)
        .properties(
            height=320,
            width="container",
            title=f"{title_prefix} — observed (black) vs p10–p90 band, p50, ensemble mean (dashed)",
        )
        .interactive(),
        chart_selection=False,
        legend_selection=False,
    )
    forecast_chart
    return mean_one, series_id, title_prefix, unit


@app.cell
def _(mean_one, title_prefix, unit):
    residual_df = mean_one.with_columns(
        residual=pl.col("observed_power") - pl.col("power_fcst_mean")
    )
    residual_chart = mo.ui.altair_chart(
        alt.Chart(residual_df)
        .mark_line(strokeWidth=1, color="#54a24b")
        .encode(
            x=date_x(),
            y=alt.Y("residual:Q", title=f"observed − ensemble mean ({unit})"),
            tooltip=[
                alt.Tooltip("valid_time:T", format="%Y-%m-%d %H:%M", title="valid time"),
                alt.Tooltip("residual:Q", format=".2f", title="residual"),
            ],
        )
        .properties(
            height=200,
            width="container",
            title=f"{title_prefix} — residual (a sustained level shift is the switching signature)",
        )
        .interactive(),
        chart_selection=False,
        legend_selection=False,
    )
    residual_chart
    return


@app.cell
def _(export_dir, stem_ui):
    window_ui = mo.ui.range_slider(
        start=1,
        stop=28,
        value=[1, 14],
        step=1,
        label="Ensemble spaghetti: window (days from series start)",
    )
    full_lf = pl.scan_parquet(export_dir / f"{stem_ui.value}_full_ensemble.parquet")
    window_ui
    return full_lf, window_ui


@app.cell
def _(full_lf, series_id, title_prefix, unit, window_ui):
    one = full_lf.filter(pl.col("time_series_id") == series_id)
    start = one.select(pl.col("valid_time").min()).collect().item()
    lo = start + pl.duration(days=window_ui.value[0] - 1)
    hi = start + pl.duration(days=window_ui.value[1])
    ens_one = (
        one.filter(pl.col("valid_time") >= lo, pl.col("valid_time") < hi)
        .select("valid_time", "ensemble_member", "power_fcst", "observed_power")
        .collect()
    )
    members = (
        alt.Chart(ens_one)
        .mark_line(strokeWidth=0.5, opacity=0.3, color="#4c78a8")
        .encode(
            x=date_x(),
            y=alt.Y("power_fcst:Q", title=f"power ({unit})"),
            detail="ensemble_member:N",
        )
    )
    observed = (
        alt.Chart(ens_one.unique("valid_time"))
        .mark_line(strokeWidth=1.2, color="#333333")
        .encode(x=date_x(), y="observed_power:Q")
    )
    mo.ui.altair_chart(
        (members + observed)
        .properties(
            height=300,
            width="container",
            title=f"{title_prefix} — all ensemble members (blue) vs observed (black)",
        )
        .interactive(),
        chart_selection=False,
        legend_selection=False,
    )
    return


if __name__ == "__main__":
    app.run()
