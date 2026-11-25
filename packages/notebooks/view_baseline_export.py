"""Inspect the parquet file ``scripts/forecasting/export_baseline_forecasts.py`` produces.

For a chosen time series the notebook draws two charts. The first overlays observed power against
the forecast median (p50) and the p10-p90 band, the range the forecast expects eight outcomes in
ten to fall inside. The second plots the observed-minus-median residual, the raw material for
switching-event detection: the residual shows a sudden step in the gap between measured and
forecast power — the signature of a feeder being switched onto or off the substation.

Both charts read the exported ``*_quantiles.parquet`` file. Every chart is pan/zoom interactive.
Titles carry the series id, type, and name. The y-axis is labelled in the series' own unit (MW or
MVA).

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
    from plotting import ocf_theme

    DEFAULT_EXPORT_DIR = PROJECT_ROOT / "data" / "exports"

    # A single series spans ~17k half-hourly rows, because the leaderboard fold validates on 12
    # months. mo.ui.altair_chart serves those rows as a virtual file rather than inlining the
    # rows. Lift altair's default 5,000-row guard the way the dashboard does.
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

        A stem is present when its ``*_quantiles.parquet`` file exists.

        Args:
            export_dir: Directory holding the exported parquet files.

        Returns:
            Sorted list of stems, empty if the directory has no exports.
        """
        if not export_dir.is_dir():
            return []
        return sorted(
            p.name.removesuffix("_quantiles.parquet")
            for p in export_dir.glob("*_quantiles.parquet")
        )

    def load_frame(export_dir: Path, stem: str) -> pl.DataFrame:
        """Read one exported ``*_quantiles.parquet`` file.

        Args:
            export_dir: Directory holding the exported parquet files.
            stem: The ``{experiment}__{fold}`` stem naming which export to read.

        Returns:
            The whole parquet, eagerly, for every time series in the export.
        """
        return pl.read_parquet(export_dir / f"{stem}_quantiles.parquet")

    def series_descr(series_id: int) -> tuple[str, str]:
        """Return ``(title_prefix, unit)`` for a series from its name, type, and unit.

        Args:
            series_id: The ``time_series_id`` to describe.

        Returns:
            A ``(title_prefix, unit)`` tuple — e.g. ``("id 7 · PV · Foo Solar Farm", "MW")``. The
            middle field of ``title_prefix`` is a ``LIST_OF_TIME_SERIES_TYPES`` value, and so
            carries the contract's own capitalisation. Falls back to ``("id {series_id}",
            "MW/MVA")`` when the series is absent from ``SERIES_META``.
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

    Weather/calendar-only XGBoost baseline (no power lags) — issue #179. A baseline is the
    deliberately simple model that better models have to beat; this one is built from weather and
    calendar inputs alone, and ignores recently measured power. Pick an export and a time series
    to compare the forecast against observed power and inspect the residual. Drag to pan and
    scroll to zoom on any chart.
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
        mo.md(f"No `*_quantiles.parquet` files found in `{export_dir}`."),
    )
    stem_ui
    return export_dir, stem_ui


@app.cell
def _(export_dir, stem_ui):
    quantiles_df = load_frame(export_dir, stem_ui.value)

    series_ids = quantiles_df["time_series_id"].unique().sort().to_list()
    series_ui = mo.ui.dropdown(
        options=[str(s) for s in series_ids],
        value=str(series_ids[0]),
        label="Time series id",
    )
    series_ui
    return quantiles_df, series_ui


@app.cell
def _(quantiles_df, series_ui):
    series_id = int(series_ui.value)
    title_prefix, unit = series_descr(series_id)
    y_title = f"power ({unit})"
    quant_one = quantiles_df.filter(pl.col("time_series_id") == series_id).sort("valid_time")

    band = (
        alt.Chart(quant_one)
        .mark_area(opacity=0.25, color=ocf_theme.DATA_BLUE)
        .encode(  # ty: ignore[missing-attribute]  # astral-sh/ty#2520
            x=date_x(),
            y=alt.Y("power_fcst_p10:Q", title=y_title),
            y2="power_fcst_p90:Q",
        )
    )
    p50_line = (
        alt.Chart(quant_one)
        .mark_line(strokeWidth=1, color=ocf_theme.DATA_BLUE)
        .encode(x=date_x(), y="power_fcst_p50:Q")  # ty: ignore[missing-attribute]
    )
    observed_line = (
        alt.Chart(quant_one)
        .mark_line(strokeWidth=1.2, color=ocf_theme.BLACK_1)
        .encode(  # ty: ignore[missing-attribute]  # astral-sh/ty#2520
            x=date_x(),
            y=alt.Y("observed_power:Q", title=y_title),
            tooltip=[
                alt.Tooltip("valid_time:T", format="%Y-%m-%d %H:%M", title="valid time"),
                alt.Tooltip("observed_power:Q", format=".2f", title="observed"),
                alt.Tooltip("power_fcst_p50:Q", format=".2f", title="median"),
            ],
        )
    )
    forecast_chart = mo.ui.altair_chart(
        (band + p50_line + observed_line)
        .properties(
            height=320,
            width="container",
            title=f"{title_prefix} — observed (black) vs forecast p10–p90 band and p50 (blue)",
        )
        .interactive(),
        chart_selection=False,
        legend_selection=False,
    )
    forecast_chart
    return quant_one, title_prefix, unit


@app.cell
def _(quant_one, title_prefix, unit):
    residual_df = quant_one.with_columns(
        residual=pl.col("observed_power") - pl.col("power_fcst_p50")
    )
    residual_chart = mo.ui.altair_chart(
        alt.Chart(residual_df)
        .mark_line(strokeWidth=1, color=ocf_theme.DATA_PURPLE)
        .encode(  # ty: ignore[missing-attribute]  # astral-sh/ty#2520
            x=date_x(),
            y=alt.Y("residual:Q", title=f"observed − forecast median ({unit})"),
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


if __name__ == "__main__":
    app.run()
