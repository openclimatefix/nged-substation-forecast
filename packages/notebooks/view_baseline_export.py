"""Inspect the three baseline-export parquets produced by ``scripts/export_forecasts_for_alex.py``.

For a chosen time series it overlays observed power against the forecast (ensemble mean, p50, and
the p10-p90 band), plots the observed-minus-expected residual (the raw material for switching-event
detection), and draws the raw ensemble members over a selected window. All three exported parquets
(``*_full_ensemble``, ``*_ensemble_mean``, ``*_quantiles``) are used.

    uv run marimo edit packages/notebooks/view_baseline_export.py
"""

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl
    from contracts.settings import PROJECT_ROOT

    DEFAULT_EXPORT_DIR = PROJECT_ROOT / "data" / "exports"

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


@app.cell
def _():
    mo.md(
        """
        # Baseline export viewer

        Weather/calendar-only XGBoost baseline (no power lags) — issue #179. Pick an export and a
        time series to compare the forecast against observed power and inspect the residual.
        """
    )
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
    mean_one = mean_df.filter(pl.col("time_series_id") == series_id).sort("valid_time")
    quant_one = quantiles_df.filter(pl.col("time_series_id") == series_id).sort("valid_time")

    band = (
        alt.Chart(quant_one)
        .mark_area(opacity=0.25, color="#4c78a8")
        .encode(
            x=alt.X("valid_time:T", title="valid time"),
            y=alt.Y("power_fcst_p10:Q", title="power (MW / MVA)"),
            y2="power_fcst_p90:Q",
        )
    )
    p50_line = (
        alt.Chart(quant_one)
        .mark_line(strokeWidth=1, color="#4c78a8")
        .encode(x="valid_time:T", y="power_fcst_p50:Q")
    )
    mean_line = (
        alt.Chart(mean_one)
        .mark_line(strokeWidth=1, strokeDash=[4, 2], color="#f58518")
        .encode(x="valid_time:T", y="power_fcst_mean:Q")
    )
    observed_line = (
        alt.Chart(mean_one)
        .mark_line(strokeWidth=1.2, color="#333333")
        .encode(x="valid_time:T", y=alt.Y("observed_power:Q"))
    )
    forecast_chart = mo.ui.altair_chart(
        (band + p50_line + mean_line + observed_line).properties(
            height=320,
            width="container",
            title="Observed (black) vs p10–p90 band, p50, ensemble mean (dashed)",
        )
    )
    forecast_chart
    return mean_one, series_id


@app.cell
def _(mean_one):
    residual_df = mean_one.with_columns(
        residual=pl.col("observed_power") - pl.col("power_fcst_mean")
    )
    residual_chart = mo.ui.altair_chart(
        alt.Chart(residual_df)
        .mark_line(strokeWidth=1, color="#54a24b")
        .encode(
            x=alt.X("valid_time:T", title="valid time"),
            y=alt.Y("residual:Q", title="observed − ensemble mean (MW / MVA)"),
        )
        .properties(
            height=200,
            width="container",
            title="Residual — a sustained level shift here is the switching-event signature",
        )
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
def _(full_lf, series_id, window_ui):
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
            x=alt.X("valid_time:T", title="valid time"),
            y=alt.Y("power_fcst:Q", title="power (MW / MVA)"),
            detail="ensemble_member:N",
        )
    )
    observed = (
        alt.Chart(ens_one.unique("valid_time"))
        .mark_line(strokeWidth=1.2, color="#333333")
        .encode(x="valid_time:T", y="observed_power:Q")
    )
    mo.ui.altair_chart(
        (members + observed).properties(
            height=300, width="container", title="All ensemble members (blue) vs observed (black)"
        )
    )
    return


if __name__ == "__main__":
    app.run()
