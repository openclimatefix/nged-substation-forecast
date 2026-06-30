"""Power forecast chart builder."""

import warnings
from typing import Final

import altair as alt
import polars as pl
from plotting import ocf_theme

_PANEL_WIDTH: Final[int] = 800
"""Pixel width of every panel; a fixed, shared width keeps the x-axes aligned across panels."""

_PANEL_HEIGHT: Final[int] = 250
"""Pixel height of each per-time-series panel."""


def build_forecast_chart(
    forecasts: pl.DataFrame,
    ground_truth: pl.DataFrame,
    metadata: pl.DataFrame,
    time_series_ids: list[int],
) -> alt.VConcatChart:
    """Build a stacked, interactive forecast chart — one panel per ``time_series_id``.

    Each panel layers all 51 ensemble members as thin lines (``power_fcst`` vs ``valid_time``)
    and, where ground truth is available for that series, the observed power as a thick line.
    Panels are stacked vertically with independent y-scales but a shared, zoomable x-scale so
    panning/zooming one panel moves them all and the time axes stay aligned.

    Args:
        forecasts: ``PowerForecast`` rows for the chosen init time, already filtered to
            ``time_series_ids``.
        ground_truth: ``PowerTimeSeries`` observations over the plotted window (may be empty for
            some or all series).
        metadata: ``TimeSeriesMetadata`` for the plotted series, used for panel titles and units.
        time_series_ids: The series to plot, in the order their panels should appear.

    Returns:
        A vertically concatenated Altair chart ready to ``.save(...)`` as interactive HTML.
    """
    # The full ensemble across up to 4 panels far exceeds Altair's 5000-row default guard.
    alt.data_transformers.disable_max_rows()

    # One shared, explicitly-named interval selection bound to the x-scale; adding the same param to
    # every panel makes Vega-Lite hoist it to a single top-level param, so zoom/pan is synchronised
    # across panels (and the time axes stay aligned).
    x_zoom = alt.selection_interval(bind="scales", encodings=["x"], name="shared_x_zoom")

    panels: list[alt.LayerChart | alt.FacetChart] = []
    for time_series_id in time_series_ids:
        forecast_panel = forecasts.filter(pl.col("time_series_id") == time_series_id)
        truth_panel = ground_truth.filter(pl.col("time_series_id") == time_series_id)
        meta_row = metadata.filter(pl.col("time_series_id") == time_series_id)

        units = meta_row["units"].item() if meta_row.height else "MW/MVA"
        name = meta_row["time_series_name"].item() if meta_row.height else str(time_series_id)
        series_type = meta_row["time_series_type"].item() if meta_row.height else "unknown"

        ensemble_layer = (
            alt.Chart(forecast_panel)
            .mark_line(strokeWidth=1, opacity=0.3, color=ocf_theme.ENSEMBLE_LINE)
            .encode(
                x=alt.X("valid_time:T", title="Valid time"),
                y=alt.Y("power_fcst:Q", title=f"Power ({units})"),
                detail="ensemble_member:N",
                tooltip=["ensemble_member", "valid_time", "power_fcst"],
            )
        )

        layers: list[alt.Chart] = [ensemble_layer]
        if truth_panel.height:
            truth_layer = (
                alt.Chart(truth_panel)
                .mark_line(strokeWidth=2.5, color=ocf_theme.BLUE)
                .encode(
                    x=alt.X("time:T"),
                    y=alt.Y("power:Q"),
                    tooltip=["time", "power"],
                )
            )
            layers.append(truth_layer)

        panel = (
            alt.layer(*layers)
            .properties(
                title=f"{name} (id={time_series_id}) — {series_type}",
                width=_PANEL_WIDTH,
                height=_PANEL_HEIGHT,
            )
            .add_params(x_zoom)
        )
        panels.append(panel)

    # Altair warns that it deduplicated the identical x_zoom param across panels — that dedup is
    # exactly how the shared zoom is achieved here, so the warning is expected, not a problem.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Automatically deduplicated selection parameter")
        return alt.vconcat(*panels).resolve_scale(y="independent")
