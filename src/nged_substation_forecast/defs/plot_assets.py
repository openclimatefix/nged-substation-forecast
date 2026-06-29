"""Ad-hoc visualisation assets for inspecting forecasts.

``plot_power_forecast`` is an unpartitioned, config-driven asset: rather than producing a data
artifact for the pipeline it renders an interactive HTML plot of forecasts that already exist in
the ``power_forecasts`` Delta table, so a human can eyeball one forecast. It reads the Delta/parquet
sources directly (no ``deps`` into the partitioned CV graph) and delegates chart construction to the
pure, unit-testable :func:`build_forecast_chart`.
"""

import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from contracts.settings import Settings
from dagster import AssetExecutionContext, Config, MetadataValue, asset
from pydantic import Field

FORECAST_HORIZON: Final[timedelta] = timedelta(days=14)
"""Length of the forecast horizon plotted from ``power_fcst_init_time``."""

_PANEL_WIDTH: Final[int] = 800
"""Pixel width of every panel; a fixed, shared width keeps the x-axes aligned across panels."""

_PANEL_HEIGHT: Final[int] = 250
"""Pixel height of each per-time-series panel."""


class PlotPowerForecastConfig(Config):
    """Run config for ``plot_power_forecast``.

    The defaults point at the smoke-test forecasts currently materialised in the
    ``power_forecasts`` Delta table, so the asset is runnable with no config edits.
    """

    experiment_name: str = Field(
        default="xgboost_smoke_test_3",
        description="Experiment whose forecasts to plot; selects the Delta partition to read.",
    )
    fold_id: str = Field(
        default="smoke_test",
        description="CV fold (or 'live') whose forecasts to plot; selects the Delta partition.",
    )
    power_fcst_init_time: str = Field(
        default="2025-01-29T06:00:00+00:00",
        description=(
            "ISO-8601 forecast init time to plot (Dagster config has no native datetime type)."
            " The plot spans this time to this time plus the 14-day forecast horizon. A naive"
            " value is interpreted as UTC."
        ),
    )
    time_series_ids: list[int] = Field(
        default=[1, 2, 3, 4],
        min_length=1,
        max_length=4,
        description="Between 1 and 4 time_series_ids; each is drawn on its own panel.",
    )


def build_forecast_chart(
    forecasts: pl.DataFrame,
    ground_truth: pl.DataFrame,
    metadata: pl.DataFrame,
    time_series_ids: list[int],
) -> alt.VConcatChart:
    """Build a stacked, interactive forecast chart — one panel per ``time_series_id``.

    Each panel layers all 51 ensemble members as thin grey lines (``power_fcst`` vs ``valid_time``)
    and, where ground truth is available for that series, the observed power as a thick blue line.
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
            .mark_line(strokeWidth=1, opacity=0.3, color="lightgray")
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
                .mark_line(strokeWidth=2.5, color="steelblue")
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


@asset
def plot_power_forecast(context: AssetExecutionContext, config: PlotPowerForecastConfig) -> None:
    """Render an interactive HTML plot of one forecast for 1–4 ``time_series_id``s.

    Reads the ``power_forecasts`` Delta table for ``(experiment_name, fold_id)`` at the chosen
    ``power_fcst_init_time``, the observed power over the 14-day horizon, and the series metadata,
    then writes an interactive Altair HTML file to ``settings.plots_data_path``. The asset is
    unpartitioned and reads Delta directly (no ``deps``), so it plots whatever forecasts already
    exist on disk.
    """
    settings = Settings()
    init_time = datetime.fromisoformat(config.power_fcst_init_time)
    if init_time.tzinfo is None:
        init_time = init_time.replace(tzinfo=UTC)
    window_end = init_time + FORECAST_HORIZON

    forecasts = (
        pl.scan_delta(str(settings.power_forecasts_data_path))
        .filter(
            pl.col("experiment_name") == config.experiment_name,
            pl.col("fold_id") == config.fold_id,
            pl.col("time_series_id").is_in(config.time_series_ids),
            pl.col("power_fcst_init_time") == init_time,
        )
        .collect()
    )
    if forecasts.height == 0:
        raise ValueError(
            "No forecasts found for "
            f"experiment_name={config.experiment_name!r}, fold_id={config.fold_id!r}, "
            f"power_fcst_init_time={init_time.isoformat()}, "
            f"time_series_ids={config.time_series_ids}. "
            "Check that the matching partition has been materialised."
        )

    ground_truth = (
        pl.scan_delta(str(settings.nged_data_path / "power_time_series.delta"))
        .filter(
            pl.col("time_series_id").is_in(config.time_series_ids),
            pl.col("time") >= init_time,
            pl.col("time") <= window_end,
        )
        .collect()
    )
    metadata = pl.read_parquet(settings.nged_data_path / "metadata.parquet").filter(
        pl.col("time_series_id").is_in(config.time_series_ids)
    )

    chart = build_forecast_chart(forecasts, ground_truth, metadata, config.time_series_ids)

    settings.plots_data_path.mkdir(parents=True, exist_ok=True)
    ids_label = "-".join(str(i) for i in config.time_series_ids)
    filename = (
        f"{config.experiment_name}__{config.fold_id}__{init_time:%Y%m%dT%H%MZ}__ts-{ids_label}.html"
    )
    out_path: Path = settings.plots_data_path / filename
    chart.save(str(out_path))

    context.add_output_metadata(
        {
            "plot_path": MetadataValue.path(str(out_path)),
            "experiment_name": config.experiment_name,
            "fold_id": config.fold_id,
            "power_fcst_init_time": init_time.isoformat(),
            "time_series_ids": str(config.time_series_ids),
            "n_forecast_rows": forecasts.height,
            "n_ensemble_members": forecasts["ensemble_member"].n_unique(),
            "n_ground_truth_rows": ground_truth.height,
        }
    )
