"""Manually launched visualisation jobs for inspecting forecasts.

``plot_power_forecast_job`` renders an interactive HTML plot of forecasts that already exist in the
``power_forecasts`` Delta table, so a human can eyeball one forecast. It is a job (not an asset)
because the plot is a throwaway artifact for human eyes — keyed by ``(init time, time_series_ids)``,
with no lineage and no durable catalog identity — rather than a tracked data object. Its op reads
the Delta/parquet sources directly and delegates chart construction to the pure, unit-testable
:func:`plotting.forecast_chart.build_forecast_chart`.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import polars as pl
from contracts.settings import Settings
from dagster import Config, MetadataValue, OpExecutionContext, job, op
from plotting.forecast_chart import build_forecast_chart
from pydantic import Field

FORECAST_HORIZON: Final[timedelta] = timedelta(days=14)
"""Length of the forecast horizon plotted from ``power_fcst_init_time``."""


class PlotPowerForecastConfig(Config):
    """Run config for ``plot_power_forecast_job``.

    The defaults point at the smoke-test forecasts currently materialised in the
    ``power_forecasts`` Delta table, so the job is runnable with no config edits.
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


@op
def plot_power_forecast(context: OpExecutionContext, config: PlotPowerForecastConfig) -> None:
    """Render an interactive HTML plot of one forecast for 1–4 ``time_series_id``s.

    Reads the ``power_forecasts`` Delta table for ``(experiment_name, fold_id)`` at the chosen
    ``power_fcst_init_time``, the observed power over the 14-day horizon, and the series metadata,
    then writes an interactive Altair HTML file to ``settings.plots_data_path``. Reads Delta
    directly, so it plots whatever forecasts already exist on disk.
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
        pl.scan_delta(settings.power_time_series_data_path)
        .filter(
            pl.col("time_series_id").is_in(config.time_series_ids),
            pl.col("time") >= init_time,
            pl.col("time") <= window_end,
        )
        .collect()
    )
    metadata = pl.read_parquet(settings.metadata_path).filter(
        pl.col("time_series_id").is_in(config.time_series_ids)
    )

    chart = build_forecast_chart(forecasts, ground_truth, metadata, config.time_series_ids)

    plots_data_path = Path(settings.plots_data_path)
    plots_data_path.mkdir(parents=True, exist_ok=True)
    ids_label = "-".join(str(i) for i in config.time_series_ids)
    filename = (
        f"{config.experiment_name}__{config.fold_id}__{init_time:%Y%m%dT%H%MZ}__ts-{ids_label}.html"
    )
    out_path: Path = plots_data_path / filename
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


@job
def plot_power_forecast_job() -> None:
    """Render an interactive HTML plot of one forecast for 1–4 ``time_series_id``s."""
    plot_power_forecast()
