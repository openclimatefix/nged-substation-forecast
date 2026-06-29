"""Tests for the ``plot_power_forecast`` asset and its pure chart builder.

The chart builder and config bounds are exercised as fast unit tests; the asset itself is run
end-to-end against temp Delta/parquet sources via ``materialize`` to confirm an HTML file lands on
disk.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from altair import VConcatChart
from dagster import DagsterInstance, RunConfig, materialize
from pydantic import ValidationError

from nged_substation_forecast.defs.plot_assets import (
    PlotPowerForecastConfig,
    build_forecast_chart,
    plot_power_forecast,
)

EXPERIMENT_NAME = "exp_plot"
FOLD_ID = "smoke_test"
INIT_TIME = datetime(2025, 1, 29, 6, 0, tzinfo=UTC)


def _valid_times() -> pl.Series:
    return pl.datetime_range(
        INIT_TIME, INIT_TIME + timedelta(hours=4), interval="30m", time_zone="UTC", eager=True
    )


def _forecasts(time_series_ids: tuple[int, ...], members: tuple[int, ...]) -> pl.DataFrame:
    rows = [
        {
            "experiment_name": EXPERIMENT_NAME,
            "fold_id": FOLD_ID,
            "time_series_id": ts,
            "ensemble_member": m,
            "power_fcst_init_time": INIT_TIME,
            "valid_time": vt,
            "power_fcst": 10.0 + m + i,
        }
        for ts in time_series_ids
        for m in members
        for i, vt in enumerate(_valid_times())
    ]
    return pl.DataFrame(rows).cast(
        {
            "time_series_id": pl.Int32,
            "ensemble_member": pl.Int8,
            "power_fcst_init_time": pl.Datetime("us", "UTC"),
            "valid_time": pl.Datetime("us", "UTC"),
            "power_fcst": pl.Float32,
        }
    )


def _ground_truth(time_series_ids: tuple[int, ...]) -> pl.DataFrame:
    rows = [
        {"time_series_id": ts, "time": t, "power": 12.0 + i}
        for ts in time_series_ids
        for i, t in enumerate(_valid_times())
    ]
    return pl.DataFrame(rows).cast(
        {"time_series_id": pl.Int32, "time": pl.Datetime("us", "UTC"), "power": pl.Float32}
    )


def _metadata(time_series_ids: tuple[int, ...]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time_series_id": pl.Series(list(time_series_ids), dtype=pl.Int32),
            "units": [("MW" if ts % 2 else "MVA") for ts in time_series_ids],
            "time_series_name": [f"Substation {ts}" for ts in time_series_ids],
            "time_series_type": ["PV" for _ in time_series_ids],
        }
    )


def test_build_forecast_chart_one_panel_per_series_with_optional_truth() -> None:
    # ts1 has ground truth (ensemble + truth layers); ts2 has none (ensemble layer only).
    forecasts = _forecasts((1, 2), (0, 1, 2))
    ground_truth = _ground_truth((1,))
    metadata = _metadata((1, 2))

    chart = build_forecast_chart(forecasts, ground_truth, metadata, [1, 2])

    assert isinstance(chart, VConcatChart)
    assert len(chart.vconcat) == 2
    assert len(chart.vconcat[0].layer) == 2  # ts1: ensemble + ground truth
    assert len(chart.vconcat[1].layer) == 1  # ts2: ensemble only
    assert "ensemble_member" in chart.to_html()


def test_config_rejects_out_of_range_time_series_id_counts() -> None:
    with pytest.raises(ValidationError):
        PlotPowerForecastConfig(time_series_ids=[])
    with pytest.raises(ValidationError):
        PlotPowerForecastConfig(time_series_ids=[1, 2, 3, 4, 5])


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    nged_path = tmp_path / "NGED"
    nged_path.mkdir()
    plots_path = tmp_path / "plots"
    monkeypatch.setenv("NGED_S3_BUCKET_URL", "https://example.com")
    monkeypatch.setenv("NGED_S3_BUCKET_ACCESS_KEY", "dummy")
    monkeypatch.setenv("NGED_S3_BUCKET_SECRET", "dummy")
    monkeypatch.setenv("NGED_DATA_PATH", str(nged_path))
    monkeypatch.setenv("POWER_FORECASTS_DATA_PATH", str(tmp_path / "power_forecasts"))
    monkeypatch.setenv("PLOTS_DATA_PATH", str(plots_path))

    _forecasts((1, 2), (0, 1, 2)).write_delta(
        str(tmp_path / "power_forecasts"),
        delta_write_options={"partition_by": ["experiment_name", "fold_id"]},
    )
    _ground_truth((1, 2)).write_delta(str(nged_path / "power_time_series.delta"))
    _metadata((1, 2)).write_parquet(nged_path / "metadata.parquet")
    return plots_path


@pytest.mark.integration
def test_plot_power_forecast_writes_html(env: Path) -> None:
    result = materialize(
        [plot_power_forecast],
        instance=DagsterInstance.ephemeral(),
        run_config=RunConfig(
            ops={
                "plot_power_forecast": PlotPowerForecastConfig(
                    experiment_name=EXPERIMENT_NAME,
                    fold_id=FOLD_ID,
                    power_fcst_init_time=INIT_TIME.isoformat(),
                    time_series_ids=[1, 2],
                )
            }
        ),
    )
    assert result.success

    html_files = list(env.glob("*.html"))
    assert len(html_files) == 1
    assert html_files[0].stat().st_size > 0
    assert "ensemble_member" in html_files[0].read_text()
