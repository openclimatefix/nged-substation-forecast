"""Tests for compute_metrics()."""

import math
from datetime import datetime, timezone

import patito as pt
import polars as pl
import pytest
from contracts.ml_schemas import Metrics
from contracts.power_schemas import PowerForecast, PowerTimeSeries
from ml_core.metrics import compute_metrics


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def _make_actuals(
    time_series_id: int,
    times: list[datetime],
    power: list[float],
) -> pt.LazyFrame[PowerTimeSeries]:
    df = pl.DataFrame(
        {
            "time_series_id": pl.Series([time_series_id] * len(times), dtype=pl.Int32),
            "time": pl.Series(times, dtype=pl.Datetime("us", "UTC")),
            "power": pl.Series(power, dtype=pl.Float32),
        }
    )
    return pt.LazyFrame.from_existing(df.lazy()).set_model(PowerTimeSeries)


def _make_cv_forecasts(
    time_series_id: int,
    times: list[datetime],
    power_fcst: list[float],
    fold_id: str = "2022",
) -> pt.DataFrame[PowerForecast]:
    n = len(times)
    init_time = _utc(2022, 1, 1)
    df = pl.DataFrame(
        {
            "time_series_id": pl.Series([time_series_id] * n, dtype=pl.Int32),
            "valid_time": pl.Series(times, dtype=pl.Datetime("us", "UTC")),
            "power_fcst_init_time": pl.Series([init_time] * n, dtype=pl.Datetime("us", "UTC")),
            "ensemble_member": pl.Series([0] * n, dtype=pl.Int8),
            "nwp_init_time": pl.Series([None] * n, dtype=pl.Datetime("us", "UTC")),
            "power_fcst": pl.Series(power_fcst, dtype=pl.Float32),
            "power_fcst_model_name": pl.Series(["stub"] * n).cast(pl.Categorical),
            "power_fcst_model_version": pl.Series([1] * n, dtype=pl.Int16),
            "ml_flow_experiment_id": pl.Series([None] * n, dtype=pl.Int32),
            "fold_id": pl.Series([fold_id] * n).cast(pl.Categorical),
        }
    )
    return PowerForecast.validate(df, allow_superfluous_columns=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_compute_metrics_returns_metrics_schema():
    times = [_utc(2022, 1, 1, 0, 0), _utc(2022, 1, 1, 0, 30)]
    actuals = _make_actuals(1, times, [10.0, 10.0])
    forecasts = _make_cv_forecasts(1, times, [12.0, 8.0])
    result = compute_metrics(forecasts, actuals)
    assert isinstance(result, pl.DataFrame)
    Metrics.validate(result, allow_superfluous_columns=True)


def test_compute_metrics_mae_correctness():
    """MAE should equal mean absolute error of (fcst - actual)."""
    times = [_utc(2022, 1, 1, 0, 0), _utc(2022, 1, 1, 0, 30)]
    actuals = _make_actuals(1, times, [10.0, 10.0])
    # errors: +2, -2 → MAE = 2.0
    forecasts = _make_cv_forecasts(1, times, [12.0, 8.0])
    result = compute_metrics(forecasts, actuals)
    mae_row = result.filter((pl.col("metric_name") == "mae") & (pl.col("horizon_slice") == "all"))
    assert math.isclose(mae_row["metric_value"][0], 2.0, rel_tol=1e-5)


def test_compute_metrics_mbe_sign():
    """MBE is positive when forecasts over-predict."""
    times = [_utc(2022, 1, 1, 0, 0), _utc(2022, 1, 1, 0, 30)]
    actuals = _make_actuals(1, times, [10.0, 10.0])
    forecasts = _make_cv_forecasts(1, times, [15.0, 15.0])  # over-predict by 5
    result = compute_metrics(forecasts, actuals)
    mbe_row = result.filter((pl.col("metric_name") == "mbe") & (pl.col("horizon_slice") == "all"))
    assert mbe_row["metric_value"][0] > 0


def test_compute_metrics_nmae_normalisation():
    """NMAE = MAE / mean(|actual|). When actuals are all 10 and MAE=2, NMAE=0.2."""
    times = [_utc(2022, 1, 1, 0, 0), _utc(2022, 1, 1, 0, 30)]
    actuals = _make_actuals(1, times, [10.0, 10.0])
    forecasts = _make_cv_forecasts(1, times, [12.0, 8.0])  # MAE=2, mean|actual|=10
    result = compute_metrics(forecasts, actuals)
    nmae_row = result.filter((pl.col("metric_name") == "nmae") & (pl.col("horizon_slice") == "all"))
    assert math.isclose(nmae_row["metric_value"][0], 0.2, rel_tol=1e-5)


def test_compute_metrics_ensemble_averaging():
    """Ensemble mean should be taken before computing metrics."""
    time = _utc(2022, 1, 1, 0, 0)
    init_time = _utc(2022, 1, 1)
    # Two ensemble members: forecasts 8 and 12 → ensemble mean = 10 → error = 0
    df = pl.DataFrame(
        {
            "time_series_id": pl.Series([1, 1], dtype=pl.Int32),
            "valid_time": pl.Series([time, time], dtype=pl.Datetime("us", "UTC")),
            "power_fcst_init_time": pl.Series(
                [init_time, init_time], dtype=pl.Datetime("us", "UTC")
            ),
            "ensemble_member": pl.Series([0, 1], dtype=pl.Int8),
            "nwp_init_time": pl.Series([None, None], dtype=pl.Datetime("us", "UTC")),
            "power_fcst": pl.Series([8.0, 12.0], dtype=pl.Float32),
            "power_fcst_model_name": pl.Series(["stub", "stub"]).cast(pl.Categorical),
            "power_fcst_model_version": pl.Series([1, 1], dtype=pl.Int16),
            "ml_flow_experiment_id": pl.Series([None, None], dtype=pl.Int32),
            "fold_id": pl.Series(["2022", "2022"]).cast(pl.Categorical),
        }
    )
    forecasts = PowerForecast.validate(df, allow_superfluous_columns=True)
    actuals = _make_actuals(1, [time], [10.0])
    result = compute_metrics(forecasts, actuals)
    mae_row = result.filter(pl.col("metric_name") == "mae")
    assert math.isclose(mae_row["metric_value"][0], 0.0, abs_tol=1e-5)


def test_compute_metrics_multiple_folds():
    """Metrics are computed independently per fold."""
    times = [_utc(2022, 1, 1, 0, 0), _utc(2023, 1, 1, 0, 0)]
    actuals = _make_actuals(1, times, [10.0, 10.0])
    fold1 = _make_cv_forecasts(1, [times[0]], [15.0], fold_id="2022")
    fold2 = _make_cv_forecasts(1, [times[1]], [10.0], fold_id="2023")
    forecasts = PowerForecast.validate(pl.concat([fold1, fold2]), allow_superfluous_columns=True)
    result = compute_metrics(forecasts, actuals)
    fold1_mae = result.filter((pl.col("fold_id") == "2022") & (pl.col("metric_name") == "mae"))
    fold2_mae = result.filter((pl.col("fold_id") == "2023") & (pl.col("metric_name") == "mae"))
    assert math.isclose(fold1_mae["metric_value"][0], 5.0, rel_tol=1e-5)
    assert math.isclose(fold2_mae["metric_value"][0], 0.0, abs_tol=1e-5)


def test_compute_metrics_raises_on_no_join():
    """If forecasts and actuals have no overlapping times, raise ValueError."""
    actuals = _make_actuals(1, [_utc(2022, 6, 1)], [10.0])
    forecasts = PowerForecast.validate(
        _make_cv_forecasts(1, [_utc(2023, 6, 1)], [10.0]), allow_superfluous_columns=True
    )
    with pytest.raises(ValueError, match="No rows"):
        compute_metrics(forecasts, actuals)
