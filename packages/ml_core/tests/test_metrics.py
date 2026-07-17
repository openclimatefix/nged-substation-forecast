"""Tests for compute_metrics() and build_mlflow_aggregate_metrics()."""

import math
from datetime import datetime, timedelta, timezone

import patito as pt
import polars as pl
import pytest
from contracts.ml_schemas import Metrics
from contracts.power_schemas import (
    LIST_OF_TIME_SERIES_TYPES,
    EffectiveCapacity,
    PowerForecast,
    PowerTimeSeries,
    TimeSeriesMetadata,
)
from ml_core.metrics import build_mlflow_aggregate_metrics, compute_metrics


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
    init_times: list[datetime] | None = None,
) -> pt.DataFrame[PowerForecast]:
    n = len(times)
    if init_times is None:
        init_times = [_utc(2022, 1, 1)] * n
    df = pl.DataFrame(
        {
            "time_series_id": pl.Series([time_series_id] * n, dtype=pl.Int32),
            "valid_time": pl.Series(times, dtype=pl.Datetime("us", "UTC")),
            "power_fcst_init_time": pl.Series(init_times, dtype=pl.Datetime("us", "UTC")),
            "ensemble_member": pl.Series([0] * n, dtype=pl.Int8),
            "nwp_init_time": pl.Series([None] * n, dtype=pl.Datetime("us", "UTC")),
            "power_fcst": pl.Series(power_fcst, dtype=pl.Float32),
            "power_fcst_model_name": pl.Series(["stub"] * n, dtype=pl.String),
            "power_fcst_model_version": pl.Series([1] * n, dtype=pl.Int16),
            "ml_flow_experiment_id": pl.Series([None] * n, dtype=pl.Int32),
            "experiment_name": pl.Series(["stub_exp"] * n, dtype=pl.String),
            "fold_id": pl.Series([fold_id] * n, dtype=pl.String),
        }
    )
    return PowerForecast.validate(df, allow_superfluous_columns=True)


def _make_metadata(
    time_series_ids: list[int], time_series_type: str = "Disaggregated Demand"
) -> pt.DataFrame[TimeSeriesMetadata]:
    n = len(time_series_ids)
    return TimeSeriesMetadata.validate(
        pl.DataFrame(
            {
                "time_series_id": pl.Series(time_series_ids, dtype=pl.Int32),
                "time_series_name": [f"Substation {i}" for i in time_series_ids],
                "time_series_type": pl.Series([time_series_type] * n).cast(
                    pl.Enum(LIST_OF_TIME_SERIES_TYPES)
                ),
                "units": pl.Series(["MW"] * n).cast(pl.Enum(["MW", "MVA"])),
                "licence_area": pl.Series(["EMids"] * n).cast(pl.Enum(["EMids"])),
                "substation_number": pl.Series([i + 1 for i in range(n)], dtype=pl.Int32),
                "substation_type": pl.Series(["Primary"] * n).cast(
                    pl.Enum(["BSP", "EHV Customer", "GSP", "HV Customer", "Primary"])
                ),
                "latitude": pl.Series([53.0] * n, dtype=pl.Float32),
                "longitude": pl.Series([-0.5] * n, dtype=pl.Float32),
                "h3_res_5": pl.Series([617700169958293503] * n, dtype=pl.UInt64),
            }
        )
    )


# ---------------------------------------------------------------------------
# compute_metrics tests
# ---------------------------------------------------------------------------


def test_compute_metrics_returns_metrics_schema():
    times = [_utc(2022, 1, 1, 0, 0), _utc(2022, 1, 1, 0, 30)]
    actuals = _make_actuals(1, times, [10.0, 10.0])
    forecasts = _make_cv_forecasts(1, times, [12.0, 8.0])
    result = compute_metrics(forecasts, actuals, _make_metadata([1]), _make_capacity([1], [10.0]))
    assert isinstance(result, pl.DataFrame)
    Metrics.validate(result, allow_superfluous_columns=True)


def test_compute_metrics_mae_correctness():
    """MAE should equal mean absolute error of (fcst - actual)."""
    times = [_utc(2022, 1, 1, 0, 0), _utc(2022, 1, 1, 0, 30)]
    actuals = _make_actuals(1, times, [10.0, 10.0])
    # errors: +2, -2 → MAE = 2.0
    forecasts = _make_cv_forecasts(1, times, [12.0, 8.0])
    result = compute_metrics(forecasts, actuals, _make_metadata([1]), _make_capacity([1], [10.0]))
    mae_row = result.filter((pl.col("metric_name") == "mae") & (pl.col("horizon_slice") == "all"))
    assert math.isclose(mae_row["metric_value"][0], 2.0, rel_tol=1e-5)


def test_compute_metrics_mbe_sign():
    """MBE is positive when forecasts over-predict."""
    times = [_utc(2022, 1, 1, 0, 0), _utc(2022, 1, 1, 0, 30)]
    actuals = _make_actuals(1, times, [10.0, 10.0])
    forecasts = _make_cv_forecasts(1, times, [15.0, 15.0])  # over-predict by 5
    result = compute_metrics(forecasts, actuals, _make_metadata([1]), _make_capacity([1], [10.0]))
    mbe_row = result.filter((pl.col("metric_name") == "mbe") & (pl.col("horizon_slice") == "all"))
    assert mbe_row["metric_value"][0] > 0


def test_compute_metrics_nmae_normalisation():
    """NMAE = MAE / effective_capacity_mw. With capacity 10 and MAE=2, NMAE=0.2."""
    times = [_utc(2022, 1, 1, 0, 0), _utc(2022, 1, 1, 0, 30)]
    actuals = _make_actuals(1, times, [10.0, 10.0])
    forecasts = _make_cv_forecasts(1, times, [12.0, 8.0])  # MAE=2, capacity=10
    result = compute_metrics(forecasts, actuals, _make_metadata([1]), _make_capacity([1], [10.0]))
    nmae_row = result.filter((pl.col("metric_name") == "nmae") & (pl.col("horizon_slice") == "all"))
    assert math.isclose(nmae_row["metric_value"][0], 0.2, rel_tol=1e-5)


def _make_capacity(
    time_series_ids: list[int], effective_capacity_mw: list[float]
) -> pt.DataFrame[EffectiveCapacity]:
    n = len(time_series_ids)
    return EffectiveCapacity.validate(
        pl.DataFrame(
            {
                "time_series_id": pl.Series(time_series_ids, dtype=pl.Int32),
                "time": pl.Series([_utc(2026, 1, 1)] * n, dtype=pl.Datetime("us", "UTC")),
                "effective_capacity_mw": pl.Series(effective_capacity_mw, dtype=pl.Float32),
            }
        )
    )


def test_compute_metrics_nmae_uses_supplied_capacity():
    """When a capacity frame is supplied, NMAE = MAE / effective_capacity_mw, not the window P99."""
    times = [_utc(2022, 1, 1, 0, 0), _utc(2022, 1, 1, 0, 30)]
    actuals = _make_actuals(1, times, [10.0, 10.0])  # window P99(|actual|) = 10
    forecasts = _make_cv_forecasts(1, times, [12.0, 8.0])  # MAE = 2
    # Capacity of 20 → NMAE = 2/20 = 0.1, distinct from the window-P99 result of 2/10 = 0.2.
    capacity = _make_capacity([1], [20.0])
    result = compute_metrics(forecasts, actuals, _make_metadata([1]), capacity)
    nmae_row = result.filter((pl.col("metric_name") == "nmae") & (pl.col("horizon_slice") == "all"))
    assert math.isclose(nmae_row["metric_value"][0], 0.1, rel_tol=1e-5)


def test_compute_metrics_raises_when_series_missing_capacity():
    """A scored series absent from `capacity` is a loud error, not a silent fallback."""
    times = [_utc(2022, 1, 1, 0, 0), _utc(2022, 1, 1, 0, 30)]
    actuals = _make_actuals(1, times, [10.0, 10.0])
    forecasts = _make_cv_forecasts(1, times, [12.0, 8.0])
    # Capacity covers only ts 99, not the scored ts 1.
    capacity = _make_capacity([99], [5.0])
    with pytest.raises(ValueError, match="No effective_capacity row"):
        compute_metrics(forecasts, actuals, _make_metadata([1]), capacity)


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
            "power_fcst_model_name": pl.Series(["stub", "stub"], dtype=pl.String),
            "power_fcst_model_version": pl.Series([1, 1], dtype=pl.Int16),
            "ml_flow_experiment_id": pl.Series([None, None], dtype=pl.Int32),
            "experiment_name": pl.Series(["stub_exp", "stub_exp"], dtype=pl.String),
            "fold_id": pl.Series(["2022", "2022"], dtype=pl.String),
        }
    )
    forecasts = PowerForecast.validate(df, allow_superfluous_columns=True)
    actuals = _make_actuals(1, [time], [10.0])
    result = compute_metrics(forecasts, actuals, _make_metadata([1]), _make_capacity([1], [10.0]))
    mae_row = result.filter(pl.col("metric_name") == "mae")
    assert math.isclose(mae_row["metric_value"][0], 0.0, abs_tol=1e-5)


def test_compute_metrics_multiple_folds():
    """Metrics are computed independently per fold."""
    times = [_utc(2022, 1, 1, 0, 0), _utc(2023, 1, 1, 0, 0)]
    actuals = _make_actuals(1, times, [10.0, 10.0])
    fold1 = _make_cv_forecasts(1, [times[0]], [15.0], fold_id="2022")
    fold2 = _make_cv_forecasts(1, [times[1]], [10.0], fold_id="2023")
    forecasts = PowerForecast.validate(pl.concat([fold1, fold2]), allow_superfluous_columns=True)
    result = compute_metrics(forecasts, actuals, _make_metadata([1]), _make_capacity([1], [10.0]))
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
        compute_metrics(forecasts, actuals, _make_metadata([1]), _make_capacity([1], [10.0]))


def test_compute_metrics_populates_time_series_type():
    """time_series_type from metadata is joined onto each metric row."""
    times = [_utc(2022, 1, 1, 0, 0)]
    actuals = _make_actuals(1, times, [10.0])
    forecasts = _make_cv_forecasts(1, times, [10.0])
    result = compute_metrics(
        forecasts, actuals, _make_metadata([1], "PV"), _make_capacity([1], [10.0])
    )
    assert (result["time_series_type"] == "PV").all()


def test_compute_metrics_unknown_series_gets_null_type():
    """Series absent from metadata receive null time_series_type (left join)."""
    times = [_utc(2022, 1, 1, 0, 0)]
    actuals = _make_actuals(1, times, [10.0])
    forecasts = _make_cv_forecasts(1, times, [10.0])
    # ts_id=1 is absent — metadata only knows about ts_id=99.
    result = compute_metrics(forecasts, actuals, _make_metadata([99]), _make_capacity([1], [10.0]))
    assert result["time_series_type"].is_null().all()


# ---------------------------------------------------------------------------
# horizon-slice tests
# ---------------------------------------------------------------------------


def _slice_mae(result: pl.DataFrame, horizon_slice: str) -> float:
    """Extract the single MAE value for one horizon slice from a Metrics frame."""
    rows = result.filter(
        (pl.col("metric_name") == "mae") & (pl.col("horizon_slice") == horizon_slice)
    )
    assert rows.height == 1
    return float(rows["metric_value"][0])


def test_compute_metrics_horizon_slice_band_boundaries():
    """Lead times land in the correct left-closed bands, including exact boundary values.

    One forecast run (single init time) with valid_times at lead times 0 h and 5.5 h
    (intraday), 6 h and 35.5 h (day_ahead), 36 h and 167.5 h (short_medium_range), and
    168 h and 336 h (extended_range). Distinct errors per band make per-slice MAE reveal
    band membership.
    """
    init_time = _utc(2022, 1, 1)
    leads_hours = [0.0, 5.5, 6.0, 35.5, 36.0, 167.5, 168.0, 336.0]
    times = [init_time + timedelta(hours=h) for h in leads_hours]
    # errors: intraday 1, 2 → MAE 1.5; day_ahead 3, 4 → 3.5; smr 5, 6 → 5.5; extended 7, 8 → 7.5
    errors = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    actuals = _make_actuals(1, times, [10.0] * len(times))
    forecasts = _make_cv_forecasts(1, times, [10.0 + e for e in errors])
    result = compute_metrics(forecasts, actuals, _make_metadata([1]), _make_capacity([1], [10.0]))
    assert math.isclose(_slice_mae(result, "intraday"), 1.5, rel_tol=1e-5)
    assert math.isclose(_slice_mae(result, "day_ahead"), 3.5, rel_tol=1e-5)
    assert math.isclose(_slice_mae(result, "short_medium_range"), 5.5, rel_tol=1e-5)
    assert math.isclose(_slice_mae(result, "extended_range"), 7.5, rel_tol=1e-5)
    assert math.isclose(_slice_mae(result, "all"), 4.5, rel_tol=1e-5)


def test_compute_metrics_all_slice_is_row_weighted():
    """The "all" slice averages over every scored row, not over the per-slice metric values."""
    init_time = _utc(2022, 1, 1)
    leads_hours = [0.0, 0.5, 5.5, 6.0, 35.5, 36.0, 168.0]
    times = [init_time + timedelta(hours=h) for h in leads_hours]
    # intraday errors 1, 2, 3 → MAE 2; day_ahead 6, 8 → 7; smr 9 → 9; extended 11 → 11.
    errors = [1.0, 2.0, 3.0, 6.0, 8.0, 9.0, 11.0]
    actuals = _make_actuals(1, times, [10.0] * len(times))
    forecasts = _make_cv_forecasts(1, times, [10.0 + e for e in errors])
    result = compute_metrics(forecasts, actuals, _make_metadata([1]), _make_capacity([1], [10.0]))
    # Row-weighted mean = 40/7 ≈ 5.714; the mean of the four slice MAEs would be 7.25.
    assert math.isclose(_slice_mae(result, "all"), 40.0 / 7.0, rel_tol=1e-5)


def test_compute_metrics_collapses_ensemble_per_forecast_run():
    """Runs covering the same valid_time at different lead times are scored independently.

    Two runs forecast the same valid_time: one predicts 8, the other 12, actual is 10.
    Per-run scoring gives errors −2 and +2 → MAE 2. Pooling the runs into one lagged-ensemble
    mean would give a single forecast of 10 → MAE 0.
    """
    valid_time = _utc(2022, 1, 2)
    init_times = [_utc(2022, 1, 1, 0), _utc(2022, 1, 1, 6)]  # leads 24 h and 18 h: day_ahead
    actuals = _make_actuals(1, [valid_time], [10.0])
    forecasts = _make_cv_forecasts(1, [valid_time, valid_time], [8.0, 12.0], init_times=init_times)
    result = compute_metrics(forecasts, actuals, _make_metadata([1]), _make_capacity([1], [10.0]))
    assert math.isclose(_slice_mae(result, "day_ahead"), 2.0, rel_tol=1e-5)
    assert math.isclose(_slice_mae(result, "all"), 2.0, rel_tol=1e-5)


def test_compute_metrics_negative_lead_time_raises():
    """A valid_time before power_fcst_init_time is corrupted data and must fail loudly."""
    valid_time = _utc(2022, 1, 1)
    actuals = _make_actuals(1, [valid_time], [10.0])
    forecasts = _make_cv_forecasts(1, [valid_time], [10.0], init_times=[_utc(2022, 1, 2)])
    with pytest.raises(ValueError, match="negative lead time"):
        compute_metrics(forecasts, actuals, _make_metadata([1]), _make_capacity([1], [10.0]))


# ---------------------------------------------------------------------------
# build_mlflow_aggregate_metrics tests
# ---------------------------------------------------------------------------


def _make_metrics_df(
    rmse_by_type: dict[str, float],
) -> pl.DataFrame:
    """Build a minimal Metrics-shaped DataFrame for aggregate metric tests."""
    rows = []
    for ts_type, rmse in rmse_by_type.items():
        rows.append(
            {
                "time_series_id": 1,
                "metric_name": "rmse",
                "metric_value": rmse,
                "time_series_type": ts_type,
                "horizon_slice": "all",
                "metric_param": "all",
            }
        )
    return pl.DataFrame(rows).cast(
        {
            "time_series_id": pl.Int32,
            "metric_value": pl.Float32,
        }
    )


def test_build_mlflow_aggregate_metrics_all_key():
    """rmse__all is the mean RMSE across all series."""
    df = _make_metrics_df({"PV": 2.0, "Wind": 4.0})
    result = build_mlflow_aggregate_metrics(df)
    assert math.isclose(result["rmse__all"], 3.0, rel_tol=1e-5)


def test_build_mlflow_aggregate_metrics_per_type_keys():
    """Per-type keys use slugified type names."""
    df = _make_metrics_df({"PV": 2.0, "Disaggregated Demand": 6.0})
    result = build_mlflow_aggregate_metrics(df)
    assert "rmse__pv" in result
    assert "rmse__disaggregated_demand" in result
    assert math.isclose(result["rmse__pv"], 2.0, rel_tol=1e-5)
    assert math.isclose(result["rmse__disaggregated_demand"], 6.0, rel_tol=1e-5)


def test_build_mlflow_aggregate_metrics_other_demand_slug():
    """Parentheses in type names are stripped from the slug."""
    df = _make_metrics_df({"Other (Demand)": 3.0})
    result = build_mlflow_aggregate_metrics(df)
    assert "rmse__other_demand" in result


def test_build_mlflow_aggregate_metrics_sliced_keys():
    """Non-"all" horizon slices produce overall {metric}__all__{slice} keys only.

    The "all"-slice keys must ignore sliced rows, and no per-type sliced keys are emitted.
    """
    rows = [
        {
            "time_series_id": 1,
            "metric_name": "rmse",
            "metric_value": 2.0,
            "time_series_type": "PV",
            "horizon_slice": "all",
            "metric_param": "all",
        },
        {
            "time_series_id": 1,
            "metric_name": "rmse",
            "metric_value": 4.0,
            "time_series_type": "PV",
            "horizon_slice": "day_ahead",
            "metric_param": "all",
        },
        {
            "time_series_id": 2,
            "metric_name": "rmse",
            "metric_value": 6.0,
            "time_series_type": "Wind",
            "horizon_slice": "day_ahead",
            "metric_param": "all",
        },
    ]
    df = pl.DataFrame(rows).cast({"time_series_id": pl.Int32, "metric_value": pl.Float32})
    result = build_mlflow_aggregate_metrics(df)
    # Sliced key is the mean over both series' day_ahead rows.
    assert math.isclose(result["rmse__all__day_ahead"], 5.0, rel_tol=1e-5)
    # "all"-slice keys ignore the sliced rows.
    assert math.isclose(result["rmse__all"], 2.0, rel_tol=1e-5)
    assert math.isclose(result["rmse__pv"], 2.0, rel_tol=1e-5)
    # No per-type sliced keys.
    assert "rmse__pv__day_ahead" not in result
    assert "rmse__wind__day_ahead" not in result


def test_build_mlflow_aggregate_metrics_null_type_excluded_from_per_type():
    """Series with null time_series_type contribute to 'all' but not to per-type keys."""
    rows = [
        {
            "time_series_id": 1,
            "metric_name": "rmse",
            "metric_value": 2.0,
            "time_series_type": "PV",
            "horizon_slice": "all",
            "metric_param": "all",
        },
        {
            "time_series_id": 2,
            "metric_name": "rmse",
            "metric_value": 4.0,
            "time_series_type": None,
            "horizon_slice": "all",
            "metric_param": "all",
        },
    ]
    df = pl.DataFrame(rows).cast({"time_series_id": pl.Int32, "metric_value": pl.Float32})
    result = build_mlflow_aggregate_metrics(df)
    # null type: only "pv" per-type key, not a null-type key
    assert "rmse__pv" in result
    assert all("__none" not in k and "null" not in k for k in result)
    # "all" includes both series
    assert math.isclose(result["rmse__all"], 3.0, rel_tol=1e-5)
