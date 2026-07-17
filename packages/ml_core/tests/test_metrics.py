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
from ml_core.metrics import (
    NoOverlappingActualsError,
    build_mlflow_aggregate_metrics,
    compute_metrics,
)


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
        # 30 min before the fixtures' customary first valid_time (2022-01-01 00:00): the
        # PowerForecast contract requires valid_time strictly after power_fcst_init_time.
        init_times = [_utc(2021, 12, 31, 23, 30)] * n
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
    init_time = _utc(2021, 12, 31, 23, 30)
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
    """No overlapping times raises the dedicated (batch-skippable) ValueError subclass."""
    actuals = _make_actuals(1, [_utc(2022, 6, 1)], [10.0])
    forecasts = PowerForecast.validate(
        _make_cv_forecasts(1, [_utc(2023, 6, 1)], [10.0]), allow_superfluous_columns=True
    )
    with pytest.raises(NoOverlappingActualsError, match="No rows"):
        compute_metrics(forecasts, actuals, _make_metadata([1]), _make_capacity([1], [10.0]))


def test_compute_metrics_per_series_batching_is_equivalent():
    """Scoring series separately and concatenating equals scoring them in one call.

    This property is what lets the metrics asset score a fold in per-series batches
    (bounded memory) without changing any metric value.
    """
    times = [_utc(2022, 1, 1, 0, 0), _utc(2022, 1, 1, 0, 30)]
    actuals = pt.LazyFrame.from_existing(
        pl.concat(
            [
                _make_actuals(1, times, [10.0, 10.0]).collect().lazy(),
                _make_actuals(2, times, [20.0, 20.0]).collect().lazy(),
            ]
        )
    ).set_model(PowerTimeSeries)
    fcst_1 = _make_cv_forecasts(1, times, [12.0, 8.0])
    fcst_2 = _make_cv_forecasts(2, times, [25.0, 19.0])
    both = PowerForecast.validate(pl.concat([fcst_1, fcst_2]), allow_superfluous_columns=True)
    metadata = _make_metadata([1, 2])
    capacity = _make_capacity([1, 2], [10.0, 20.0])

    whole = compute_metrics(both, actuals, metadata, capacity)
    batched = pl.concat(
        [
            compute_metrics(fcst_1, actuals, metadata, capacity),
            compute_metrics(fcst_2, actuals, metadata, capacity),
        ]
    )

    # metric_param must be a sort key: each group has 13 pinball_loss rows (one per
    # quantile), so sorting by metric_name alone leaves ties in non-deterministic order.
    sort_keys = ["time_series_id", "horizon_slice", "metric_name", "metric_param"]
    assert whole.sort(sort_keys).equals(batched.sort(sort_keys))


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

    One forecast run (single init time) with valid_times at lead times 0.5 h and 5.5 h
    (intraday), 6 h and 35.5 h (day_ahead), 36 h and 167.5 h (short_medium_range), and
    168 h and 336 h (extended_range). Distinct errors per band make per-slice MAE reveal
    band membership. 0.5 h is the shortest deliverable lead: the PowerForecast contract
    requires valid_time strictly after power_fcst_init_time, so lead 0 cannot occur.
    """
    init_time = _utc(2022, 1, 1)
    leads_hours = [0.5, 5.5, 6.0, 35.5, 36.0, 167.5, 168.0, 336.0]
    times = [init_time + timedelta(hours=h) for h in leads_hours]
    # errors: intraday 1, 2 → MAE 1.5; day_ahead 3, 4 → 3.5; smr 5, 6 → 5.5; extended 7, 8 → 7.5
    errors = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    actuals = _make_actuals(1, times, [10.0] * len(times))
    forecasts = _make_cv_forecasts(
        1, times, [10.0 + e for e in errors], init_times=[init_time] * len(times)
    )
    result = compute_metrics(forecasts, actuals, _make_metadata([1]), _make_capacity([1], [10.0]))
    assert math.isclose(_slice_mae(result, "intraday"), 1.5, rel_tol=1e-5)
    assert math.isclose(_slice_mae(result, "day_ahead"), 3.5, rel_tol=1e-5)
    assert math.isclose(_slice_mae(result, "short_medium_range"), 5.5, rel_tol=1e-5)
    assert math.isclose(_slice_mae(result, "extended_range"), 7.5, rel_tol=1e-5)
    assert math.isclose(_slice_mae(result, "all"), 4.5, rel_tol=1e-5)


def test_compute_metrics_all_slice_is_row_weighted():
    """The "all" slice averages over every scored row, not over the per-slice metric values."""
    init_time = _utc(2022, 1, 1)
    leads_hours = [0.5, 1.0, 5.5, 6.0, 35.5, 36.0, 168.0]
    times = [init_time + timedelta(hours=h) for h in leads_hours]
    # intraday errors 1, 2, 3 → MAE 2; day_ahead 6, 8 → 7; smr 9 → 9; extended 11 → 11.
    errors = [1.0, 2.0, 3.0, 6.0, 8.0, 9.0, 11.0]
    actuals = _make_actuals(1, times, [10.0] * len(times))
    forecasts = _make_cv_forecasts(
        1, times, [10.0 + e for e in errors], init_times=[init_time] * len(times)
    )
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
    """A valid_time before power_fcst_init_time is corrupted data and must fail loudly.

    ``PowerForecast.validate`` rejects hindcast rows at construction, so to exercise
    ``compute_metrics``' own defence-in-depth guard we corrupt the frame *after* validation
    (``with_columns`` does not re-validate).
    """
    valid_time = _utc(2022, 1, 1)
    actuals = _make_actuals(1, [valid_time], [10.0])
    forecasts = _make_cv_forecasts(1, [valid_time], [10.0]).with_columns(
        power_fcst_init_time=pl.lit(_utc(2022, 1, 2)).cast(pl.Datetime("us", "UTC"))
    )
    with pytest.raises(ValueError, match="negative lead time"):
        compute_metrics(forecasts, actuals, _make_metadata([1]), _make_capacity([1], [10.0]))


def test_power_forecast_contract_rejects_hindcast_rows():
    """The PowerForecast contract itself refuses valid_time at or before power_fcst_init_time.

    Enforced declaratively via the ``valid_time`` field's Patito constraint;
    ``DataFrameValidationError`` is a ``ValueError`` subclass, so callers treating validation
    failures as ``ValueError`` keep working.
    """
    valid_time = _utc(2022, 1, 1)
    with pytest.raises(pt.exceptions.DataFrameValidationError, match="valid_time"):
        _make_cv_forecasts(1, [valid_time], [10.0], init_times=[valid_time])  # lead 0
    with pytest.raises(pt.exceptions.DataFrameValidationError, match="valid_time"):
        _make_cv_forecasts(1, [valid_time], [10.0], init_times=[_utc(2022, 1, 2)])


# ---------------------------------------------------------------------------
# probabilistic-metric tests
# ---------------------------------------------------------------------------


def _make_ensemble_forecasts(
    time_series_id: int,
    times: list[datetime],
    member_values: list[list[float]],
    init_times: list[datetime] | None = None,
) -> pt.DataFrame[PowerForecast]:
    """Build forecasts with one row per (valid_time, ensemble_member).

    ``member_values[i]`` lists the ensemble-member forecasts for ``times[i]``. Two rows with
    the same ``valid_time`` but different ``init_times`` entries form two separate forecast
    runs, each with its own members.
    """
    if init_times is None:
        init_times = [_utc(2021, 12, 31, 23, 30)] * len(times)
    valid_times: list[datetime] = []
    run_init_times: list[datetime] = []
    members: list[int] = []
    values: list[float] = []
    for valid_time, init_time, run_members in zip(times, init_times, member_values, strict=True):
        for member, value in enumerate(run_members):
            valid_times.append(valid_time)
            run_init_times.append(init_time)
            members.append(member)
            values.append(value)
    n = len(values)
    df = pl.DataFrame(
        {
            "time_series_id": pl.Series([time_series_id] * n, dtype=pl.Int32),
            "valid_time": pl.Series(valid_times, dtype=pl.Datetime("us", "UTC")),
            "power_fcst_init_time": pl.Series(run_init_times, dtype=pl.Datetime("us", "UTC")),
            "ensemble_member": pl.Series(members, dtype=pl.Int8),
            "nwp_init_time": pl.Series([None] * n, dtype=pl.Datetime("us", "UTC")),
            "power_fcst": pl.Series(values, dtype=pl.Float32),
            "power_fcst_model_name": pl.Series(["stub"] * n, dtype=pl.String),
            "power_fcst_model_version": pl.Series([1] * n, dtype=pl.Int16),
            "ml_flow_experiment_id": pl.Series([None] * n, dtype=pl.Int32),
            "experiment_name": pl.Series(["stub_exp"] * n, dtype=pl.String),
            "fold_id": pl.Series(["2022"] * n, dtype=pl.String),
        }
    )
    return PowerForecast.validate(df, allow_superfluous_columns=True)


def _metric_value(
    result: pl.DataFrame,
    metric_name: str,
    horizon_slice: str = "all",
    metric_param: str = "all",
) -> float:
    """Extract the single value of one (metric_name, horizon_slice, metric_param) row."""
    rows = result.filter(
        (pl.col("metric_name") == metric_name)
        & (pl.col("horizon_slice") == horizon_slice)
        & (pl.col("metric_param") == metric_param)
    )
    assert rows.height == 1
    return float(rows["metric_value"][0])


def test_compute_metrics_probabilistic_three_member_toy():
    """Every probabilistic metric matches hand computation on a 3-member toy ensemble.

    Members [1, 2, 4] forecast an actual of 2.5 (one timestamp, one run, m = 3):

    - Fair CRPS = mean|xᵢ − y| − Σᵢ<ⱼ|xᵢ − xⱼ| / (m(m−1))
      = (1.5 + 0.5 + 1.5)/3 − (1 + 3 + 2)/6 = 7/6 − 1 = 1/6.
    - Ensemble mean = 7/3 → error = 7/3 − 2.5 = −1/6, so RMSE = 1/6. Sample variance
      (ddof=1) = 7/3; Fortin-corrected variance = (m+1)/m · 7/3 = 28/9;
      spread_skill_ratio = √(28/9) / (1/6) = 6·√(28/9).
    - Linear empirical quantiles interpolate at position (m−1)·τ: p10 → 1.2, p50 → 2,
      p90 → 3.6, p99 → 3.96.
    - Pinball loss (y = 2.5): p10 → 0.1·1.3 = 0.13; p50 → 0.5·0.5 = 0.25;
      p90 → 0.1·1.1 = 0.11; p99 → 0.01·1.46 = 0.0146.
    - p10–p90 band = [1.2, 3.6] contains 2.5 → PICP = 1.0, interval width = 2.4.
    """
    times = [_utc(2022, 1, 1, 0, 0)]
    actuals = _make_actuals(1, times, [2.5])
    forecasts = _make_ensemble_forecasts(1, times, [[1.0, 2.0, 4.0]])
    result = compute_metrics(forecasts, actuals, _make_metadata([1]), _make_capacity([1], [10.0]))

    assert math.isclose(_metric_value(result, "crps"), 1.0 / 6.0, rel_tol=1e-5)
    assert math.isclose(
        _metric_value(result, "spread_skill_ratio"), 6.0 * math.sqrt(28.0 / 9.0), rel_tol=1e-5
    )
    assert math.isclose(
        _metric_value(result, "pinball_loss", metric_param="p10"), 0.13, rel_tol=1e-5
    )
    assert math.isclose(
        _metric_value(result, "pinball_loss", metric_param="p50"), 0.25, rel_tol=1e-5
    )
    assert math.isclose(
        _metric_value(result, "pinball_loss", metric_param="p90"), 0.11, rel_tol=1e-5
    )
    assert math.isclose(
        _metric_value(result, "pinball_loss", metric_param="p99"), 0.0146, rel_tol=1e-4
    )
    assert math.isclose(_metric_value(result, "picp", metric_param="p10_p90"), 1.0, rel_tol=1e-5)
    assert math.isclose(
        _metric_value(result, "interval_width", metric_param="p10_p90"), 2.4, rel_tol=1e-5
    )
    # mean_pinball_loss is the unweighted mean over all 13 delivery quantiles.
    pinball_rows = result.filter(
        (pl.col("metric_name") == "pinball_loss") & (pl.col("horizon_slice") == "all")
    )
    assert pinball_rows.height == 13
    mean_of_quantile_pinballs = pinball_rows["metric_value"].mean()
    assert isinstance(mean_of_quantile_pinballs, float)
    assert math.isclose(
        _metric_value(result, "mean_pinball_loss"), mean_of_quantile_pinballs, rel_tol=1e-5
    )
    # Pinball loss is non-negative at every quantile.
    assert (pinball_rows["metric_value"] >= 0).all()


def test_compute_metrics_single_member_ensemble_degenerates_gracefully():
    """A deterministic (single-member) forecast is scored honestly, with no null values.

    Fair CRPS reduces to MAE (the pairwise term is defined as 0 at m = 1), the spread-skill
    ratio is 0 (zero spread, not null — ``.var(ddof=1)`` alone would return null), every
    quantile coincides with the forecast so interval widths are 0, and PICP is 0 whenever the
    actual differs from the forecast.
    """
    times = [_utc(2022, 1, 1, 0, 0), _utc(2022, 1, 1, 0, 30)]
    actuals = _make_actuals(1, times, [10.0, 10.0])
    forecasts = _make_cv_forecasts(1, times, [12.0, 8.0])  # single member; MAE = 2
    result = compute_metrics(forecasts, actuals, _make_metadata([1]), _make_capacity([1], [10.0]))

    assert result["metric_value"].is_not_null().all()
    assert math.isclose(_metric_value(result, "crps"), _metric_value(result, "mae"), rel_tol=1e-6)
    assert _metric_value(result, "spread_skill_ratio") == 0.0
    assert _metric_value(result, "interval_width", metric_param="p10_p90") == 0.0
    assert _metric_value(result, "picp", metric_param="p10_p90") == 0.0


def test_compute_metrics_fair_crps_handles_duplicate_members():
    """The sorted-member pairwise identity is exact when members tie.

    Members [2, 2, 5], actual 3.5: CRPS = (1.5 + 1.5 + 1.5)/3 − (0 + 3 + 3)/6 = 1.5 − 1 = 0.5.
    """
    times = [_utc(2022, 1, 1, 0, 0)]
    actuals = _make_actuals(1, times, [3.5])
    forecasts = _make_ensemble_forecasts(1, times, [[2.0, 2.0, 5.0]])
    result = compute_metrics(forecasts, actuals, _make_metadata([1]), _make_capacity([1], [10.0]))
    assert math.isclose(_metric_value(result, "crps"), 0.5, rel_tol=1e-5)


def test_compute_metrics_crps_scored_per_forecast_run():
    """CRPS pools members within a run, never across runs covering the same valid_time.

    Two 2-member runs forecast the same valid_time (actual 10): members [7, 9] give
    CRPS = (3 + 1)/2 − 2/2 = 1, and members [11, 13] likewise give 1, so per-run scoring
    averages to 1. Pooling all four members into one lagged ensemble would instead give
    (3 + 1 + 1 + 3)/4 − 20/12 = 1/3.
    """
    valid_time = _utc(2022, 1, 2)
    init_times = [_utc(2022, 1, 1, 0), _utc(2022, 1, 1, 6)]  # leads 24 h and 18 h: day_ahead
    actuals = _make_actuals(1, [valid_time], [10.0])
    forecasts = _make_ensemble_forecasts(
        1, [valid_time, valid_time], [[7.0, 9.0], [11.0, 13.0]], init_times=init_times
    )
    result = compute_metrics(forecasts, actuals, _make_metadata([1]), _make_capacity([1], [10.0]))
    assert math.isclose(_metric_value(result, "crps"), 1.0, rel_tol=1e-5)


def test_compute_metrics_probabilistic_metrics_per_slice():
    """Probabilistic metrics are aggregated per horizon slice, with "all" as the row mean.

    One run: an intraday timestamp with members [7, 9] (CRPS 1) and a day_ahead timestamp
    with members [4, 8] (CRPS = (6 + 2)/2 − 4/2 = 2), both against an actual of 10.
    """
    init_time = _utc(2022, 1, 1)
    times = [init_time + timedelta(hours=0.5), init_time + timedelta(hours=24)]
    actuals = _make_actuals(1, times, [10.0, 10.0])
    forecasts = _make_ensemble_forecasts(
        1, times, [[7.0, 9.0], [4.0, 8.0]], init_times=[init_time] * 2
    )
    result = compute_metrics(forecasts, actuals, _make_metadata([1]), _make_capacity([1], [10.0]))
    assert math.isclose(_metric_value(result, "crps", horizon_slice="intraday"), 1.0, rel_tol=1e-5)
    assert math.isclose(_metric_value(result, "crps", horizon_slice="day_ahead"), 2.0, rel_tol=1e-5)
    assert math.isclose(_metric_value(result, "crps"), 1.5, rel_tol=1e-5)


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


def _parametric_metrics_row(
    metric_name: str,
    metric_param: str,
    metric_value: float,
    horizon_slice: str = "all",
    time_series_id: int = 1,
    time_series_type: str = "PV",
) -> dict[str, object]:
    """One Metrics-shaped row for parametric-key MLflow tests."""
    return {
        "time_series_id": time_series_id,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "time_series_type": time_series_type,
        "horizon_slice": horizon_slice,
        "metric_param": metric_param,
    }


def test_build_mlflow_aggregate_metrics_parametric_key_tokens():
    """Parametric metrics use {name}_{param} key tokens, restricted to the headline allowlist.

    metric_param="all" metrics keep the bare-name token (existing key formats unchanged);
    pinball_loss is logged at p10/p50/p90 only and picp/interval_width at p10_p90 only —
    the full 13-quantile / 6-band detail stays in the Delta table.
    """
    rows = [
        _parametric_metrics_row("crps", "all", 1.5),
        _parametric_metrics_row("pinball_loss", "p10", 2.0),
        _parametric_metrics_row("pinball_loss", "p1", 3.0),  # not in the allowlist
        _parametric_metrics_row("picp", "p10_p90", 0.7),
        _parametric_metrics_row("picp", "p5_p95", 0.8),  # not in the allowlist
        _parametric_metrics_row("interval_width", "p10_p90", 5.0),
    ]
    df = pl.DataFrame(rows).cast({"time_series_id": pl.Int32, "metric_value": pl.Float32})
    result = build_mlflow_aggregate_metrics(df)

    assert math.isclose(result["crps__all"], 1.5, rel_tol=1e-5)
    assert math.isclose(result["pinball_loss_p10__all"], 2.0, rel_tol=1e-5)
    assert math.isclose(result["picp_p10_p90__all"], 0.7, rel_tol=1e-5)
    assert math.isclose(result["interval_width_p10_p90__all"], 5.0, rel_tol=1e-5)
    # The allowlist also applies to the per-type key family.
    assert math.isclose(result["pinball_loss_p10__pv"], 2.0, rel_tol=1e-5)
    # Non-allowlisted params are absent from every key family.
    assert not any(key.startswith("pinball_loss_p1_") for key in result)
    assert not any(key.startswith("picp_p5_p95") for key in result)


def test_build_mlflow_aggregate_metrics_parametric_sliced_keys():
    """Allowlisted parametric metrics get per-slice keys; non-allowlisted ones do not."""
    rows = [
        _parametric_metrics_row("pinball_loss", "p10", 2.0, horizon_slice="day_ahead"),
        _parametric_metrics_row("pinball_loss", "p1", 3.0, horizon_slice="day_ahead"),
    ]
    df = pl.DataFrame(rows).cast({"time_series_id": pl.Int32, "metric_value": pl.Float32})
    result = build_mlflow_aggregate_metrics(df)
    assert math.isclose(result["pinball_loss_p10__all__day_ahead"], 2.0, rel_tol=1e-5)
    assert not any(key.startswith("pinball_loss_p1_") for key in result)
