"""Tests that the per-row scores reproduce `ml_core.metrics.compute_metrics`."""

from datetime import UTC, datetime, timedelta

import numpy as np
import patito as pt
import polars as pl
import pytest
from contracts.power_schemas import (
    LIST_OF_TIME_SERIES_TYPES,
    EffectiveCapacity,
    PowerForecast,
    PowerTimeSeries,
    TimeSeriesMetadata,
)
from ml_core.metrics import compute_metrics
from per_run_scores import (
    lead_band_means,
    leaderboard_means,
    normalised_by_capacity,
    per_run_scores,
)
from report_tables import markdown_table

CAPACITIES = {1: 10.0, 2: 40.0}


def _utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=UTC)


def _forecasts(*, rng: np.random.Generator) -> pl.DataFrame:
    """Two series, two initialisation times, three valid times, five unequal members each."""
    rows = [
        {
            "time_series_id": series_id,
            "power_fcst_init_time": init,
            "valid_time": init + timedelta(hours=step),
            "ensemble_member": member,
            "nwp_init_time": init,
            "power_fcst": float(rng.normal(5.0 * series_id, 2.0 * series_id)),
        }
        for series_id in CAPACITIES
        for init in (_utc(2022, 1, 1, 0), _utc(2022, 1, 1, 12))
        for step in (1, 2, 12, 40, 200)
        for member in range(3 + step % 4)
    ]
    frame = pl.DataFrame(rows).cast(
        {"time_series_id": pl.Int32, "ensemble_member": pl.Int8, "power_fcst": pl.Float32}
    )
    return frame.with_columns(
        power_fcst_model_name=pl.lit("stub"),
        power_fcst_model_version=pl.lit(1, dtype=pl.Int16),
        ml_flow_experiment_id=pl.lit(None, dtype=pl.Int32),
        experiment_name=pl.lit("stub_exp"),
        fold_id=pl.lit("fold"),
    )


def _actuals(*, forecasts: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    times = forecasts.select("time_series_id", "valid_time").unique().sort("valid_time")
    # Drop every seventh valid time, so some forecast runs have no observation to be scored on.
    times = times.filter(pl.int_range(pl.len()) % 7 != 0)
    return times.rename({"valid_time": "time"}).with_columns(
        power=pl.Series(
            rng.normal(5.0, 3.0, times.height) * times["time_series_id"].to_numpy()
        ).cast(pl.Float32)
    )


def _metadata() -> pt.DataFrame[TimeSeriesMetadata]:
    ids = list(CAPACITIES)
    n = len(ids)
    return TimeSeriesMetadata.validate(
        pl.DataFrame(
            {
                "time_series_id": pl.Series(ids, dtype=pl.Int32),
                "time_series_name": [f"S{i}" for i in ids],
                "time_series_type": pl.Series(["Disaggregated Demand"] * n).cast(
                    pl.Enum(LIST_OF_TIME_SERIES_TYPES)
                ),
                "units": pl.Series(["MW"] * n).cast(pl.Enum(["MW", "MVA"])),
                "licence_area": pl.Series(["EMids"] * n).cast(pl.Enum(["EMids"])),
                "substation_number": pl.Series([1, 2], dtype=pl.Int32),
                "substation_type": pl.Series(["Primary"] * n).cast(
                    pl.Enum(["BSP", "EHV Customer", "GSP", "HV Customer", "Primary"])
                ),
                "latitude": pl.Series([53.0] * n, dtype=pl.Float32),
                "longitude": pl.Series([-0.5] * n, dtype=pl.Float32),
                "h3_res_5": pl.Series([617700169958293503] * n, dtype=pl.UInt64),
            }
        )
    )


def _capacity() -> pt.DataFrame[EffectiveCapacity]:
    return EffectiveCapacity.validate(
        pl.DataFrame(
            {
                "time_series_id": pl.Series(list(CAPACITIES), dtype=pl.Int32),
                "time": pl.Series([_utc(2026, 1, 1)] * 2, dtype=pl.Datetime("us", "UTC")),
                "effective_capacity_mw": pl.Series(list(CAPACITIES.values()), dtype=pl.Float32),
            }
        )
    )


def test_means_equal_compute_metrics_for_every_series():
    """CRPS, MAE, and NMAE per series equal the leaderboard code's values on the same frames."""
    rng = np.random.default_rng(0)
    forecasts = _forecasts(rng=rng)
    actuals = _actuals(forecasts=forecasts, rng=rng)
    expected = compute_metrics(
        PowerForecast.validate(forecasts, allow_superfluous_columns=True),
        pt.LazyFrame.from_existing(actuals.lazy()).set_model(PowerTimeSeries),
        _metadata(),
        _capacity(),
    ).filter((pl.col("horizon_slice") == "all") & (pl.col("metric_param") == "all"))

    scores = normalised_by_capacity(
        scores=per_run_scores(forecasts=forecasts, actuals=actuals),
        capacity=_capacity().select("time_series_id", "effective_capacity_mw"),
    )
    got = leaderboard_means(scores=scores)

    for metric in ("crps", "mae", "nmae"):
        wanted = expected.filter(pl.col("metric_name") == metric).sort("time_series_id")
        assert wanted.height == 2
        np.testing.assert_allclose(
            got[metric].to_numpy(), wanted["metric_value"].to_numpy(), rtol=1e-6
        )


def test_horizon_slices_match_compute_metrics():
    """Slice labels agree with the leaderboard's lead-time bands, which differ across the rows."""
    rng = np.random.default_rng(1)
    forecasts = _forecasts(rng=rng)
    scores = per_run_scores(forecasts=forecasts, actuals=_actuals(forecasts=forecasts, rng=rng))
    by_lead = dict(
        zip(scores["lead_hours"].to_list(), scores["horizon_slice"].to_list(), strict=True)
    )
    assert by_lead[1.0] == "intraday"
    assert by_lead[2.0] == "intraday"
    assert by_lead[40.0] == "short_medium_range"


def test_runs_with_different_initialisation_times_are_not_pooled():
    """Two runs forecasting one valid time give two rows, not one blended ensemble."""
    rng = np.random.default_rng(2)
    forecasts = _forecasts(rng=rng).filter(pl.col("time_series_id") == 1)
    both = forecasts.with_columns(
        valid_time=pl.lit(_utc(2022, 1, 2, 0)).cast(pl.Datetime("us", "UTC"))
    )
    actuals = pl.DataFrame(
        {"time_series_id": [1], "time": [_utc(2022, 1, 2, 0)], "power": [3.0]}
    ).cast({"time_series_id": pl.Int32, "power": pl.Float32})
    scores = per_run_scores(forecasts=both, actuals=actuals)
    assert scores.height == 2


def test_duplicate_actuals_do_not_double_the_members():
    """A duplicated observation leaves the CRPS unchanged."""
    rng = np.random.default_rng(3)
    forecasts = _forecasts(rng=rng)
    actuals = _actuals(forecasts=forecasts, rng=rng)
    once = per_run_scores(forecasts=forecasts, actuals=actuals)
    twice = per_run_scores(forecasts=forecasts, actuals=pl.concat([actuals, actuals]))
    np.testing.assert_allclose(once["crps"].to_numpy(), twice["crps"].to_numpy())


def test_missing_capacity_raises():
    """A scored series with no capacity row is named, not silently dropped."""
    rng = np.random.default_rng(4)
    forecasts = _forecasts(rng=rng)
    scores = per_run_scores(forecasts=forecasts, actuals=_actuals(forecasts=forecasts, rng=rng))
    capacity = pl.DataFrame({"time_series_id": [1], "effective_capacity_mw": [10.0]}).cast(
        {"time_series_id": pl.Int32}
    )
    with pytest.raises(ValueError, match="series \\[2\\]"):
        normalised_by_capacity(scores=scores, capacity=capacity)


def test_lead_bands_use_their_own_rows_and_all_uses_every_row():
    """A 1 h, a 60 h, and a 100 h lead: `days 0 to 2` sees the first two, `days 3 to 10` the third."""
    rows = pl.DataFrame(
        {
            "time_series_id": [1, 1, 1],
            "lead_hours": [1.0, 60.0, 100.0],
            "crps": [1.0, 100.0, 4.0],
            "abs_error": [1.0, 100.0, 4.0],
            "abs_error_norm": [0.1, 10.0, 0.4],
        }
    ).cast({"time_series_id": pl.Int32})
    means = lead_band_means(scores=rows).sort("lead_band")
    by_band = dict(zip(means["lead_band"].to_list(), means["mae"].to_list(), strict=True))
    assert by_band == pytest.approx({"all": 35.0, "days 0 to 2": 50.5, "days 3 to 10": 4.0})


def test_markdown_table_rounds_floats_and_blanks_nulls():
    """Floats print to the requested places, nulls print empty, other values print as-is."""
    frame = pl.DataFrame({"arm": ["a", "b"], "crps": [1.23456, None]})
    assert markdown_table(frame=frame, decimals=2).splitlines() == [
        "| arm | crps |",
        "|---|---|",
        "| a | 1.23 |",
        "| b |  |",
    ]


def test_slice_means_equal_compute_metrics_per_horizon_slice():
    """CRPS per series and horizon slice equals the leaderboard's, across all four slices."""
    rng = np.random.default_rng(5)
    forecasts = _forecasts(rng=rng)
    actuals = _actuals(forecasts=forecasts, rng=rng)
    expected = compute_metrics(
        PowerForecast.validate(forecasts, allow_superfluous_columns=True),
        pt.LazyFrame.from_existing(actuals.lazy()).set_model(PowerTimeSeries),
        _metadata(),
        _capacity(),
    ).filter((pl.col("metric_name") == "crps") & (pl.col("metric_param") == "all"))
    scores = per_run_scores(forecasts=forecasts, actuals=actuals)
    got = scores.group_by("time_series_id", "horizon_slice").agg(pl.col("crps").mean())
    assert set(got["horizon_slice"].cast(pl.String)) == {
        "intraday",
        "day_ahead",
        "short_medium_range",
        "extended_range",
    }
    merged = got.join(
        expected.select("time_series_id", "horizon_slice", "metric_value"),
        on=["time_series_id", "horizon_slice"],
    )
    assert merged.height == got.height
    np.testing.assert_allclose(
        merged["crps"].to_numpy(), merged["metric_value"].to_numpy(), rtol=1e-6
    )


def test_duplicate_actuals_leave_compute_metrics_and_the_wrapper_in_agreement():
    """A duplicated observation changes neither the wrapper nor `compute_metrics`, and they agree."""
    rng = np.random.default_rng(6)
    forecasts = _forecasts(rng=rng)
    actuals = pl.concat([_actuals(forecasts=forecasts, rng=rng)] * 2)
    expected = compute_metrics(
        PowerForecast.validate(forecasts, allow_superfluous_columns=True),
        pt.LazyFrame.from_existing(actuals.lazy()).set_model(PowerTimeSeries),
        _metadata(),
        _capacity(),
    ).filter(
        (pl.col("horizon_slice") == "all")
        & (pl.col("metric_param") == "all")
        & (pl.col("metric_name") == "crps")
    )
    got = leaderboard_means(scores=per_run_scores(forecasts=forecasts, actuals=actuals))
    np.testing.assert_allclose(
        got.sort("time_series_id")["crps"].to_numpy(),
        expected.sort("time_series_id")["metric_value"].to_numpy(),
        rtol=1e-6,
    )
