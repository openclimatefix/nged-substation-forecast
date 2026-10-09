import itertools
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import patito as pt
import polars as pl
import pytest
from baseline_forecasters import climatology
from baseline_forecasters.climatology import (
    CLIMATOLOGY_MEMBER_COUNT,
    CLIMATOLOGY_QUANTILE_COLUMNS,
    ClimatologyForecaster,
    _local_calendar_cell_keys,
)
from contracts.ml_schemas import AllFeatures
from contracts.power_schemas import PowerForecast
from ml_core.base_forecaster import BaseForecasterConfig
from ml_core.features.feature_engineer import DEFAULT_LOCAL_TIMEZONE
from polars.testing import assert_frame_equal

_UTC = pl.Datetime("us", "UTC")
_INIT_TIME = datetime(2025, 1, 1, tzinfo=UTC)
_KEY_COLUMNS = ["time_series_id", "local_month", "local_half_hour_of_day", "local_is_weekend"]
_LOGGER_NAME = "baseline_forecasters.climatology"


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def _features(rows: list[tuple[int, datetime, float | None]]) -> pt.LazyFrame[AllFeatures]:
    """Engineered rows of ``(time_series_id, valid_time, power)``, all issued at ``_INIT_TIME``."""
    frame = pl.DataFrame(
        {
            "time_series_id": [row[0] for row in rows],
            "valid_time": [row[1] for row in rows],
            "power": [row[2] for row in rows],
            "power_fcst_init_time": [_INIT_TIME] * len(rows),
        },
        schema={
            "time_series_id": pl.Int32,
            "valid_time": _UTC,
            "power": pl.Float32,
            "power_fcst_init_time": _UTC,
        },
    )
    return pt.LazyFrame.from_existing(frame.lazy()).set_model(AllFeatures)


def _forecaster(**config: object) -> ClimatologyForecaster:
    return ClimatologyForecaster(BaseForecasterConfig(selected_features=set(), **config))


def _january_weekday_samples() -> list[tuple[int, datetime, float | None]]:
    """Five samples at local January half-hour 0 on weekdays; the first is Monday 00:00."""
    return [
        (1, _utc(2025, 1, 6 + day), 10.0 * day)  # Monday 6 January to Friday 10 January
        for day in range(5)
    ]


def _expected_members(top: float) -> list[float]:
    return [top * (k + 0.5) / CLIMATOLOGY_MEMBER_COUNT for k in range(CLIMATOLOGY_MEMBER_COUNT)]


def _trained_on_january_weekdays() -> ClimatologyForecaster:
    forecaster = _forecaster()
    forecaster.train(_features(_january_weekday_samples()), [1])
    return forecaster


def test_construction_accepts_no_feature_and_rejects_a_selected_feature() -> None:
    ClimatologyForecaster(BaseForecasterConfig(selected_features=set()))
    with pytest.raises(ValueError, match="selected_features must be empty"):
        ClimatologyForecaster(BaseForecasterConfig(selected_features={"power_lag_168h"}))


def test_train_stores_hand_computed_quantiles_over_the_wrapped_neighbourhood() -> None:
    forecaster = _trained_on_january_weekdays()

    lookup = forecaster._lookup.sort(_KEY_COLUMNS)

    # Months 12, 1, and 2 times half-hours 47, 0, and 1, all weekday.
    assert lookup.select(_KEY_COLUMNS).rows() == [
        (1, month, half_hour, False) for month in (1, 2, 12) for half_hour in (0, 1, 47)
    ]
    expected = _expected_members(40.0)
    assert len(set(expected)) == CLIMATOLOGY_MEMBER_COUNT
    for row in lookup.iter_rows(named=True):
        actual = [row[column] for column in CLIMATOLOGY_QUANTILE_COLUMNS]
        assert actual == pytest.approx(expected, abs=1e-4)
    assert expected[0] == pytest.approx(0.392, abs=1e-3)
    assert expected[25] == pytest.approx(20.0)
    assert expected[50] == pytest.approx(39.608, abs=1e-3)


def test_train_deduplicates_the_targets_that_several_nwp_runs_repeat() -> None:
    samples = _january_weekday_samples()
    # The two largest samples are repeated under three further forecast-run rows each.
    duplicated = samples + [samples[3]] * 3 + [samples[4]] * 3
    plain = _forecaster()
    plain.train(_features(samples), [1])
    repeated = _forecaster()
    repeated.train(_features(duplicated), [1])

    assert_frame_equal(repeated._lookup, plain._lookup)


def test_predict_emits_members_in_increasing_quantile_order() -> None:
    forecaster = _trained_on_january_weekdays()
    forecast = forecaster.predict(_features([(1, _utc(2025, 1, 14), 5.0)])).sort("ensemble_member")

    assert forecast["ensemble_member"].to_list() == list(range(CLIMATOLOGY_MEMBER_COUNT))
    values = forecast["power_fcst"].to_list()
    assert values == pytest.approx(_expected_members(40.0), abs=1e-4)
    assert all(low < high for low, high in itertools.pairwise(values))
    assert values[25] == pytest.approx(20.0)


def test_cell_keys_are_local_across_both_clock_changes_and_local_midnight() -> None:
    cases = [
        # Saturday 00:30 BST.
        (_utc(2025, 6, 6, 23, 30), (6, 1, True)),
        # Tuesday 1 July 00:30 BST.
        (_utc(2025, 6, 30, 23, 30), (7, 1, False)),
        # Friday 23:30 GMT.
        (_utc(2025, 1, 17, 23, 30), (1, 47, False)),
        # Sunday 30 March 02:00 BST, just after the spring clock change.
        (_utc(2025, 3, 30, 1, 0), (3, 4, True)),
        # Sunday 26 October 01:30 BST, then 01:30 GMT: the doubled local hour is one cell.
        (_utc(2025, 10, 26, 0, 30), (10, 3, True)),
        (_utc(2025, 10, 26, 1, 30), (10, 3, True)),
    ]
    frame = pl.DataFrame({"valid_time": [case[0] for case in cases]}, schema={"valid_time": _UTC})

    keys = frame.select(
        **_local_calendar_cell_keys(pl.col("valid_time"), time_zone=DEFAULT_LOCAL_TIMEZONE)
    )

    assert keys.rows() == [case[1] for case in cases]
    assert keys.schema["local_month"] == pl.Int8
    assert keys.schema["local_half_hour_of_day"] == pl.Int8


def test_predict_drops_an_unseen_cell_and_logs_but_fills_a_neighbour_cell(
    caplog: pytest.LogCaptureFixture,
) -> None:
    forecaster = _trained_on_january_weekdays()
    rows = [
        (1, _utc(2025, 2, 4, 0, 30), 0.0),  # Tuesday, February, half-hour 1: a neighbour's sample.
        (1, _utc(2025, 1, 11), 0.0),  # Saturday: no weekend sample.
        (1, _utc(2025, 3, 4), 0.0),  # Tuesday, March: two months from the samples.
    ]

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        forecast = forecaster.predict(_features(rows))

    assert forecast["valid_time"].unique().to_list() == [_utc(2025, 2, 4, 0, 30)]
    assert forecast.sort("ensemble_member")["power_fcst"].to_list() == pytest.approx(
        _expected_members(40.0), abs=1e-4
    )
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "dropped 2 forecast rows" in warnings[0]
    assert "[1]" in warnings[0]


def test_predict_on_empty_input_returns_a_valid_empty_frame() -> None:
    forecast = _trained_on_january_weekdays().predict(_features([]))

    assert forecast.height == 0
    PowerForecast.validate(forecast)


def test_predict_stamps_the_identity_columns() -> None:
    forecaster = _forecaster(experiment_name="exp_clim", ml_flow_experiment_id=7)
    forecaster.train(_features(_january_weekday_samples()), [1])

    forecast = forecaster.predict(_features([(1, _utc(2025, 1, 14), 5.0)]), fold_id="fold_x")

    assert forecast.height == CLIMATOLOGY_MEMBER_COUNT
    assert forecast["power_fcst_model_name"].unique().to_list() == ["climatology"]
    assert forecast["power_fcst_model_version"].unique().to_list() == [1]
    assert forecast["experiment_name"].unique().to_list() == ["exp_clim"]
    assert forecast["ml_flow_experiment_id"].unique().to_list() == [7]
    assert forecast["fold_id"].unique().to_list() == ["fold_x"]
    assert forecast["nwp_init_time"].is_null().all()


def test_predict_ignores_the_power_of_the_forecast_period() -> None:
    forecaster = _trained_on_january_weekdays()
    rows = [(1, _utc(2025, 1, 14), 5.0), (1, _utc(2025, 1, 15), None)]
    shouted = [(series, valid_time, 1e9) for series, valid_time, _ in rows]

    quiet = forecaster.predict(_features(rows))
    loud = forecaster.predict(_features(shouted))

    assert_frame_equal(
        quiet.sort("valid_time", "ensemble_member"), loud.sort("valid_time", "ensemble_member")
    )


@pytest.mark.parametrize("batch_size", [100, 1])
def test_train_keeps_the_requested_series_with_power_and_keeps_series_apart(
    monkeypatch: pytest.MonkeyPatch, batch_size: int
) -> None:
    monkeypatch.setattr(climatology, "_POOLING_SERIES_PER_BATCH", batch_size)
    rows = [
        (1, _utc(2025, 1, 6), 10.0),
        (1, _utc(2025, 1, 7), 20.0),
        (2, _utc(2025, 1, 6), None),
        (3, _utc(2025, 1, 6), 5.0),
        (4, _utc(2025, 1, 6), 100.0),
        (4, _utc(2025, 1, 7), 200.0),
    ]
    forecaster = _forecaster()

    forecaster.train(_features(rows), [1, 2, 4])

    assert forecaster.trained_time_series_ids == [1, 4]
    lookup = forecaster._lookup
    assert sorted(lookup["time_series_id"].unique().to_list()) == [1, 4]
    shared_cell = lookup.filter(
        pl.col("local_month") == 1,
        pl.col("local_half_hour_of_day") == 0,
        ~pl.col("local_is_weekend"),
    ).sort("time_series_id")
    assert shared_cell["power_quantile_member_25"].to_list() == pytest.approx([15.0, 150.0])
    # The lookup does not depend on how many series are pooled together.
    monkeypatch.setattr(climatology, "_POOLING_SERIES_PER_BATCH", 100)
    one_batch = _forecaster()
    one_batch.train(_features(rows), [1, 2, 4])
    assert_frame_equal(lookup, one_batch._lookup)


def test_save_then_load_round_trips_and_replaces_the_directory(tmp_path: Path) -> None:
    forecaster = _forecaster(experiment_name="exp")
    forecaster.train(_features(_january_weekday_samples()), [1])
    (tmp_path / "stale.ubj").write_text("left over from a larger model")
    rows = [(1, _utc(2025, 1, 14), 5.0)]

    forecaster.save(tmp_path)
    loaded = ClimatologyForecaster.load(tmp_path)

    assert not (tmp_path / "stale.ubj").exists()
    assert loaded.trained_time_series_ids == [1]
    assert loaded.model_params == forecaster.model_params
    assert_frame_equal(loaded._lookup, forecaster._lookup)
    assert_frame_equal(loaded.predict(_features(rows)), forecaster.predict(_features(rows)))
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["model_class"] == "baseline_forecasters.climatology.ClimatologyForecaster"


def test_neighbourhood_is_one_month_and_one_half_hour_with_the_same_day_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rows = [
        (1, _utc(2025, 3, 4, 5), 0.0),  # Tuesday, March, half-hour 10
        (1, _utc(2025, 3, 5, 5), 40.0),  # Wednesday, March, half-hour 10
        (1, _utc(2025, 3, 6, 5, 30), 20.0),  # Thursday, March, half-hour 11
        (1, _utc(2025, 5, 6, 4), 100.0),  # Tuesday, May, half-hour 10 (04:00 UTC is 05:00 BST)
        (1, _utc(2025, 3, 8, 5), 1000.0),  # Saturday, March, half-hour 10
    ]
    forecaster = _forecaster()

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        forecaster.train(_features(rows), [1])

    def members(month: int, half_hour: int, *, weekend: bool) -> list[float] | None:
        cell = forecaster._lookup.filter(
            pl.col("local_month") == month,
            pl.col("local_half_hour_of_day") == half_hour,
            pl.col("local_is_weekend") == weekend,
        )
        if cell.height == 0:
            return None
        assert cell.height == 1
        return [cell[column][0] for column in CLIMATOLOGY_QUANTILE_COLUMNS]

    level = [(k + 0.5) / CLIMATOLOGY_MEMBER_COUNT for k in range(CLIMATOLOGY_MEMBER_COUNT)]
    april = [60 * p if p <= 2 / 3 else 180 * p - 80 for p in level]
    assert members(3, 10, weekend=False) == pytest.approx([40 * p for p in level], abs=1e-4)
    assert members(4, 10, weekend=False) == pytest.approx(april, abs=1e-4)
    assert april[33] == pytest.approx(39.412, abs=1e-3)
    assert april[34] == pytest.approx(41.765, abs=1e-3)
    assert april[50] == pytest.approx(98.235, abs=1e-3)
    assert members(6, 10, weekend=False) == pytest.approx([100.0] * 51)
    assert members(3, 12, weekend=False) == pytest.approx([20.0] * 51)
    assert members(3, 10, weekend=True) == pytest.approx([1000.0] * 51)
    assert members(3, 13, weekend=False) is None
    assert members(7, 10, weekend=False) is None
    assert any("minimum 1," in record.getMessage() for record in caplog.records)
