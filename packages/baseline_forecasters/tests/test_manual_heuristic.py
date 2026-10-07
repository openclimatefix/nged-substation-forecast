import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import patito as pt
import polars as pl
import pytest
import yaml
from _nwp_test_data import cast_to_nwp_dtypes, nwp_records
from _power_test_data import power_at
from baseline_forecasters import ManualHeuristicForecaster
from baseline_forecasters.manual_heuristic import PowerLagsPerNwpRunFeatureEngineer
from contracts.ml_schemas import AllFeatures
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from contracts.weather_schemas import Nwp
from ml_core.base_forecaster import BaseForecasterConfig
from ml_core.features import TabularFeatureEngineer
from polars.testing import assert_frame_equal

_MODEL_YAML = Path(__file__).parents[3] / "conf" / "model" / "manual_heuristic.yaml"
_WEEKLY_LAG_HOURS = [168 * k for k in range(1, 7)]
_ANNUAL_LAG_HOURS = [168 * k for k in range(49, 56)]
_ALL_LAG_HOURS = _WEEKLY_LAG_HOURS + _ANNUAL_LAG_HOURS
_ALL_LAG_FEATURES = {f"power_lag_{hours}h" for hours in _ALL_LAG_HOURS}
_CELL_A = 599423199024775167
_CELL_B = 599423199024775167 + 2**30
_UTC = pl.Datetime("us", "UTC")


def _power_frame(time_series_ids: list[int], start: datetime, end: datetime) -> pl.DataFrame:
    times = pl.datetime_range(start, end, interval="30m", time_zone="UTC", eager=True)
    frames = [
        pl.DataFrame(
            {
                "time_series_id": pl.Series([ts_id] * len(times), dtype=pl.Int32),
                "time": times,
                "power": pl.Series(
                    [power_at(t, offset=7 * ts_id) for t in times.to_list()], dtype=pl.Float32
                ),
            }
        )
        for ts_id in time_series_ids
    ]
    return pl.concat(frames)


def _metadata(cells: dict[int, int]) -> pt.DataFrame[TimeSeriesMetadata]:
    frame = pl.DataFrame(
        {
            "time_series_id": pl.Series(list(cells), dtype=pl.Int32),
            "h3_res_5": pl.Series(list(cells.values()), dtype=pl.UInt64),
        }
    )
    return pt.DataFrame(frame).set_model(TimeSeriesMetadata)


def _nwp(
    runs: list[tuple[datetime, tuple[int, ...], timedelta]], cells: list[int]
) -> pt.LazyFrame[Nwp]:
    """ECMWF-shaped NWP: 3-hourly valid times from each run's first valid offset to 360 h."""
    records = []
    for init_time, members, first_valid_offset in runs:
        valid_times = [
            init_time + timedelta(hours=3 * step)
            for step in range(121)
            if timedelta(hours=3 * step) >= first_valid_offset
        ]
        for cell in cells:
            records += nwp_records(cell, init_time, members, valid_times=valid_times)
    frame = pl.DataFrame(records)
    frame = cast_to_nwp_dtypes(frame, *frame.columns)
    return pt.LazyFrame.from_existing(frame.lazy()).set_model(Nwp)


def _power_lazy(frame: pl.DataFrame) -> pt.LazyFrame[PowerTimeSeries]:
    return pt.LazyFrame.from_existing(frame.lazy()).set_model(PowerTimeSeries)


def _lag_frame(rows: list[dict], lag_columns: list[str]) -> pt.LazyFrame[AllFeatures]:
    """An engineered-features frame holding the key columns and the given lag columns."""
    frame = pl.DataFrame(
        rows,
        schema={
            "valid_time": _UTC,
            "time_series_id": pl.Int32,
            "power_fcst_init_time": _UTC,
            **dict.fromkeys(lag_columns, pl.Float32),
        },
    )
    return pt.LazyFrame.from_existing(frame.lazy()).set_model(AllFeatures)


def test_construction_orders_the_yaml_lags_by_hours_and_rejects_no_power_lag() -> None:
    raw = yaml.safe_load(_MODEL_YAML.read_text())
    forecaster = ManualHeuristicForecaster(
        BaseForecasterConfig(selected_features=set(raw["model_params"]["selected_features"]))
    )

    assert forecaster._lag_hours == _ALL_LAG_HOURS
    with pytest.raises(ValueError, match="at least one power lag"):
        ManualHeuristicForecaster(BaseForecasterConfig(selected_features={"local_time_of_day_sin"}))


def test_shedding_keeps_member_identity_through_the_real_pipeline() -> None:
    init_time = datetime(2025, 3, 3, tzinfo=UTC)
    power_fcst_init_time = init_time + timedelta(hours=9)
    power = _power_frame([1], start=init_time - timedelta(weeks=56), end=power_fcst_init_time)
    forecaster = ManualHeuristicForecaster(
        BaseForecasterConfig(selected_features=_ALL_LAG_FEATURES)
    )

    features = ManualHeuristicForecaster.feature_engineer.engineer(
        selected_features=_ALL_LAG_FEATURES,
        power_time_series=_power_lazy(power),
        time_series_metadata=_metadata({1: _CELL_A}),
        nwp=_nwp([(init_time, (0, 1), timedelta(0))], [_CELL_A]),
    )
    forecast = forecaster.predict(features).with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_seconds() / 3600
    )

    # Each value is the observed power at valid_time minus the lag whose rank is the member.
    for row in forecast.iter_rows(named=True):
        lag = _ALL_LAG_HOURS[row["ensemble_member"]]
        expected = power_at(row["valid_time"] - timedelta(hours=lag), offset=7)
        assert row["power_fcst"] == expected
    members_by_band = {
        "under 168 h": (forecast.filter(pl.col("lead_hours") < 168), set(range(13))),
        "168 h to 336 h": (
            forecast.filter(pl.col("lead_hours").is_between(168, 336, closed="left")),
            set(range(1, 13)),
        ),
        "336 h to 351 h": (
            forecast.filter(pl.col("lead_hours") >= 336),
            set(range(2, 13)),
        ),
    }
    for band, (rows, expected_members) in members_by_band.items():
        assert rows.height > 0, band
        assert set(rows["ensemble_member"].to_list()) == expected_members, band


def test_predict_drops_a_row_whose_lags_are_all_null() -> None:
    columns = ["power_lag_168h", "power_lag_336h"]
    init_time = datetime(2025, 1, 1, tzinfo=UTC)
    data = _lag_frame(
        [
            {
                "valid_time": init_time + timedelta(hours=1),
                "time_series_id": 1,
                "power_fcst_init_time": init_time,
                "power_lag_168h": None,
                "power_lag_336h": None,
            },
            {
                "valid_time": init_time + timedelta(hours=2),
                "time_series_id": 1,
                "power_fcst_init_time": init_time,
                "power_lag_168h": 5.0,
                "power_lag_336h": None,
            },
        ],
        columns,
    )

    forecast = ManualHeuristicForecaster(
        BaseForecasterConfig(selected_features=set(columns))
    ).predict(data)

    assert forecast["valid_time"].to_list() == [init_time + timedelta(hours=2)]
    assert forecast["ensemble_member"].to_list() == [0]


def test_predict_on_empty_input_returns_a_valid_empty_frame() -> None:
    columns = ["power_lag_168h", "power_lag_336h"]

    forecast = ManualHeuristicForecaster(
        BaseForecasterConfig(selected_features=set(columns))
    ).predict(_lag_frame([], columns))

    assert forecast.height == 0


def test_train_records_requested_series_with_observed_power() -> None:
    frame = pl.DataFrame(
        {
            "time_series_id": pl.Series([1, 1, 2, 3], dtype=pl.Int32),
            "power": pl.Series([1.0, None, None, 4.0], dtype=pl.Float32),
        }
    )
    forecaster = ManualHeuristicForecaster(
        BaseForecasterConfig(selected_features={"power_lag_168h"})
    )

    forecaster.train(pt.LazyFrame.from_existing(frame.lazy()).set_model(AllFeatures), [1, 2])

    assert forecaster.trained_time_series_ids == [1]


def test_save_then_load_round_trips_and_replaces_the_directory(tmp_path: Path) -> None:
    config = BaseForecasterConfig(
        selected_features={"power_lag_168h", "power_lag_336h"}, experiment_name="exp"
    )
    forecaster = ManualHeuristicForecaster(config)
    forecaster._trained_ids = [3, 5]
    (tmp_path / "stale.ubj").write_text("left over from a larger model")

    forecaster.save(tmp_path)
    loaded = ManualHeuristicForecaster.load(tmp_path)

    assert not (tmp_path / "stale.ubj").exists()
    assert loaded.trained_time_series_ids == [3, 5]
    assert loaded.model_params == config
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["model_class"] == "baseline_forecasters.manual_heuristic.ManualHeuristicForecaster"


def test_engineer_rows_equal_the_tabular_rows_deduplicated_across_members() -> None:
    features = {"power_lag_24h", "power_lag_168h", "power_lag_8232h"}
    day = datetime(2025, 3, 3, tzinfo=UTC)
    runs: list[tuple[datetime, tuple[int, ...], timedelta]] = [
        (day, (0, 1), timedelta(0)),
        # The second run's earliest valid times are cut off, as the window loader clips a run.
        (day + timedelta(days=1), (0, 1), timedelta(hours=24)),
        # The third run has no control member.
        (day + timedelta(days=2), (1,), timedelta(0)),
    ]
    nwp = _nwp(runs, [_CELL_A, _CELL_B])
    power = _power_frame(
        [1, 2, 3], start=day - timedelta(weeks=52), end=day + timedelta(days=2, hours=9)
    )
    metadata = _metadata({1: _CELL_A, 2: _CELL_A, 3: _CELL_B})
    power_time_series = _power_lazy(power)
    key_columns = ["time_series_id", "power_fcst_init_time", "valid_time"]
    compared_columns = [*key_columns, "nwp_init_time", "power", *sorted(features)]

    engineered = (
        PowerLagsPerNwpRunFeatureEngineer()
        .engineer(
            selected_features=features,
            power_time_series=power_time_series,
            time_series_metadata=metadata,
            nwp=nwp,
        )
        .select(compared_columns)
        .collect()
        .sort(key_columns)
    )
    tabular = (
        TabularFeatureEngineer()
        .engineer(
            selected_features=features,
            power_time_series=power_time_series,
            time_series_metadata=metadata,
            nwp=nwp,
        )
        .select(compared_columns)
        .collect()
        .unique()
        .sort(key_columns)
    )

    assert engineered.select(key_columns).n_unique() == engineered.height
    assert engineered.height > 0
    # Each non-null lag type appears, the annual lag among them.
    for lag in sorted(features):
        assert engineered[lag].null_count() < engineered.height, lag
    assert_frame_equal(engineered, tabular)


def test_engineer_raises_in_single_run_mode() -> None:
    init_time = datetime(2025, 3, 3, tzinfo=UTC)
    with pytest.raises(NotImplementedError, match="bulk mode only"):
        PowerLagsPerNwpRunFeatureEngineer().engineer(
            selected_features={"power_lag_168h"},
            power_time_series=_power_lazy(_power_frame([1], init_time, init_time)),
            time_series_metadata=_metadata({1: _CELL_A}),
            nwp=_nwp([(init_time, (0,), timedelta(0))], [_CELL_A]),
            power_fcst_init_time=init_time,
        )
