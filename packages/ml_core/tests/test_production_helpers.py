"""Unit tests for ``ml_core._production_helpers``.

``select_nwp_init_time`` and ``build_live_power_frame`` take an injected clock and no wall-clock
reads, so they are exercised directly with fixed datetimes. ``load_forecaster_from_dir`` is
exercised against a real ``XGBoostForecaster`` save/load round-trip (no MLflow involved — it
only reads a plain disk directory).
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest
from contracts.common import UTC_DATETIME_DTYPE
from contracts.power_schemas import PowerTimeSeries
from ml_core._production_helpers import (
    build_live_power_frame,
    load_forecaster_from_dir,
    select_nwp_init_time,
)
from xgboost_forecaster.forecaster import XGBoostConfig, XGBoostForecaster

import patito as pt

_POWER_FCST_INIT_TIME = datetime(2026, 7, 4, 6, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# select_nwp_init_time
# ---------------------------------------------------------------------------


def test_live_picks_freshest_run_at_or_before_power_fcst_init_time() -> None:
    available = [
        _POWER_FCST_INIT_TIME - timedelta(hours=24),
        _POWER_FCST_INIT_TIME - timedelta(hours=6),
        _POWER_FCST_INIT_TIME,
    ]
    assert (
        select_nwp_init_time(
            available, power_fcst_init_time=_POWER_FCST_INIT_TIME, availability_mode="live"
        )
        == _POWER_FCST_INIT_TIME
    )


def test_replay_picks_freshest_run_at_or_before_delayed_cutoff() -> None:
    available = [
        _POWER_FCST_INIT_TIME - timedelta(hours=24),
        _POWER_FCST_INIT_TIME - timedelta(hours=6),
        _POWER_FCST_INIT_TIME,
    ]
    # Replay cutoff is power_fcst_init_time - 6h (the default delay): the run exactly there
    # qualifies, not the run at power_fcst_init_time itself.
    assert select_nwp_init_time(
        available, power_fcst_init_time=_POWER_FCST_INIT_TIME, availability_mode="replay"
    ) == _POWER_FCST_INIT_TIME - timedelta(hours=6)


def test_live_and_replay_diverge_when_a_fresher_run_exists_within_the_delay_window() -> None:
    """The whole point of the two modes: a run inside (power_fcst_init_time-6h,
    power_fcst_init_time] is live-only visible."""
    available = [
        _POWER_FCST_INIT_TIME - timedelta(hours=24),
        _POWER_FCST_INIT_TIME - timedelta(hours=1),
    ]

    live = select_nwp_init_time(
        available, power_fcst_init_time=_POWER_FCST_INIT_TIME, availability_mode="live"
    )
    replay = select_nwp_init_time(
        available, power_fcst_init_time=_POWER_FCST_INIT_TIME, availability_mode="replay"
    )

    assert live == _POWER_FCST_INIT_TIME - timedelta(hours=1)
    assert replay == _POWER_FCST_INIT_TIME - timedelta(hours=24)
    assert live != replay


def test_raises_when_no_run_qualifies() -> None:
    available = [_POWER_FCST_INIT_TIME + timedelta(hours=1)]
    with pytest.raises(ValueError, match="No NWP run available"):
        select_nwp_init_time(
            available, power_fcst_init_time=_POWER_FCST_INIT_TIME, availability_mode="live"
        )


# ---------------------------------------------------------------------------
# build_live_power_frame
# ---------------------------------------------------------------------------


def test_build_live_power_frame_grid_bounds_join_and_nulls() -> None:
    history = timedelta(hours=1)
    horizon = timedelta(hours=1)
    # Inside (power_fcst_init_time-history, power_fcst_init_time].
    observed_time = _POWER_FCST_INIT_TIME - timedelta(minutes=30)

    power = pt.LazyFrame.from_existing(
        pl.DataFrame(
            {
                "time_series_id": pl.Series([1], dtype=pl.Int32),
                "time": pl.Series([observed_time], dtype=UTC_DATETIME_DTYPE),
                "power": pl.Series([42.0], dtype=pl.Float32),
            }
        ).lazy()
    ).set_model(PowerTimeSeries)

    result = build_live_power_frame(
        power,
        [1, 2],
        power_fcst_init_time=_POWER_FCST_INIT_TIME,
        history=history,
        horizon=horizon,
    ).collect()

    # Grid is (power_fcst_init_time-history, power_fcst_init_time+horizon] on a half-hourly step:
    # 4 slots here.
    expected_times = [
        _POWER_FCST_INIT_TIME - timedelta(minutes=30),
        _POWER_FCST_INIT_TIME,
        _POWER_FCST_INIT_TIME + timedelta(minutes=30),
        _POWER_FCST_INIT_TIME + timedelta(hours=1),
    ]
    ts1_times = sorted(result.filter(pl.col("time_series_id") == 1)["time"].to_list())
    assert ts1_times == expected_times

    # Every requested id is present, even ts2, which has no observations at all.
    assert set(result["time_series_id"].unique().to_list()) == {1, 2}
    assert result.filter(pl.col("time_series_id") == 2)["power"].is_null().all()

    # The one genuine observation joins in...
    observed_row = result.filter(
        (pl.col("time_series_id") == 1) & (pl.col("time") == observed_time)
    )
    assert observed_row["power"].to_list() == [42.0]

    # ...and every other (future / unobserved) slot for ts1 is null.
    other_rows = result.filter((pl.col("time_series_id") == 1) & (pl.col("time") != observed_time))
    assert other_rows["power"].is_null().all()


# ---------------------------------------------------------------------------
# load_forecaster_from_dir
# ---------------------------------------------------------------------------


def test_load_forecaster_from_dir_round_trips_xgboost(tmp_path: Path) -> None:
    config = XGBoostConfig(selected_features={"temperature_2m"}, experiment_name="exp")
    XGBoostForecaster(config).save(tmp_path)

    loaded = load_forecaster_from_dir(tmp_path)

    assert isinstance(loaded, XGBoostForecaster)
    assert loaded.model_params == config


def test_load_forecaster_from_dir_raises_on_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Materialise"):
        load_forecaster_from_dir(tmp_path / "missing")


def test_load_forecaster_from_dir_raises_on_missing_model_class(tmp_path: Path) -> None:
    (tmp_path / "meta.json").write_text(
        json.dumps({"model_params": {}, "trained_time_series_ids": []})
    )
    with pytest.raises(ValueError, match="model_class"):
        load_forecaster_from_dir(tmp_path)
