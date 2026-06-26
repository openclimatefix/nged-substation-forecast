"""Unit tests for the pure CV helpers (no I/O)."""

from datetime import date, datetime, timezone

import polars as pl
from contracts.hydra_schemas import CvFoldConfig
from ml_core._cv_helpers import (
    _parse_cv_partition_key,
    _subtract_months,
    eligible_time_series_ids,
    flatten_config,
    training_window,
)
from pydantic import BaseModel

_UTC = pl.Datetime("us", "UTC")


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0):
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# training_window
# ---------------------------------------------------------------------------


def test_training_window_inclusive_end_of_day() -> None:
    fold = CvFoldConfig(
        fold_id="2024",
        train_start=date(2020, 1, 1),
        train_end=date(2023, 12, 31),
        val_start=date(2024, 1, 1),
        val_end=date(2024, 12, 31),
    )
    start, end = training_window(fold)
    assert start == _utc(2020, 1, 1, 0, 0, 0)
    assert end == _utc(2023, 12, 31, 23, 59, 59)


def test_training_window_honours_train_end_when_gap_before_val_start() -> None:
    """A gap/embargo between train_end and val_start is respected: training stops at train_end."""
    fold = CvFoldConfig(
        fold_id="2024",
        train_start=date(2020, 1, 1),
        train_end=date(2023, 11, 30),  # one-month embargo before val_start
        val_start=date(2024, 1, 1),
        val_end=date(2024, 12, 31),
    )
    _, end = training_window(fold)
    assert end == _utc(2023, 11, 30, 23, 59, 59)
    # Crucially, training does NOT run up to val_start.
    assert end < _utc(2024, 1, 1, 0, 0, 0)


# ---------------------------------------------------------------------------
# _subtract_months
# ---------------------------------------------------------------------------


def test_subtract_months_simple() -> None:
    assert _subtract_months(date(2024, 7, 15), 6) == date(2024, 1, 15)


def test_subtract_months_crosses_year_boundary() -> None:
    assert _subtract_months(date(2024, 1, 1), 6) == date(2023, 7, 1)


def test_subtract_months_clamps_day_of_month() -> None:
    # 31 March minus 1 month -> February, clamped to the last valid day.
    assert _subtract_months(date(2023, 3, 31), 1) == date(2023, 2, 28)


# ---------------------------------------------------------------------------
# eligible_time_series_ids
# ---------------------------------------------------------------------------


def _fold_2024() -> CvFoldConfig:
    return CvFoldConfig(
        fold_id="2024",
        train_start=date(2020, 1, 1),
        train_end=date(2023, 12, 31),
        val_start=date(2024, 1, 1),
        val_end=date(2024, 12, 31),
    )


def _coverage() -> pl.DataFrame:
    # With min_training_months=6 and val_start=2024-01-01, the earliest required first_time is
    # 2023-07-01 00:00 and data must reach val_end (2024-12-31 23:59:59).
    return pl.DataFrame(
        {
            "time_series_id": pl.Series([1, 2, 3, 4], dtype=pl.Int32),
            "first_time": pl.Series(
                [
                    _utc(2020, 1, 1),  # ts1: plenty of history
                    _utc(2024, 10, 1),  # ts2: starts after val_start -> too little history
                    _utc(2020, 1, 1),  # ts3: plenty of history...
                    _utc(2023, 7, 1),  # ts4: exactly on the threshold (inclusive)
                ],
                dtype=_UTC,
            ),
            "last_time": pl.Series(
                [
                    _utc(2025, 1, 1),  # ts1: covers val_end
                    _utc(2025, 1, 1),  # ts2: covers val_end (but fails the history test)
                    _utc(2024, 6, 1),  # ts3: ...but stops before val_end -> excluded
                    _utc(2024, 12, 31, 23, 59, 59),  # ts4: exactly on val_end (inclusive)
                ],
                dtype=_UTC,
            ),
        }
    )


def test_eligible_time_series_ids_excludes_ineligible() -> None:
    eligible = eligible_time_series_ids(_coverage(), _fold_2024(), min_training_months=6)
    assert eligible == [1, 4]


def test_eligible_time_series_ids_is_data_only() -> None:
    """Eligibility depends on the data alone — repeated calls are identical and order-stable."""
    fold = _fold_2024()
    first = eligible_time_series_ids(_coverage(), fold, min_training_months=6)
    second = eligible_time_series_ids(_coverage(), fold, min_training_months=6)
    assert first == second == sorted(first)


def test_eligible_time_series_ids_respects_min_training_months() -> None:
    # Loosening the requirement to 1 month lets ts2 (first_time 2024-10-01) still fail, but a
    # series starting just before val_start would now pass. Verify the threshold moves.
    coverage = pl.DataFrame(
        {
            "time_series_id": pl.Series([5], dtype=pl.Int32),
            "first_time": pl.Series([_utc(2023, 12, 1)], dtype=_UTC),
            "last_time": pl.Series([_utc(2024, 12, 31, 23, 59, 59)], dtype=_UTC),
        }
    )
    fold = _fold_2024()
    assert eligible_time_series_ids(coverage, fold, min_training_months=6) == []
    assert eligible_time_series_ids(coverage, fold, min_training_months=1) == [5]


# ---------------------------------------------------------------------------
# _parse_cv_partition_key — round trips, including names containing "__"
# ---------------------------------------------------------------------------


def test_parse_cv_partition_key_simple() -> None:
    assert _parse_cv_partition_key("baseline__2022") == ("baseline", "2022")


def test_parse_cv_partition_key_experiment_name_with_double_underscore() -> None:
    assert _parse_cv_partition_key("my__weird__exp__2022") == ("my__weird__exp", "2022")


# ---------------------------------------------------------------------------
# flatten_config
# ---------------------------------------------------------------------------


def test_flatten_config_nested_dict() -> None:
    config = {"name": "x", "params": {"a": 1, "b": {"c": 2}}, "items": [1, 2, 3]}
    assert flatten_config(config) == {
        "name": "x",
        "params.a": "1",
        "params.b.c": "2",
        "items": "[1, 2, 3]",
    }


def test_flatten_config_pydantic_model() -> None:
    class Inner(BaseModel):
        learning_rate: float = 0.05

    class Outer(BaseModel):
        experiment_name: str = "baseline"
        inner: Inner = Inner()

    assert flatten_config(Outer()) == {
        "experiment_name": "baseline",
        "inner.learning_rate": "0.05",
    }
