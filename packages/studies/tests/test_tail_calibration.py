from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from studies.tail_calibration import (
    TABLE_COLUMNS,
    align_arms,
    level_column,
    pinball,
    quantile_levels,
    read_arms,
    tail_calibration_markdown,
    tail_calibration_table,
)

from studies import tail_calibration

START = datetime(2026, 1, 1, tzinfo=UTC)


def _frame(*, actual: list[float | None], q05: list[float], q95: list[float]) -> pl.DataFrame:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            "power_fcst_init_time": [start + timedelta(days=45 * i) for i in range(len(actual))],
            "actual": actual,
            "q05": q05,
            "q95": q95,
        }
    )


def test_level_column_names_whole_percents() -> None:
    assert level_column(level=0.05) == "q05"
    assert level_column(level=0.99) == "q99"
    assert level_column(level=0.01) == "q01"


@pytest.mark.parametrize("level", [0.0, 1.0, 0.055, 1.5])
def test_level_column_rejects_levels_that_are_not_a_whole_percent(level: float) -> None:
    with pytest.raises(ValueError, match="whole percent"):
        level_column(level=level)


def test_pinball_charges_the_shortfall_at_level_and_the_excess_at_one_minus_level() -> None:
    loss = pinball(actual=np.array([10.0, 10.0]), quantile=np.array([8.0, 12.0]), level=0.9)
    np.testing.assert_allclose(loss, [0.9 * 2.0, 0.1 * 2.0])


def test_table_reports_coverage_gap_and_pinball_per_delivered_level() -> None:
    # Actual is below q05 once in four rows, and below q95 in all four.
    frame = _frame(actual=[0.0, 5.0, 6.0, 7.0], q05=[1.0, 1.0, 1.0, 1.0], q95=[9.0, 9.0, 9.0, 9.0])
    table = tail_calibration_table(frame=frame, levels=(0.05, 0.95))
    assert table.columns == list(TABLE_COLUMNS)
    low, high = table.iter_rows(named=True)
    assert low["coverage"] == pytest.approx(0.25)
    assert low["gap"] == pytest.approx(0.20)
    assert low["pinball"] == pytest.approx(np.mean([0.95 * 1.0, 0.05 * 4, 0.05 * 5, 0.05 * 6]))
    assert high["coverage"] == pytest.approx(1.0)
    assert high["gap"] == pytest.approx(0.05)
    assert high["n_rows"] == 4


def test_table_marks_an_undelivered_level_and_the_markdown_says_so() -> None:
    frame = _frame(actual=[0.0, 5.0], q05=[1.0, 1.0], q95=[9.0, 9.0])
    table = tail_calibration_table(frame=frame)
    delivered = table.filter(pl.col("n_rows") > 0)["level"].to_list()
    assert delivered == [0.05, 0.95]
    assert table.filter(pl.col("level") == 0.01)["coverage"].item() is None
    markdown = tail_calibration_markdown(tables={"arm-a": table}, unit="points")
    assert "| arm-a | 0.01 | not delivered |" in markdown
    assert "| arm-a | 0.05 | 0.500" in markdown


def test_table_skips_rows_with_a_null_actual() -> None:
    frame = _frame(actual=[0.0, None, 5.0], q05=[1.0, 1.0, 1.0], q95=[9.0, 9.0, 9.0])
    table = tail_calibration_table(frame=frame, levels=(0.05,))
    assert table["n_rows"].item() == 2


def _arm_rows(*, days: int) -> pl.DataFrame:
    times = [START + timedelta(days=day) for day in range(days)]
    return pl.DataFrame(
        {
            "time_series_id": 1,
            "power_fcst_init_time": times,
            "valid_time": times,
            "seed": 0,
            "actual": [float(day) for day in range(days)],
            "q05": [0.0] * days,
            "q95": [1.0] * days,
        }
    ).cast({"time_series_id": pl.Int32, "seed": pl.Int8})


def test_quantile_levels_sorts_and_requires_two_columns() -> None:
    frame = pl.DataFrame({"q95": [1.0], "q05": [0.0], "other": [1]})
    assert quantile_levels(frame=frame) == [(0.05, "q05"), (0.95, "q95")]
    with pytest.raises(ValueError, match="at least two"):
        quantile_levels(frame=pl.DataFrame({"q05": [0.0]}))


def test_align_arms_keeps_shared_rows_and_drops_a_nan_actual() -> None:
    frame = _arm_rows(days=10)
    first = frame.with_columns(
        actual=pl.when(pl.col("valid_time") == START).then(float("nan")).otherwise(pl.col("actual"))
    )
    second = frame.slice(5)
    aligned = align_arms(arms={"first": first, "second": second})
    assert aligned["first"].height == aligned["second"].height == frame.height - 5
    assert aligned["first"]["valid_time"].to_list() == aligned["second"]["valid_time"].to_list()
    assert aligned["first"]["actual"].is_finite().all()


def test_a_nan_actual_in_a_later_arm_is_dropped_by_align_arms() -> None:
    frame = _arm_rows(days=10)
    other = frame.with_columns(
        actual=pl.when(pl.col("valid_time") == START).then(float("nan")).otherwise(pl.col("actual"))
    )
    aligned = align_arms(arms={"first": frame, "other": other})
    assert aligned["first"].height == aligned["other"].height == frame.height - 1


def test_read_arms_reads_each_named_set_from_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = {"a": _arm_rows(days=3), "b": _arm_rows(days=5)}
    for name, frame in frames.items():
        frame.write_parquet(tmp_path / f"{name}.parquet")

    def read_stored(*, experiment_id: int, name: str, row_set_id: str) -> pl.DataFrame:
        assert (experiment_id, row_set_id) == (48, "row_set")
        return pl.read_parquet(tmp_path / f"{name}.parquet")

    monkeypatch.setattr(tail_calibration, "read_predictions", read_stored)
    arms = read_arms(experiment_id=48, names=["a", "b"], row_set_id="row_set")
    assert list(arms) == ["a", "b"]
    assert all(arms[name].equals(frame) for name, frame in frames.items())
