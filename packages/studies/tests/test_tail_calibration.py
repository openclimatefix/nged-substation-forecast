from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from studies.tail_calibration import (
    TABLE_COLUMNS,
    level_column,
    pinball,
    tail_calibration_markdown,
    tail_calibration_table,
)


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
