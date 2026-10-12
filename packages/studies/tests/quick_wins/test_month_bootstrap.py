"""Tests for the month bootstrap: known answers, pairing, and series weighting."""

import numpy as np
import polars as pl
import pytest
from month_bootstrap import (
    MonthlyTable,
    difference_interval,
    level_interval,
    monthly_table,
)


def _scores(*, values: dict[tuple[int, str], list[float]]) -> pl.DataFrame:
    rows = [
        {"time_series_id": series_id, "valid_time": f"{month}-15 12:00:00", "m": value}
        for (series_id, month), column in values.items()
        for value in column
    ]
    return pl.DataFrame(rows).with_columns(
        pl.col("valid_time").str.to_datetime(time_zone="UTC"),
        pl.col("time_series_id").cast(pl.Int32),
    )


def _twelve_months(*, level: float, series: tuple[int, ...] = (1, 2)) -> pl.DataFrame:
    months = [f"2025-{m:02d}" for m in range(1, 13)]
    return _scores(values={(s, month): [level, level] for s in series for month in months})


def test_monthly_table_totals_and_counts_by_series_and_month():
    """Two rows in January for series 1 total to their sum with a count of 2."""
    table = monthly_table(
        scores=_scores(
            values={
                (1, "2025-01"): [1.0, 3.0],
                (1, "2025-02"): [5.0],
                (2, "2025-01"): [7.0],
                (2, "2025-02"): [9.0],
            }
        ),
        metric="m",
    )
    assert table.series_ids == (1, 2)
    assert table.months == ("2025-01", "2025-02")
    assert table.totals.tolist() == [[4.0, 5.0], [7.0, 9.0]]
    assert table.counts.tolist() == [[2.0, 1.0], [1.0, 1.0]]


def test_a_constant_metric_has_a_point_interval_equal_to_that_constant():
    """Every row is 2.0, so every resample is 2.0."""
    interval = level_interval(table=monthly_table(scores=_twelve_months(level=2.0), metric="m"))
    assert interval.estimate == pytest.approx(2.0)
    assert interval.lower_95 == pytest.approx(2.0)
    assert interval.upper_95 == pytest.approx(2.0)


def test_the_metric_is_the_mean_of_series_means_not_the_mean_of_rows():
    """Series 1 has 1 row at 0 and series 2 has 9 rows at 10: the leaderboard mean is 5, not 9."""
    scores = _scores(values={(1, "2025-01"): [0.0], (2, "2025-01"): [10.0] * 9})
    table = monthly_table(scores=scores, metric="m")
    table = MonthlyTable(
        series_ids=table.series_ids,
        months=("2025-01", "2025-02"),
        totals=np.hstack([table.totals, table.totals]),
        counts=np.hstack([table.counts, table.counts]),
    )
    interval = level_interval(table=table)
    assert interval.estimate == pytest.approx(5.0)


def test_a_uniform_improvement_gives_a_negative_difference_that_every_month_shares():
    """An arm 0.5 lower in every month: difference -0.5, 12 of 12 months better, tight interval."""
    arm = monthly_table(scores=_twelve_months(level=1.5), metric="m")
    reference = monthly_table(scores=_twelve_months(level=2.0), metric="m")
    interval = difference_interval(arm=arm, reference=reference)
    assert interval.estimate == pytest.approx(-0.5)
    assert interval.months_better == 12
    assert interval.upper_95 < 0
    assert interval.t_upper_95 == pytest.approx(-0.5)


def test_pairing_removes_a_swing_both_arms_share():
    """Both arms swing by month together; the difference stays exactly 0.1 in every resample."""
    swing = {m: float(m) for m in range(1, 13)}

    def build(offset: float) -> pl.DataFrame:
        return _scores(
            values={
                (s, f"2025-{m:02d}"): [swing[m] + offset] * 2 for s in (1, 2) for m in range(1, 13)
            }
        )

    arm = monthly_table(scores=build(0.1), metric="m")
    reference = monthly_table(scores=build(0.0), metric="m")
    paired = difference_interval(arm=arm, reference=reference)
    own = level_interval(table=arm)
    assert paired.lower_95 == pytest.approx(0.1)
    assert paired.upper_95 == pytest.approx(0.1)
    assert own.upper_95 - own.lower_95 > 1.0


def test_arms_on_different_rows_are_refused():
    """A contrast between arms that were not scored on the same rows is an error."""
    one = monthly_table(scores=_twelve_months(level=1.0), metric="m")
    other = monthly_table(scores=_twelve_months(level=1.0, series=(1, 3)), metric="m")
    with pytest.raises(ValueError, match="exactly the same"):
        difference_interval(arm=one, reference=other)
