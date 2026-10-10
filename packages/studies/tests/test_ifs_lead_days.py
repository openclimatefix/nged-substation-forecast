from datetime import UTC, datetime

import polars as pl
import pytest
from studies.ifs_lead_days import (
    join_forecast_at_lead_day,
    raise_unless_same_folds,
    with_ifs_eras,
)


def _rows(times: list[datetime]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "site": ["A"] * len(times),
            "time": times,
            "month": [time.strftime("%Y-%m") for time in times],
        }
    ).with_columns(pl.col("time").dt.cast_time_unit("us"))


def _forecasts(inits: list[datetime], hours: list[int]) -> pl.DataFrame:
    rows = [
        {
            "site": "A",
            "init_time": init,
            "valid_time": init,
            "lead_hours": lead,
            # Encode the run's day and the lead in the value, so a wrong join shows in the number.
            "cloud_cover": float(init.day * 1000 + lead),
        }
        for init in inits
        for lead in hours
    ]
    return pl.DataFrame(rows, schema_overrides={"lead_hours": pl.Int32}).with_columns(
        pl.col("init_time").dt.cast_time_unit("us"), pl.col("valid_time").dt.cast_time_unit("us")
    )


def test_a_lead_day_one_row_takes_the_run_of_the_day_before_at_lead_24_plus_hour():
    rows = _rows([datetime(2025, 3, 10, 13, tzinfo=UTC)])
    forecasts = _forecasts([datetime(2025, 3, 9), datetime(2025, 3, 8)], list(range(80)))

    joined = join_forecast_at_lead_day(rows=rows, forecasts=forecasts, lead_day=1)

    assert joined["cloud_cover"].to_list() == [9 * 1000 + 24 + 13]
    assert joined["lead_hours"].to_list() == [37]


def test_the_run_day_crosses_a_month_end_correctly():
    rows = _rows([datetime(2025, 4, 1, 6, tzinfo=UTC)])
    forecasts = _forecasts([datetime(2025, 3, 30)], list(range(80)))

    joined = join_forecast_at_lead_day(rows=rows, forecasts=forecasts, lead_day=2)

    assert joined["cloud_cover"].to_list() == [30 * 1000 + 48 + 6]


def test_a_row_whose_run_is_missing_is_dropped():
    rows = _rows([datetime(2025, 3, 10, 13, tzinfo=UTC), datetime(2025, 3, 11, 13, tzinfo=UTC)])
    forecasts = _forecasts([datetime(2025, 3, 9)], list(range(80)))

    joined = join_forecast_at_lead_day(rows=rows, forecasts=forecasts, lead_day=1)

    assert joined["time"].dt.day().to_list() == [10]


def test_the_join_refuses_a_run_that_does_not_start_at_midnight():
    rows = _rows([datetime(2025, 3, 10, 13, tzinfo=UTC)])
    forecasts = _forecasts([datetime(2025, 3, 9, 12)], [37])

    with pytest.raises(ValueError, match="00 UTC"):
        join_forecast_at_lead_day(rows=rows, forecasts=forecasts, lead_day=1)


def test_a_lead_day_below_one_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        join_forecast_at_lead_day(
            rows=_rows([datetime(2025, 3, 10, 13, tzinfo=UTC)]),
            forecasts=_forecasts([datetime(2025, 3, 10)], [13]),
            lead_day=0,
        )


def test_eras_drop_the_straddling_months_and_everything_after_the_second_change():
    months = [
        "2024-04",
        "2024-05",
        "2024-06",
        "2024-07",
        "2024-08",
        "2024-09",
        "2024-10",
        "2024-11",
        "2024-12",
        "2025-01",
        "2025-02",
        "2025-03",
        "2025-04",
        "2025-05",
        "2026-04",
        "2026-05",
        "2026-06",
    ]
    rows = pl.DataFrame(
        {"site": ["A"] * len(months), "month": months, "time": [datetime(2025, 1, 1)] * len(months)}
    )

    labelled = with_ifs_eras(rows=rows, fold_offsets={0: 0, 1: 0})

    assert "2024-11" not in labelled["month"].to_list()
    assert "2026-05" not in labelled["month"].to_list()
    assert "2026-06" not in labelled["month"].to_list()
    # The first era holds 2024-04 to 2024-10, and the second starts at 2024-12.
    first = labelled.filter(pl.col("month") == "2024-10")["era_code"].to_list()
    second = labelled.filter(pl.col("month") == "2024-12")["era_code"].to_list()
    assert first == [0]
    assert second == [1]


def test_folds_that_differ_across_lead_days_are_refused():
    time = datetime(2025, 3, 10, 13, tzinfo=UTC)
    one = pl.DataFrame({"site": ["A"], "time": [time], "fold": [1]})
    other = pl.DataFrame({"site": ["A"], "time": [time], "fold": [2]})

    with pytest.raises(ValueError, match="different folds"):
        raise_unless_same_folds(frames=[one, other])


def test_identical_folds_across_lead_days_pass():
    time = datetime(2025, 3, 10, 13, tzinfo=UTC)
    frame = pl.DataFrame({"site": ["A"], "time": [time], "fold": [1]})

    raise_unless_same_folds(frames=[frame, frame])
