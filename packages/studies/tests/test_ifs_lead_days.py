from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from studies.cross_validation import calendar_month_coverage, raise_on_uncovered_months
from studies.ifs_lead_days import (
    PARTNER_COLUMNS,
    blend_month_folds,
    find_fold_offsets,
    join_aifs_partner,
    join_forecast_at_lead_day,
    raise_unless_same_folds,
    with_blend_folds,
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
            "shortwave_radiation": float(init.day * 1000 + lead),
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

    # An instantaneous variable is the mean of its values at leads 36 and 37; radiation is as is.
    assert joined["cloud_cover"].to_list() == [9 * 1000 + 36.5]
    assert joined["shortwave_radiation"].to_list() == [9 * 1000 + 37]
    assert joined["lead_hours"].to_list() == [37]


def test_a_long_lead_day_does_not_overflow_the_hour_of_day():
    # Int8 holds at most 127, so 24 * 5 + 13 computed in Int8 would wrap to a negative lead.
    rows = _rows([datetime(2025, 3, 10, 13, tzinfo=UTC)])
    forecasts = _forecasts([datetime(2025, 3, 5)], list(range(241)))

    joined = join_forecast_at_lead_day(rows=rows, forecasts=forecasts, lead_day=5)

    assert joined["lead_hours"].to_list() == [133]
    assert joined["shortwave_radiation"].to_list() == [5 * 1000 + 133]


def test_hour_zero_averages_with_the_previous_days_last_lead_hour_of_the_same_run():
    rows = _rows([datetime(2025, 3, 10, 0, tzinfo=UTC)])
    forecasts = _forecasts([datetime(2025, 3, 9), datetime(2025, 3, 8)], list(range(80)))

    joined = join_forecast_at_lead_day(rows=rows, forecasts=forecasts, lead_day=1)

    assert joined["lead_hours"].to_list() == [24]
    assert joined["cloud_cover"].to_list() == [9 * 1000 + 23.5]


def test_a_missing_neighbouring_lead_makes_an_instantaneous_value_missing_and_keeps_the_row():
    rows = _rows([datetime(2025, 3, 10, 13, tzinfo=UTC)])
    forecasts = _forecasts([datetime(2025, 3, 9)], list(range(80))).filter(
        pl.col("lead_hours") != 36
    )

    joined = join_forecast_at_lead_day(rows=rows, forecasts=forecasts, lead_day=1)

    assert joined["cloud_cover"].to_list() == [None]
    assert joined["shortwave_radiation"].to_list() == [9 * 1000 + 37]


def test_the_run_day_crosses_a_month_end_correctly():
    rows = _rows([datetime(2025, 4, 1, 6, tzinfo=UTC)])
    forecasts = _forecasts([datetime(2025, 3, 30)], list(range(80)))

    joined = join_forecast_at_lead_day(rows=rows, forecasts=forecasts, lead_day=2)

    assert joined["shortwave_radiation"].to_list() == [30 * 1000 + 48 + 6]


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


def _partner(
    *, rows: list[tuple[datetime, datetime | None, float | None]], lead_day: int
) -> pl.DataFrame:
    prefix = f"aifs_single_day{lead_day}"
    return pl.DataFrame(
        {
            "site": ["A"] * len(rows),
            "time": [time for time, _, _ in rows],
            f"{prefix}_ghi": [ghi for _, _, ghi in rows],
            f"{prefix}_temp": [10.0 if ghi is not None else None for _, _, ghi in rows],
            f"{prefix}_init_time": [init for _, init, _ in rows],
        },
        schema_overrides={f"{prefix}_init_time": pl.Datetime("ns", "UTC")},
    ).with_columns(pl.col("time").dt.cast_time_unit("us"))


def _lead_rows(times: list[datetime]) -> pl.DataFrame:
    return _rows(times).with_columns(pl.col("time").dt.replace_time_zone("UTC"))


def test_the_partner_join_keeps_the_run_issued_lead_days_before_and_renames_its_columns():
    time = datetime(2025, 4, 10, 12, tzinfo=UTC)
    rows = _lead_rows([datetime(2025, 4, 10, 12)])
    partner = _partner(rows=[(time, datetime(2025, 4, 8, tzinfo=UTC), 300.0)], lead_day=2)

    joined = join_aifs_partner(rows=rows, partner=partner, lead_day=2)

    assert joined.columns[-2:] == list(PARTNER_COLUMNS)
    assert joined[PARTNER_COLUMNS[0]].to_list() == [300.0]
    assert joined[PARTNER_COLUMNS[1]].to_list() == [10.0]


def test_the_partner_join_refuses_a_run_from_the_wrong_day():
    time = datetime(2025, 4, 10, 12, tzinfo=UTC)
    rows = _lead_rows([datetime(2025, 4, 10, 12)])
    partner = _partner(rows=[(time, datetime(2025, 4, 9, tzinfo=UTC), 300.0)], lead_day=2)

    with pytest.raises(ValueError, match="not the 00 UTC run 2 days before"):
        join_aifs_partner(rows=rows, partner=partner, lead_day=2)


def test_the_partner_join_refuses_a_run_that_does_not_start_at_midnight():
    time = datetime(2025, 4, 10, 12, tzinfo=UTC)
    rows = _lead_rows([datetime(2025, 4, 10, 12)])
    partner = _partner(rows=[(time, datetime(2025, 4, 9, 12, tzinfo=UTC), 300.0)], lead_day=1)

    with pytest.raises(ValueError, match="00 UTC run 1 days"):
        join_aifs_partner(rows=rows, partner=partner, lead_day=1)


def test_the_partner_join_drops_runs_before_the_operational_era_and_missing_values():
    early = datetime(2025, 2, 27, 12, tzinfo=UTC)
    late = datetime(2025, 3, 5, 12, tzinfo=UTC)
    missing = datetime(2025, 3, 6, 12, tzinfo=UTC)
    rows = _lead_rows(
        [early.replace(tzinfo=None), late.replace(tzinfo=None), missing.replace(tzinfo=None)]
    )
    partner = _partner(
        rows=[
            (early, datetime(2025, 2, 26, tzinfo=UTC), 100.0),
            (late, datetime(2025, 3, 4, tzinfo=UTC), 200.0),
            (missing, None, None),
        ],
        lead_day=1,
    )

    joined = join_aifs_partner(rows=rows, partner=partner, lead_day=1)

    assert joined["time"].dt.day().to_list() == [5]


def test_the_partner_join_keeps_the_first_operational_run_and_drops_a_missing_temperature():
    first_day = datetime(2025, 3, 2, 12, tzinfo=UTC)
    no_temperature = datetime(2025, 3, 6, 12, tzinfo=UTC)
    midnight = datetime(2025, 3, 8, 0, tzinfo=UTC)
    rows = _lead_rows([t.replace(tzinfo=None) for t in (first_day, no_temperature, midnight)])
    partner = _partner(
        rows=[
            (first_day, datetime(2025, 3, 1, tzinfo=UTC), 100.0),
            (no_temperature, datetime(2025, 3, 5, tzinfo=UTC), 200.0),
            (midnight, datetime(2025, 3, 7, tzinfo=UTC), 300.0),
        ],
        lead_day=1,
    ).with_columns(
        pl.when(pl.col("time").dt.day() == 6)
        .then(None)
        .otherwise(pl.col("aifs_single_day1_temp"))
        .alias("aifs_single_day1_temp")
    )

    joined = join_aifs_partner(rows=rows, partner=partner, lead_day=1)

    # The run on exactly 2025-03-01 is kept, the null temperature is dropped, and the 00:00 label
    # reads the run of the day before its own day, so its run on 2025-03-07 is the label-day run.
    assert joined["time"].dt.day().to_list() == [2, 8]


def test_the_partner_join_refuses_a_site_and_time_held_twice():
    time = datetime(2025, 4, 10, 12, tzinfo=UTC)
    init = datetime(2025, 4, 9, tzinfo=UTC)
    partner = _partner(rows=[(time, init, 300.0), (time, init, 301.0)], lead_day=1)

    with pytest.raises(ValueError, match="twice"):
        join_aifs_partner(rows=_lead_rows([datetime(2025, 4, 10, 12)]), partner=partner, lead_day=1)


def test_the_blend_folds_depend_on_the_set_of_months_alone_not_their_order_or_repeats():
    months = [f"2025-{month:02d}" for month in range(3, 13)] + ["2026-01", "2026-02", "2026-03"]

    folds = blend_month_folds(months=months)
    shuffled = blend_month_folds(months=[*reversed(months), *months])

    assert folds == shuffled
    assert [folds[month] for month in sorted(folds)] == sorted(folds.values())
    assert set(folds.values()) == {0, 1, 2, 3, 4}


def test_a_lead_day_missing_a_month_still_gets_the_months_fold_from_the_shared_map():
    months = [f"2025-{month:02d}" for month in range(3, 13)] + ["2026-01", "2026-02", "2026-03"]
    folds = blend_month_folds(months=months)
    rows = pl.DataFrame({"site": ["A", "A"], "month": ["2025-04", "2026-03"]})

    folded = with_blend_folds(rows=rows, month_folds=folds)

    assert folded["fold"].to_list() == [folds["2025-04"], folds["2026-03"]]
    assert folded["fold"].dtype == pl.Int32


def test_blend_folds_refuse_too_few_months_and_a_month_with_no_fold():
    with pytest.raises(ValueError, match="cannot fill"):
        blend_month_folds(months=["2025-03", "2025-04"])
    with pytest.raises(ValueError, match="no fold for"):
        with_blend_folds(rows=pl.DataFrame({"month": ["2025-03"]}), month_folds={"2025-04": 0})


def test_the_first_partial_month_stays_in_the_first_era():
    rows = pl.DataFrame(
        {
            "site": ["A"] * 3,
            "month": ["2024-03", "2024-10", "2024-12"],
            "time": [datetime(2025, 1, 1)] * 3,
        }
    )

    labelled = with_ifs_eras(rows=rows, fold_offsets={0: 0, 1: 0})

    assert labelled.filter(pl.col("month") == "2024-03")["era_code"].to_list() == [0]


def _monthly_rows(*, first: str, last: str, sites: list[str]) -> pl.DataFrame:
    months = []
    year, month = int(first[:4]), int(first[5:])
    while f"{year}-{month:02d}" <= last:
        months.append(f"{year}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return pl.DataFrame(
        {
            "site": [site for site in sites for _ in months],
            "month": [m for _ in sites for m in months],
            "time": [
                datetime(int(m[:4]), int(m[5:]), 15, 12) + timedelta(hours=hour)
                for _ in sites
                for m in months
                for hour in (0,)
            ],
        }
    )


def test_find_fold_offsets_ignores_the_dropped_months_and_returns_a_covering_design():
    rows = _monthly_rows(first="2024-03", last="2026-06", sites=["A", "B"])
    without = rows.filter(~pl.col("month").is_in(["2024-11", "2026-05", "2026-06"]))

    offsets = find_fold_offsets(rows=rows)

    assert offsets == find_fold_offsets(rows=without)
    assert set(offsets) == {0, 1}
    # The design leaves no calendar month untrained once the eras are cut with it.
    folded = with_ifs_eras(rows=rows, fold_offsets=offsets)
    raise_on_uncovered_months(coverage=calendar_month_coverage(frame=folded))


def test_find_fold_offsets_raises_when_no_rotation_covers_every_calendar_month(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("studies.ifs_lead_days.search_fold_offsets", lambda **_: [])
    rows = _monthly_rows(first="2024-03", last="2026-04", sites=["A"])

    with pytest.raises(ValueError, match="no fold rotation"):
        find_fold_offsets(rows=rows)
