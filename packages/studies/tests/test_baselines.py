from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from studies.baselines import (
    CLEAR_SKY_SAMPLE_MINUTES,
    MIN_OBSERVED_CLEAR_SKY_SHARE,
    clear_sky_index,
    climatology,
    diurnal_persistence,
    haurwitz_w_m2,
    hourly_clear_sky,
    issue_time,
    persistence,
    shrunk_persistence,
)
from studies.solar import zenith

DAY = datetime(2025, 6, 10, tzinfo=UTC)


def _hourly(*, values: dict[datetime, float], site: str = "A") -> pl.DataFrame:
    return pl.DataFrame(
        {"site": [site] * len(values), "time": list(values), "power_mw": list(values.values())}
    )


def test_the_issue_time_is_nine_on_the_runs_own_day():
    frame = pl.DataFrame({"day_start": [DAY]})

    issued = frame.select(issue_time(day_start=pl.col("day_start"), day=2)).item()

    assert issued == DAY - timedelta(days=2) + timedelta(hours=9)


def test_persistence_reads_the_last_hour_ended_by_the_issue_time():
    issue = DAY + timedelta(hours=9)
    hourly = _hourly(
        values={issue - timedelta(hours=1): 1.0, issue: 2.0, issue + timedelta(hours=1): 3.0}
    )
    keys = pl.DataFrame({"site": ["A"], "issue_time": [issue]})

    assert persistence(keys=keys, hourly=hourly, observed_lag=timedelta(0)).to_list() == [2.0]


def test_a_centred_hour_is_observed_only_once_its_second_half_has_passed():
    issue = DAY + timedelta(hours=9)
    hourly = _hourly(values={issue - timedelta(hours=1): 1.0, issue: 2.0})
    keys = pl.DataFrame({"site": ["A"], "issue_time": [issue]})

    lagged = persistence(keys=keys, hourly=hourly, observed_lag=timedelta(minutes=30))

    assert lagged.to_list() == [1.0]


def test_persistence_falls_back_to_an_earlier_hour_within_a_day_only():
    issue = DAY + timedelta(hours=9)
    hourly = _hourly(values={issue - timedelta(hours=5): 4.0}).vstack(
        _hourly(values={issue - timedelta(hours=30): 7.0}, site="B")
    )
    keys = pl.DataFrame({"site": ["A", "B"], "issue_time": [issue, issue]})

    assert persistence(keys=keys, hourly=hourly, observed_lag=timedelta(0)).to_list() == [4.0, None]


def test_persistence_keeps_the_keys_order_and_each_sites_own_readings():
    issue = DAY + timedelta(hours=9)
    hourly = _hourly(values={issue: 1.0}).vstack(_hourly(values={issue: 5.0}, site="B"))
    keys = pl.DataFrame({"site": ["B", "A"], "issue_time": [issue, issue]})

    assert persistence(keys=keys, hourly=hourly, observed_lag=timedelta(0)).to_list() == [5.0, 1.0]


def test_diurnal_persistence_reads_the_same_hour_on_the_last_whole_day():
    target = DAY + timedelta(hours=14)
    hourly = _hourly(
        values={
            target - timedelta(days=2): 1.0,
            target - timedelta(days=3): 2.0,
            target - timedelta(days=2, hours=1): 9.0,
        }
    )
    keys = pl.DataFrame({"site": ["A"], "time": [target]})

    assert diurnal_persistence(keys=keys, hourly=hourly, day=1).to_list() == [1.0]


def test_diurnal_persistence_falls_back_to_an_earlier_day_at_the_same_hour():
    target = DAY + timedelta(hours=14)
    hourly = _hourly(
        values={target - timedelta(days=4): 3.0, target - timedelta(days=2, hours=1): 9.0}
    )
    keys = pl.DataFrame({"site": ["A"], "time": [target]})

    assert diurnal_persistence(keys=keys, hourly=hourly, day=1).to_list() == [3.0]


def test_haurwitz_matches_its_published_formula_and_is_zero_below_the_horizon():
    irradiance = haurwitz_w_m2(apparent_zenith_deg=np.array([0.0, 60.0, 95.0]))

    np.testing.assert_allclose(
        irradiance, [1098.0 * np.exp(-0.059), 1098.0 * 0.5 * np.exp(-0.118), 0.0]
    )


def test_the_hourly_clear_sky_is_a_mean_over_the_hour_and_zero_at_night():
    sites = pl.DataFrame({"site": ["A"], "latitude": [53.0], "longitude": [0.0]})

    table = hourly_clear_sky(
        sites=sites, first=DAY + timedelta(hours=1), last=DAY + timedelta(hours=24)
    )
    by_hour = dict(
        zip(table["time"].dt.hour().to_list(), table["clear_sky_w_m2"].to_list(), strict=True)
    )

    assert by_hour[1] == 0.0
    assert by_hour[13] > by_hour[9] > by_hour[6] > 0.0


def test_the_clear_sky_index_is_observed_energy_over_observed_clear_sky_energy():
    issue = DAY + timedelta(hours=9)
    hours = [issue - timedelta(hours=offset) for offset in range(24)]
    power = [2.0 if offset < 12 else None for offset in range(24)]
    clear = [100.0 if offset < 12 else 50.0 for offset in range(24)]
    hourly = pl.DataFrame(
        {"site": ["A"] * 24, "time": hours, "power_mw": power, "clear_sky_w_m2": clear}
    )
    keys = pl.DataFrame({"site": ["A"], "issue_time": [issue]})

    assert clear_sky_index(keys=keys, hourly=hourly).to_list() == [pytest.approx(0.02)]


def test_the_clear_sky_index_is_missing_when_too_little_sunlight_was_observed():
    issue = DAY + timedelta(hours=9)
    hours = [issue - timedelta(hours=offset) for offset in range(24)]
    power = [2.0 if offset < 3 else None for offset in range(24)]
    hourly = pl.DataFrame(
        {"site": ["A"] * 24, "time": hours, "power_mw": power, "clear_sky_w_m2": [100.0] * 24}
    )
    keys = pl.DataFrame({"site": ["A"], "issue_time": [issue]})

    assert clear_sky_index(keys=keys, hourly=hourly).to_list() == [None]


def _folded(
    *, power: list[float], folds: list[int], constrained: list[bool] | None = None
) -> pl.DataFrame:
    count = len(power)
    return pl.DataFrame(
        {
            "site": ["A"] * count,
            "time": [DAY + timedelta(days=index) for index in range(count)],
            "fold": folds,
            "constrained": constrained or [False] * count,
            "power_mw": power,
        }
    )


def test_climatology_is_the_median_of_the_other_folds_at_the_same_month_and_hour():
    frame = _folded(power=[1.0, 2.0, 9.0, 100.0], folds=[0, 0, 0, 1])

    medians = climatology(frame=frame).to_list()

    assert medians[3] == 2.0
    assert medians[:3] == [100.0, 100.0, 100.0]


def test_climatology_leaves_out_curtailed_hours():
    frame = _folded(power=[1.0, 50.0, 3.0], folds=[0, 0, 1], constrained=[False, True, False])

    assert climatology(frame=frame).to_list()[2] == 1.0


def test_climatology_falls_back_to_the_hour_over_every_month():
    frame = _folded(power=[4.0, 8.0], folds=[0, 1]).with_columns(
        time=pl.Series([DAY, DAY + timedelta(days=40)])
    )

    assert climatology(frame=frame).to_list() == [8.0, 4.0]


def test_a_day_zero_forecast_is_cut_off_at_the_runs_own_midnight():
    frame = pl.DataFrame({"day_start": [DAY]})

    issued = frame.select(issue_time(day_start=pl.col("day_start"), day=0)).item()

    assert issued == DAY


def test_day_zero_persistence_reads_nothing_from_the_target_day():
    hourly = _hourly(
        values={DAY - timedelta(hours=1): 1.0, DAY: 2.0, DAY + timedelta(hours=9): 3.0}
    )
    keys = pl.DataFrame({"site": ["A"], "time": [DAY + timedelta(hours=12)]}).with_columns(
        issue_time=issue_time(day_start=pl.col("time").dt.truncate("1d"), day=0)
    )

    lagged = persistence(keys=keys, hourly=hourly, observed_lag=timedelta(minutes=30))

    assert lagged.to_list() == [1.0]


def test_diurnal_persistence_never_reads_the_runs_own_day():
    target = DAY + timedelta(hours=14)
    hourly = _hourly(values={target - timedelta(days=1): 50.0, target - timedelta(days=2): 1.0})
    keys = pl.DataFrame({"site": ["A"], "time": [target]})

    assert diurnal_persistence(keys=keys, hourly=hourly, day=1).to_list() == [1.0]


def test_diurnal_persistence_looks_back_no_more_than_a_week():
    target = DAY + timedelta(hours=14)
    hourly = _hourly(values={target - timedelta(days=10): 3.0})
    keys = pl.DataFrame({"site": ["A"], "time": [target]})

    assert diurnal_persistence(keys=keys, hourly=hourly, day=1).to_list() == [None]


def test_the_hourly_clear_sky_is_the_mean_of_six_samples_inside_the_hour():
    sites = pl.DataFrame({"site": ["A"], "latitude": [53.0], "longitude": [0.0]})
    hour_end = DAY + timedelta(hours=8)
    samples = pl.Series(
        [hour_end - timedelta(minutes=minutes) for minutes in CLEAR_SKY_SAMPLE_MINUTES]
    )
    expected = haurwitz_w_m2(
        apparent_zenith_deg=zenith(stamps=samples, latitude=53.0, longitude=0.0)
    ).mean()

    table = hourly_clear_sky(sites=sites, first=hour_end, last=hour_end)

    assert table["clear_sky_w_m2"].to_list() == [pytest.approx(expected)]


def _window(
    *, issue: datetime, power: dict[int, float | None], clear: float = 100.0
) -> pl.DataFrame:
    """Return hours from issue-30 h to issue+2 h, power by offset from the issue, else null."""
    offsets = list(range(-30, 3))
    return pl.DataFrame(
        {
            "site": ["A"] * len(offsets),
            "time": [issue + timedelta(hours=offset) for offset in offsets],
            "power_mw": [power.get(offset) for offset in offsets],
            "clear_sky_w_m2": [clear] * len(offsets),
        }
    )


def test_the_clear_sky_window_is_the_24_hours_ending_at_the_issue():
    issue = DAY + timedelta(hours=9)
    inside = {offset: float(offset + 30) for offset in range(-23, 1)}
    hourly = _window(issue=issue, power={**inside, -24: 1000.0, 1: 1000.0, 2: 1000.0})
    keys = pl.DataFrame({"site": ["A"], "issue_time": [issue]})

    index = clear_sky_index(keys=keys, hourly=hourly).to_list()

    assert index == [pytest.approx(sum(inside.values()) / (100.0 * 24))]


def test_each_issue_time_gets_its_own_window():
    first, second = DAY + timedelta(hours=9), DAY + timedelta(days=1, hours=9)
    hourly = _window(issue=second, power=dict.fromkeys(range(-30, 3), 1.0))
    hourly = hourly.with_columns(
        power_mw=pl.when(pl.col("time") > first).then(4.0).otherwise(pl.col("power_mw"))
    )
    keys = pl.DataFrame({"site": ["A", "A"], "issue_time": [second, first]})

    index = clear_sky_index(keys=keys, hourly=hourly).to_list()

    assert index == [pytest.approx(0.04), pytest.approx(0.01)]


def test_the_clear_sky_index_needs_half_the_windows_clear_sky_energy_observed():
    issue = DAY + timedelta(hours=9)
    needed = int(24 * MIN_OBSERVED_CLEAR_SKY_SHARE)
    enough = _window(issue=issue, power={-offset: 1.0 for offset in range(needed)})
    short = _window(issue=issue, power={-offset: 1.0 for offset in range(needed - 1)})
    keys = pl.DataFrame({"site": ["A"], "issue_time": [issue]})

    assert clear_sky_index(keys=keys, hourly=enough).to_list() == [pytest.approx(0.01)]
    assert clear_sky_index(keys=keys, hourly=short).to_list() == [None]


def test_climatology_keys_on_the_calendar_month():
    june, july = datetime(2025, 6, 3, 12, tzinfo=UTC), datetime(2025, 7, 3, 12, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "site": ["A"] * 4,
            "time": [june, june + timedelta(days=1), july, june + timedelta(days=365)],
            "fold": [0, 0, 0, 1],
            "constrained": [False] * 4,
            "power_mw": [2.0, 2.0, 9.0, 0.0],
        }
    )

    assert climatology(frame=frame).to_list()[3] == 2.0


def test_the_calendar_fallback_is_the_two_neighbouring_months_median():
    # No training row shares the target's own calendar month, so the fallback cannot use the
    # month-and-hour median. The whole-year hour fallback and the neighbouring-month fallback
    # disagree here (100 against 150), so only the neighbouring-month rule gives 150.
    hour = 12
    frame = pl.DataFrame(
        {
            "site": ["A"] * 4,
            "time": [
                datetime(2025, 6, 15, hour, tzinfo=UTC),  # target: June, scored fold
                datetime(2025, 5, 15, hour, tzinfo=UTC),  # neighbour: May
                datetime(2025, 7, 15, hour, tzinfo=UTC),  # neighbour: July
                datetime(2025, 12, 15, hour, tzinfo=UTC),  # not a neighbour: December
            ],
            "fold": [0, 1, 1, 1],
            "constrained": [False] * 4,
            "power_mw": [0.0, 100.0, 200.0, 5.0],
        }
    )

    assert climatology(frame=frame).to_list()[0] == 150.0


def test_the_calendar_fallback_wraps_across_the_year_boundary():
    # December and February are January's neighbours only if the neighbour month wraps: December
    # is month 0 before wrapping (0 -> 12), and this target has no row at its own month and hour, so
    # the January estimate depends entirely on the wrap. A March row is not a neighbour of January
    # either way and must not be picked up. Without the wrap, only February's row (shifted from 2 to
    # 1) matches, giving a median of 200 rather than the neighbouring-months median of 150.
    hour = 12
    frame = pl.DataFrame(
        {
            "site": ["A"] * 4,
            "time": [
                datetime(2025, 1, 15, hour, tzinfo=UTC),  # target: January, scored fold
                datetime(2025, 12, 15, hour, tzinfo=UTC),  # neighbour: December
                datetime(2025, 2, 15, hour, tzinfo=UTC),  # neighbour: February
                datetime(2025, 3, 15, hour, tzinfo=UTC),  # not a neighbour: March
            ],
            "fold": [0, 1, 1, 1],
            "constrained": [False] * 4,
            "power_mw": [0.0, 100.0, 200.0, 5.0],
        }
    )

    assert climatology(frame=frame).to_list()[0] == 150.0


def test_the_hourly_fallback_is_a_median():
    times = [datetime(2025, month, 3, 12, tzinfo=UTC) for month in (1, 2, 3)]
    frame = pl.DataFrame(
        {
            "site": ["A"] * 4,
            "time": [*times, datetime(2025, 8, 3, 12, tzinfo=UTC)],
            "fold": [0, 0, 0, 1],
            "constrained": [False] * 4,
            "power_mw": [1.0, 2.0, 30.0, 0.0],
        }
    )

    assert climatology(frame=frame).to_list()[3] == 2.0


def _shrinkage_frame(
    *, site: str, power: tuple[float, ...], persisted: tuple[float, ...], last_persisted: float
) -> pl.DataFrame:
    """Return one generator's rows: one month per fold, one row a day, noon each day.

    Every month holds `power`; persistence reads `persisted`, except in the last month, where it
    reads `last_persisted` on every day.
    """
    return pl.DataFrame(
        [
            {
                "site": site,
                "time": datetime(2025, fold + 1, day + 1, 12, tzinfo=UTC),
                "fold": fold,
                "constrained": False,
                "power_mw": power[day],
                "persisted": persisted[day] if fold != 4 else last_persisted,
            }
            for fold in range(5)
            for day in range(len(power))
        ]
    )


def test_the_shrinkage_weight_comes_from_the_training_folds_alone():
    # Persistence carries nothing on every fold but the last, where it is exact. The weight for the
    # last fold is fitted on the other four, so it must be 0 there, however well 1 would do.
    frame = _shrinkage_frame(
        site="A", power=(10.0, 10.0, 10.0), persisted=(0.0, 0.0, 0.0), last_persisted=10.0
    )

    blended = shrunk_persistence(frame=frame, persisted="persisted")

    last = blended.filter(frame["fold"] == 4)
    assert last["weight"].to_list() == [0.0] * 3
    assert last["shrunk"].to_list() == [10.0] * 3


def test_the_shrinkage_weight_is_fitted_per_generator():
    # At A persistence is always wrong and climatology right; at B persistence is always exact and
    # climatology, the median of 1, 2 and 3, wrong on two days in three.
    frame = pl.concat(
        [
            _shrinkage_frame(
                site="A", power=(10.0, 10.0, 10.0), persisted=(0.0, 0.0, 0.0), last_persisted=0.0
            ),
            _shrinkage_frame(
                site="B", power=(1.0, 2.0, 3.0), persisted=(1.0, 2.0, 3.0), last_persisted=2.0
            ),
        ]
    )

    weights = shrunk_persistence(frame=frame, persisted="persisted")["weight"]

    assert set(weights.filter(frame["site"] == "A").to_list()) == {0.0}
    assert set(weights.filter(frame["site"] == "B").to_list()) == {1.0}


def test_the_shrinkage_ignores_curtailed_hours_and_weighs_the_absolute_error():
    # Each month: an hour persistence gets right (10, 10), one it overshoots by 5 (0, 5), one both
    # get right (0, 0), and a curtailed hour it overshoots by 1000. Climatology, the month's median,
    # is 0. The absolute error falls all the way to a weight of 1 on persistence; a squared error
    # would stop at 0.8, and the curtailed hour, if kept, would pull the weight to 0.
    pairs = ((10.0, 10.0, False), (0.0, 5.0, False), (0.0, 0.0, False), (0.0, 1000.0, True))
    frame = pl.DataFrame(
        [
            {
                "site": "A",
                "time": datetime(2025, fold + 1, day + 1, 12, tzinfo=UTC),
                "fold": fold,
                "constrained": curtailed,
                "power_mw": power,
                "persisted": persisted,
            }
            for fold in range(5)
            for day, (power, persisted, curtailed) in enumerate(pairs)
        ]
    )

    weights = shrunk_persistence(frame=frame, persisted="persisted")["weight"]

    assert set(weights.to_list()) == {1.0}


def test_the_shrinkage_weight_is_found_to_the_hundredth():
    # One hour a month decides the weight: power 3.7, persistence 10, climatology 0, so the error
    # is zero at a weight of 0.37 exactly.
    frame = _shrinkage_frame(
        site="A", power=(3.7, 0.0, 0.0), persisted=(10.0, 0.0, 0.0), last_persisted=0.0
    )

    weights = shrunk_persistence(frame=frame, persisted="persisted")["weight"]

    assert set(weights.to_list()) == {0.37}


def _two_a_month(*, power: list[list[float]], persisted: list[list[float]]) -> pl.DataFrame:
    """Return one generator's rows: one month per fold, two noon rows a month."""
    return pl.DataFrame(
        [
            {
                "site": "A",
                "time": datetime(2025, fold + 1, day + 1, 12, tzinfo=UTC),
                "fold": fold,
                "constrained": False,
                "power_mw": power[fold][day],
                "persisted": persisted[fold][day],
            }
            for fold in range(5)
            for day in range(2)
        ]
    )


def test_the_weight_is_fitted_against_a_climatology_that_never_saw_the_scored_fold():
    # A fit against each training row's own out-of-fold climatology, which is built from folds
    # that include the one being scored, chooses 0.33 and 0.6 here where the right weights are 0
    # and 0.2.
    frame = _two_a_month(
        power=[[3.0, 2.0], [2.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 3.0]],
        persisted=[[2.0, 3.0], [2.0, 2.0], [3.0, 2.0], [2.0, 2.0], [2.0, 3.0]],
    )

    weights = shrunk_persistence(frame=frame, persisted="persisted")["weight"].to_list()

    assert weights == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0])


def test_the_scored_fold_is_forecast_with_a_climatology_of_the_other_folds():
    frame = _two_a_month(power=[[10.0, 10.0]] * 4 + [[30.0, 30.0]], persisted=[[0.0, 0.0]] * 5)

    blended = shrunk_persistence(frame=frame, persisted="persisted").filter(frame["fold"] == 4)

    assert blended["shrunk"].to_list() == [10.0, 10.0]
