from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from studies.baselines import (
    clear_sky_index,
    climatology,
    diurnal_persistence,
    haurwitz_w_m2,
    hourly_clear_sky,
    issue_time,
    persistence,
    shrunk_persistence,
)

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


def test_shrinkage_weighs_persistence_fully_where_it_is_exact_on_the_training_folds():
    frame = _folded(power=[1.0, 5.0, 3.0, 7.0], folds=[0, 0, 1, 1]).with_columns(
        persisted=pl.col("power_mw"), climate=pl.lit(4.0)
    )

    blended = shrunk_persistence(frame=frame, persisted="persisted", climatological="climate")

    assert blended["weight"].to_list() == [1.0] * 4
    assert blended["shrunk"].to_list() == frame["power_mw"].to_list()


def test_shrinkage_weighs_climatology_fully_where_persistence_carries_nothing():
    frame = _folded(power=[4.0, 4.0, 4.0, 4.0], folds=[0, 0, 1, 1]).with_columns(
        persisted=pl.Series([0.0, 9.0, 1.0, 8.0]), climate=pl.lit(4.0)
    )

    blended = shrunk_persistence(frame=frame, persisted="persisted", climatological="climate")

    assert blended["weight"].to_list() == [0.0] * 4
