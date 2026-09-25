from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from studies.gfs_native import (
    DomainType,
    gfs_leads,
    served_init_time,
    served_lead_hours,
    step_means,
    three_hour_means,
    window_hours,
)


def _truth(*, seed: int = 0) -> np.ndarray:
    """Return unequal hourly means for hours 1 to 384, one row per series."""
    return np.random.default_rng(seed).uniform(10.0, 900.0, size=(3, 384))


def _windows_and_steps(*, hourly: np.ndarray, leads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return what GFS would serve (since-reset window means) and the step means it encodes."""
    window = window_hours(leads=leads)
    previous = np.concatenate([[0], leads[:-1]])
    served = np.stack(
        [hourly[:, lead - w : lead].mean(axis=1) for lead, w in zip(leads, window, strict=True)],
        axis=1,
    )
    steps = np.stack(
        [hourly[:, p:lead].mean(axis=1) for lead, p in zip(leads, previous, strict=True)], axis=1
    )
    return served, steps


def test_the_leads_are_hourly_to_120_and_three_hourly_beyond():
    leads = gfs_leads()

    assert len(leads) == 120 + (384 - 120) // 3
    assert 120 in leads
    assert 121 not in leads
    assert leads[120] == 123


def test_a_lead_that_is_not_on_the_grid_raises():
    with pytest.raises(ValueError, match="not a GFS lead"):
        gfs_leads(last_lead=121)


def test_the_window_is_one_hour_after_a_reset_and_six_hours_at_it():
    leads = np.array([1, 2, 5, 6, 7, 12, 13, 120, 123, 126])

    assert window_hours(leads=leads).tolist() == [1, 2, 5, 6, 1, 6, 1, 6, 3, 6]


def test_step_means_recover_the_mean_of_every_hour_and_every_three_hours():
    leads = gfs_leads()
    served, steps = _windows_and_steps(hourly=_truth(), leads=leads)

    np.testing.assert_allclose(step_means(values=served, leads=leads), steps, rtol=1e-9)


def test_step_means_work_on_a_run_that_stops_inside_the_hourly_leads():
    leads = gfs_leads(last_lead=30)
    served, steps = _windows_and_steps(hourly=_truth()[:, :30], leads=leads)

    np.testing.assert_allclose(step_means(values=served, leads=leads), steps, rtol=1e-9)


def test_a_missing_window_mean_spoils_only_its_own_step_and_the_next_in_the_window():
    leads = gfs_leads(last_lead=24)
    served, _ = _windows_and_steps(hourly=_truth()[:, :24], leads=leads)
    served[:, list(leads).index(8)] = np.nan

    spoiled = np.isnan(step_means(values=served, leads=leads)).all(axis=0)

    assert leads[spoiled].tolist() == [8, 9]


def test_step_means_clip_a_negative_step_to_zero():
    leads = gfs_leads(last_lead=6)
    served = np.array([[10.0, 1.0, 0.0, 0.0, 0.0, 0.0]])

    means = step_means(values=served, leads=leads)

    assert means[0, 1] == 0.0
    assert (means >= 0.0).all()


def test_step_means_reject_leads_that_do_not_start_at_lead_one():
    with pytest.raises(ValueError, match="from lead 1"):
        step_means(values=np.ones((1, 3)), leads=np.array([2, 3, 4]))


def test_three_hour_means_average_three_hourly_steps_then_pass_coarse_steps_through():
    leads = gfs_leads()
    _, steps = _windows_and_steps(hourly=_truth(), leads=leads)

    three_leads, three = three_hour_means(hourly=steps, leads=leads)

    hours = _truth()
    assert three_leads[:3].tolist() == [3, 6, 9]
    assert three_leads[-1] == 384
    np.testing.assert_allclose(three[:, 0], hours[:, 0:3].mean(axis=1))
    np.testing.assert_allclose(three[:, 39], hours[:, 117:120].mean(axis=1))
    coarse = list(three_leads).index(123)
    np.testing.assert_allclose(three[:, coarse], hours[:, 120:123].mean(axis=1))
    np.testing.assert_allclose(three[:, -1], hours[:, 381:384].mean(axis=1))


def _times(*, start: datetime) -> pl.DataFrame:
    return pl.DataFrame({"time": [start + timedelta(hours=k) for k in range(24)]}).with_columns(
        pl.col("time").dt.replace_time_zone("UTC")
    )


def _served(*, day: int, domain: DomainType, frame: pl.DataFrame) -> pl.DataFrame:
    return frame.select(
        "time",
        init=served_init_time(time=pl.col("time"), day=day, domain=domain),
        lead=served_lead_hours(time=pl.col("time"), day=day, domain=domain),
    )


def test_a_solar_hour_reads_the_run_of_the_day_its_window_falls_in():
    frame = pl.DataFrame(
        {"time": [datetime(2025, 3, 10, 0), datetime(2025, 3, 10, 13)]}
    ).with_columns(pl.col("time").dt.replace_time_zone("UTC"))

    served = _served(day=2, domain="solar", frame=frame)

    # The hour ending 00:00 on 10 March is the last hour of 9 March, so day 2 reads 7 March's run.
    assert served["init"].to_list() == [
        datetime(2025, 3, 7, tzinfo=UTC),
        datetime(2025, 3, 8, tzinfo=UTC),
    ]
    assert served["lead"].to_list() == [72, 61]


def test_a_wind_hour_reads_the_run_of_its_own_day():
    frame = pl.DataFrame(
        {"time": [datetime(2025, 3, 10, 0), datetime(2025, 3, 10, 13)]}
    ).with_columns(pl.col("time").dt.replace_time_zone("UTC"))

    served = _served(day=2, domain="wind", frame=frame)

    assert served["init"].to_list() == [datetime(2025, 3, 8, tzinfo=UTC)] * 2
    assert served["lead"].to_list() == [48, 61]


@pytest.mark.parametrize(
    ("domain", "first_lead"),
    [("solar", 24 * 3 + 1), ("wind", 24 * 3)],
)
def test_a_day_covers_24_consecutive_leads_of_one_run(domain: DomainType, first_lead: int):
    frame = _times(start=datetime(2025, 3, 10, 1 if domain == "solar" else 0))

    served = _served(day=3, domain=domain, frame=frame)

    assert served["init"].n_unique() == 1
    assert served["lead"].to_list() == list(range(first_lead, first_lead + 24))


def test_day_zero_reads_the_freshest_run_at_a_lead_of_one_to_six_hours_for_solar():
    frame = _times(start=datetime(2025, 3, 10, 0))

    served = _served(day=0, domain="solar", frame=frame)

    assert served["lead"].to_list() == [6, *([1, 2, 3, 4, 5, 6] * 3), 1, 2, 3, 4, 5]
    assert served["init"][0] == datetime(2025, 3, 9, 18, tzinfo=UTC)
    assert served["init"][12] == datetime(2025, 3, 10, 6, tzinfo=UTC)


def test_day_zero_reads_the_freshest_run_at_a_lead_of_zero_to_five_hours_for_wind():
    frame = _times(start=datetime(2025, 3, 10, 0))

    served = _served(day=0, domain="wind", frame=frame)

    assert served["lead"].to_list() == [0, 1, 2, 3, 4, 5] * 4
    assert served["init"][6] == datetime(2025, 3, 10, 6, tzinfo=UTC)
