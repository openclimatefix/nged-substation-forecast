import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

_STUDY_DIR = Path(__file__).resolve().parents[3] / "studies" / "nwp_forecast_comparison"
sys.path.insert(0, str(_STUDY_DIR))
sys.path.insert(0, str(_STUDY_DIR.parent / "beam_diffuse_split"))

from verify_gfs_native import (  # noqa: E402
    expected_served,
    lookup_verdict,
    negative_share_verdict,
    recovered_step_means,
    served_table,
    served_verdict,
    window_length,
    window_verdict,
)


def _hourly_truth() -> dict[int, float]:
    """A daylight curve: the mean over the hour ending at each valid hour of day, in W/m2."""
    return {hour: max(0.0, 600.0 * math.sin(math.pi * (hour - 5) / 15)) for hour in range(24)}


def _window_means(*, truth: dict[int, float]) -> dict[int, float]:
    """The since-reset window means GFS would serve for known hourly means."""
    means = {}
    for hour in range(24):
        length = window_length(valid_hour=hour)
        means[hour] = sum(truth[(hour - back) % 24] for back in range(length)) / length
    return means


def _table(*, coarse_follows_three_hours: bool) -> pl.DataFrame:
    truth = _hourly_truth()
    window = _window_means(truth=truth)
    last_three = {h: sum(truth[(h - back) % 24] for back in range(3)) / 3 for h in range(24)}
    records = []
    for hour in range(24):
        coarse = (last_three if coarse_follows_three_hours else window)[hour]
        records.append(
            {
                "valid_hour": hour,
                "hourly_zone": window[hour],
                "coarse_zone": coarse if hour % 3 == 0 else None,
                "step_mean": truth[hour],
                "last_three_hours": last_three[hour],
            }
        )
    return pl.DataFrame(records)


def test_the_window_is_one_hour_after_each_reset_and_six_hours_at_it():
    assert [window_length(valid_hour=hour) for hour in (0, 1, 2, 5, 6, 7, 12, 13, 18, 19, 23)] == [
        6,
        1,
        2,
        5,
        6,
        1,
        6,
        1,
        6,
        1,
        5,
    ]


def test_hourly_means_are_recovered_from_the_window_means():
    truth = _hourly_truth()

    recovered = recovered_step_means(window_means=_window_means(truth=truth))

    assert all(math.isclose(recovered[h], truth[h], abs_tol=1e-9) for h in range(24))


def test_a_store_that_follows_the_reset_rule_passes_the_window_check():
    assert window_verdict(table=_table(coarse_follows_three_hours=False)) == []


def test_three_hourly_values_that_are_plain_three_hour_means_fail_the_window_check():
    failures = window_verdict(table=_table(coarse_follows_three_hours=True))

    assert any(line.startswith("valid hour 12 UTC") for line in failures)


def test_a_bright_night_fails_the_window_check():
    table = _table(coarse_follows_three_hours=False).with_columns(
        step_mean=pl.when(pl.col("valid_hour") == 0).then(50.0).otherwise(pl.col("step_mean"))
    )

    assert any("night hour 00" in line for line in window_verdict(table=table))


def test_lead_zero_radiation_and_frequent_negative_steps_fail_the_negative_share_check():
    table = pl.DataFrame(
        {
            "window_hours": [0, 3],
            "share_negative": [0.4, 0.05],
            "share_missing": [1.0, 0.0],
        }
    )

    failures = negative_share_verdict(table=table)

    assert len(failures) == 2


def test_served_runs_agree_between_the_two_expressions_and_read_the_stated_leads():
    table = served_table(first_day=datetime(2025, 3, 10, tzinfo=UTC))

    assert served_verdict(table=table) == []


def test_a_wrong_lead_is_flagged_by_the_served_runs_check():
    table = served_table(first_day=datetime(2025, 3, 10, tzinfo=UTC)).with_columns(
        lead=pl.when((pl.col("technology") == "wind") & (pl.col("day") == 3))
        .then(1)
        .otherwise(pl.col("lead")),
        agrees=pl.lit(True),
    )

    assert served_verdict(table=table)


def test_the_plain_python_rule_matches_the_worked_examples():
    solar = expected_served(time=datetime(2025, 3, 10, 13, tzinfo=UTC), day=2, solar=True)
    wind = expected_served(time=datetime(2025, 3, 10, 13, tzinfo=UTC), day=0, solar=False)

    assert solar == (datetime(2025, 3, 8, tzinfo=UTC), 61)
    assert wind == (datetime(2025, 3, 10, 12, tzinfo=UTC), 1)


def test_a_column_with_a_differing_or_no_compared_value_fails_the_lookup_check():
    table = pl.DataFrame(
        {
            "technology": ["solar", "solar", "wind"],
            "day": [1, 1, 2],
            "column": ["a", "b", "c"],
            "compared": [10, 10, 0],
            "differing": [0, 2, 0],
            "skipped": [0, 0, 10],
        }
    )

    assert len(lookup_verdict(table=table)) == 2
