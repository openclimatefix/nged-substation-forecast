"""Tests for the training-data cleaning rules, on small series built to trip each rule."""

from datetime import UTC, datetime, timedelta

import polars as pl
from cleaning_rules import (
    commissioning_ramp,
    first_months_after_first_nonzero,
    isolated_zeros,
    rows_to_drop,
    spikes,
)

START = datetime(2025, 1, 1, tzinfo=UTC)


def _series(*, series_id: int, values: list[float], start: datetime = START) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time_series_id": pl.Series([series_id] * len(values), dtype=pl.Int32),
            "time": pl.Series(
                [start + timedelta(minutes=30 * i) for i in range(len(values))],
                dtype=pl.Datetime("us", "UTC"),
            ),
            "power": pl.Series(values, dtype=pl.Float32),
        }
    )


def test_first_months_drop_the_leading_zeros_and_the_months_after_the_first_nonzero():
    """Leading zeros go with the first month after the first reading above zero."""
    two_days_zero = [0.0] * 96
    thirty_five_days = [5.0] * (48 * 35)
    power = _series(series_id=1, values=[*two_days_zero, *thirty_five_days])
    dropped = first_months_after_first_nonzero(power=power, months=1)
    first_nonzero = START + timedelta(days=2)
    cutoff = datetime(2025, 2, 3, tzinfo=UTC)
    assert dropped["time"].min() == START
    assert dropped["time"].max() == cutoff - timedelta(minutes=30)
    assert dropped.height == (cutoff - START) // timedelta(minutes=30)
    assert first_nonzero < cutoff


def test_first_months_leave_a_series_that_has_not_started_untouched_by_other_series():
    """A rule judges each series on its own first non-zero reading."""
    early = _series(series_id=1, values=[5.0] * 48 * 5)
    late = _series(series_id=2, values=[0.0] * 48 * 3 + [5.0] * 48 * 60)
    dropped = first_months_after_first_nonzero(power=pl.concat([early, late]), months=1)
    per_series = dropped.group_by("time_series_id").len().sort("time_series_id")
    assert per_series["time_series_id"].to_list() == [1, 2]
    # Series 1 is entirely within its first month; series 2 loses 3 zero days + 1 month.
    assert per_series["len"].to_list()[0] == 48 * 5
    assert per_series["len"].to_list()[1] > 48 * 30


def test_isolated_zeros_are_dropped_but_a_run_of_zeros_and_a_night_are_kept():
    """A lone zero between clear readings goes; two zeros in a row and an edge zero stay."""
    values = [10.0, 10.0, 0.0, 10.0, 10.0, 0.0, 0.0, 10.0, 10.0]
    dropped = isolated_zeros(power=_series(series_id=1, values=values))
    assert dropped["time"].to_list() == [START + timedelta(minutes=30 * 2)]


def test_isolated_zeros_ignore_a_zero_next_to_a_gap_in_the_record():
    """A zero whose neighbour is missing is not isolated: the reading before it is not adjacent."""
    first = _series(series_id=1, values=[10.0, 10.0, 0.0])
    second = _series(series_id=1, values=[10.0, 10.0], start=START + timedelta(hours=3))
    assert isolated_zeros(power=pl.concat([first, second])).height == 0


def test_spikes_catch_a_single_jump_up_or_down_and_not_a_step():
    """A one-reading spike goes; a step change that stays does not."""
    values = [10.0] * 10 + [90.0] + [10.0] * 10 + [60.0] * 10
    dropped = spikes(power=_series(series_id=1, values=values))
    assert dropped["time"].to_list() == [START + timedelta(minutes=30 * 10)]


def test_commissioning_ramp_drops_days_until_output_reaches_the_peers_level():
    """A solar farm at 20% of its peers for 70 days, then at 100%, loses the first 70 days."""
    days = 200
    peers = [_series(series_id=pid, values=[10.0] * 48 * days) for pid in (2, 3)]
    ramp = _series(series_id=1, values=[2.0] * 48 * 70 + [10.0] * 48 * (days - 70))
    types = pl.DataFrame(
        {"time_series_id": pl.Series([1, 2, 3], dtype=pl.Int32), "time_series_type": ["PV"] * 3}
    )
    dropped = commissioning_ramp(power=pl.concat([ramp, *peers]), types=types)
    assert set(dropped["time_series_id"].to_list()) == {1}
    last_dropped_day = dropped["time"].max()
    assert last_dropped_day is not None
    # The 14-day rolling median crosses 90% a week into the settled period, so the cut falls
    # between day 70 and day 77.
    assert START + timedelta(days=69) < last_dropped_day <= START + timedelta(days=78)


def test_commissioning_ramp_ignores_demand_series_and_short_histories():
    """Only PV and wind are judged, and a series under 60 days is left alone."""
    peers = [_series(series_id=pid, values=[10.0] * 48 * 30) for pid in (2, 3)]
    short = _series(series_id=1, values=[1.0] * 48 * 30)
    types = pl.DataFrame(
        {"time_series_id": pl.Series([1, 2, 3], dtype=pl.Int32), "time_series_type": ["PV"] * 3}
    )
    assert commissioning_ramp(power=pl.concat([short, *peers]), types=types).height == 0
    demand = pl.DataFrame(
        {
            "time_series_id": pl.Series([1, 2, 3], dtype=pl.Int32),
            "time_series_type": ["Disaggregated Demand"] * 3,
        }
    )
    long_ramp = _series(series_id=1, values=[1.0] * 48 * 100)
    long_peers = [_series(series_id=pid, values=[10.0] * 48 * 100) for pid in (2, 3)]
    assert commissioning_ramp(power=pl.concat([long_ramp, *long_peers]), types=demand).height == 0


def test_rows_to_drop_none_drops_nothing_and_names_dispatch_to_their_rule():
    """The `none` rule drops nothing; `first_2_months` calls the months rule with 2."""
    power = _series(series_id=1, values=[5.0] * 48 * 100)
    types = pl.DataFrame(
        {"time_series_id": pl.Series([1], dtype=pl.Int32), "time_series_type": ["PV"]}
    )
    assert rows_to_drop(rule="none", power=power, types=types).height == 0
    two = rows_to_drop(rule="first_2_months", power=power, types=types)
    assert two.height == (datetime(2025, 3, 1, tzinfo=UTC) - START) // timedelta(minutes=30)
