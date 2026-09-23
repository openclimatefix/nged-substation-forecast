from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from studies.ensemble import check_one_run_per_hour

HOUR = datetime(2025, 6, 2, 3, tzinfo=UTC)
RUN = datetime(2025, 6, 2, tzinfo=UTC)


def _members(*, run: datetime, members: range) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "site": "A",
            "time": [HOUR] * len(members),
            "init_time": [run] * len(members),
            "member": list(members),
        }
    )


def test_one_run_with_every_member_passes():
    check_one_run_per_hour(hourly=_members(run=RUN, members=range(3)), members=3)


def test_two_runs_on_one_hour_are_refused():
    # The members of yesterday's run at the same hour, with different values, must not be averaged
    # into today's.
    both = pl.concat(
        [
            _members(run=RUN, members=range(3)),
            _members(run=RUN - timedelta(days=1), members=range(3)),
        ]
    )

    with pytest.raises(ValueError, match="more than one run"):
        check_one_run_per_hour(hourly=both, members=3)


def test_a_missing_member_is_refused():
    with pytest.raises(ValueError, match="exactly once"):
        check_one_run_per_hour(hourly=_members(run=RUN, members=range(2)), members=3)


def test_a_repeated_member_is_refused_even_with_the_right_row_count_elsewhere():
    # Members 0, 1, 2, 0: every member is present, but one twice, so only the row count betrays it.
    repeated = pl.concat([_members(run=RUN, members=range(3)), _members(run=RUN, members=range(1))])

    with pytest.raises(ValueError, match="exactly once"):
        check_one_run_per_hour(hourly=repeated, members=3)
