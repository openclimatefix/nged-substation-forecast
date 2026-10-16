"""Tests for the date sharding that the control-member backfill fetch relies on."""

from datetime import date, timedelta

import pytest
from studies.ens_backfill_fetch import shard_dates


def test_shards_partition_the_dates_and_keep_their_order() -> None:
    dates = [date(2023, 7, 1) + timedelta(days=offset) for offset in range(10)]
    shards = [shard_dates(dates=dates, shard=shard, shard_count=3) for shard in range(3)]
    assert sorted(day for shard in shards for day in shard) == dates
    assert shards[0] == [dates[0], dates[3], dates[6], dates[9]]
    assert all(shard == sorted(shard) for shard in shards)


def test_one_shard_returns_every_date() -> None:
    dates = [date(2023, 7, 1), date(2023, 7, 2)]
    assert shard_dates(dates=dates, shard=0, shard_count=1) == dates


def test_a_shard_outside_the_count_raises() -> None:
    with pytest.raises(ValueError, match="shard 2"):
        shard_dates(dates=[date(2023, 7, 1)], shard=2, shard_count=2)
