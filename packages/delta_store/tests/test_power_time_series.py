"""Tests for ``write_power_time_series`` — the ``power_time_series`` storage format end-to-end.

Writes real (tiny) ``PowerTimeSeries`` frames into a temp Delta table and asserts the append /
Hive-partition-by-``time_series_id`` behaviour ported from the old ``.write_delta()`` call site.
"""

from datetime import UTC, datetime
from pathlib import Path

import patito as pt
import polars as pl
from contracts.power_schemas import PowerTimeSeries
from delta_store.power_time_series import write_power_time_series

_T0 = datetime(2025, 6, 1, tzinfo=UTC)


def _make_power_ts(time_series_id: int, n: int = 2) -> pt.DataFrame[PowerTimeSeries]:
    return (
        PowerTimeSeries.DataFrame(
            {
                "time_series_id": [time_series_id] * n,
                "time": [_T0.replace(hour=i) for i in range(n)],
                "power": [1.5 * (i + 1) for i in range(n)],
            }
        )
        .cast()
        .validate()
    )


def test_append_partitions_by_time_series_id(tmp_path: Path) -> None:
    table = tmp_path / "power_time_series"
    write_power_time_series(_make_power_ts(1), table)
    write_power_time_series(_make_power_ts(2), table)

    partition_dirs = sorted(p.name for p in table.glob("time_series_id=*"))
    assert partition_dirs == ["time_series_id=1", "time_series_id=2"]

    stored = pl.read_delta(str(table)).sort("time_series_id", "time")
    assert stored["time_series_id"].to_list() == [1, 1, 2, 2]
