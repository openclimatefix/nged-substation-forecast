"""Tests for ``write_effective_capacity`` — the ``effective_capacity`` storage format
end-to-end.

Writes real (tiny) ``EffectiveCapacity`` frames into a temp Delta table and asserts the
whole-table overwrite semantics (no partitioning, unlike the other three new tables).
"""

from datetime import UTC, datetime
from pathlib import Path

import patito as pt
import polars as pl
from contracts.power_schemas import EffectiveCapacity
from delta_store.effective_capacity import write_effective_capacity

_T0 = datetime(2025, 6, 1, tzinfo=UTC)


def _make_capacity(time_series_ids: list[int]) -> pt.DataFrame[EffectiveCapacity]:
    return (
        EffectiveCapacity.DataFrame(
            {
                "time_series_id": time_series_ids,
                "time": [_T0] * len(time_series_ids),
                "effective_capacity_mw": [10.0] * len(time_series_ids),
            }
        )
        .cast()
        .validate()
    )


def test_overwrite_replaces_whole_table(tmp_path: Path) -> None:
    table = tmp_path / "effective_capacity"
    write_effective_capacity(_make_capacity([1, 2, 3]), table)
    write_effective_capacity(_make_capacity([4, 5]), table)

    stored = pl.read_delta(str(table))
    assert sorted(stored["time_series_id"]) == [4, 5]
