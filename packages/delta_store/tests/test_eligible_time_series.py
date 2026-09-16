"""Tests for ``write_eligible_time_series`` — the ``eligible_time_series`` storage format
end-to-end.

Writes real (tiny) ``EligibleTimeSeries`` frames into a temp Delta table and asserts the
per-``fold_id``-partition overwrite semantics, including the empty-frame case that is the reason
``fold_id`` is a required parameter rather than read off the frame.
"""

from pathlib import Path

import patito as pt
import polars as pl
from contracts.ml_schemas import EligibleTimeSeries
from delta_store.eligible_time_series import write_eligible_time_series


def _make_eligible(fold_id: str, time_series_ids: list[int]) -> pt.DataFrame[EligibleTimeSeries]:
    return (
        EligibleTimeSeries.DataFrame(
            {
                "fold_id": [fold_id] * len(time_series_ids),
                "time_series_id": time_series_ids,
            }
        )
        .cast()
        .validate()
    )


def test_overwrite_is_partition_scoped_by_fold_id(tmp_path: Path) -> None:
    table = tmp_path / "eligible_time_series"
    write_eligible_time_series(_make_eligible("fold_a", [1, 2]), table, fold_id="fold_a")
    write_eligible_time_series(_make_eligible("fold_b", [3]), table, fold_id="fold_b")
    write_eligible_time_series(_make_eligible("fold_a", [4, 5, 6]), table, fold_id="fold_a")

    partition_dirs = sorted(p.name for p in table.glob("fold_id=*"))
    assert partition_dirs == ["fold_id=fold_a", "fold_id=fold_b"]

    stored = pl.read_delta(str(table))
    assert sorted(stored.filter(pl.col("fold_id") == "fold_a")["time_series_id"]) == [4, 5, 6]
    assert stored.filter(pl.col("fold_id") == "fold_b")["time_series_id"].to_list() == [3]


def test_empty_frame_still_clears_partition(tmp_path: Path) -> None:
    table = tmp_path / "eligible_time_series"
    write_eligible_time_series(_make_eligible("fold_a", [1, 2]), table, fold_id="fold_a")

    write_eligible_time_series(_make_eligible("fold_a", []), table, fold_id="fold_a")

    stored = pl.read_delta(str(table))
    assert stored.filter(pl.col("fold_id") == "fold_a").is_empty()
