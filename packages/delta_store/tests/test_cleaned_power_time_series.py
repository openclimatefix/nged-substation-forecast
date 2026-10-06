"""Tests for ``write_cleaned_power_time_series``: overwrite, vacuum, and commit provenance."""

from datetime import UTC, datetime
from pathlib import Path

import patito as pt
import polars as pl
import pytest
from contracts.power_schemas import CleanedPowerTimeSeries
from delta_store.cleaned_power_time_series import VacuumError, write_cleaned_power_time_series
from deltalake import DeltaTable
from nged_data.cleaning import CleaningProvenance, read_cleaning_provenance

_T0 = datetime(2025, 6, 1, tzinfo=UTC)


def _provenance(raw_version: int) -> CleaningProvenance:
    return CleaningProvenance(
        raw_table_id="raw-id", raw_version=raw_version, code_hash="hash", git_sha="sha"
    )


def _cleaned(time_series_id: int) -> pt.DataFrame[CleanedPowerTimeSeries]:
    return (
        CleanedPowerTimeSeries.DataFrame(
            {
                "time_series_id": [time_series_id] * 2,
                "time": [_T0.replace(hour=i) for i in range(2)],
                "power": [1.5, 0.0],
                "drop_reason": [None, "substation_zero"],
            }
        )
        .cast()
        .validate()
    )


def _data_files(table: Path) -> list[Path]:
    return sorted(table.rglob("*.parquet"))


def test_second_write_overwrites_rather_than_appends(tmp_path: Path) -> None:
    table = tmp_path / "cleaned"
    write_cleaned_power_time_series(_cleaned(1), table, provenance=_provenance(1))
    write_cleaned_power_time_series(_cleaned(2), table, provenance=_provenance(2))

    stored = pl.scan_delta(str(table)).collect().sort("time")
    assert stored["time_series_id"].unique().to_list() == [2]
    assert stored["drop_reason"].to_list() == [None, "substation_zero"]


def test_retention_zero_deletes_the_superseded_files(tmp_path: Path) -> None:
    table = tmp_path / "cleaned"
    write_cleaned_power_time_series(
        _cleaned(1), table, provenance=_provenance(1), retention_hours=0
    )
    write_cleaned_power_time_series(
        _cleaned(2), table, provenance=_provenance(2), retention_hours=0
    )

    assert [p.parent.name for p in _data_files(table)] == ["time_series_id=2"]


def test_default_retention_keeps_the_superseded_files(tmp_path: Path) -> None:
    table = tmp_path / "cleaned"
    write_cleaned_power_time_series(_cleaned(1), table, provenance=_provenance(1))
    write_cleaned_power_time_series(_cleaned(2), table, provenance=_provenance(2))

    assert len(_data_files(table)) == 2


def test_provenance_is_readable_after_the_vacuum_commits(tmp_path: Path) -> None:
    table = tmp_path / "cleaned"
    write_cleaned_power_time_series(
        _cleaned(1), table, provenance=_provenance(1), retention_hours=0
    )
    write_cleaned_power_time_series(
        _cleaned(2), table, provenance=_provenance(5), retention_hours=0
    )

    assert read_cleaning_provenance(table) == _provenance(5)


def test_vacuum_failure_raises_vacuum_error_after_the_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_vacuum(self: DeltaTable, **kwargs: object) -> list[str]:
        raise OSError("transient object-store error")

    monkeypatch.setattr(DeltaTable, "vacuum", failing_vacuum)
    table = tmp_path / "cleaned"

    with pytest.raises(VacuumError):
        write_cleaned_power_time_series(_cleaned(1), table, provenance=_provenance(1))

    assert pl.scan_delta(str(table)).collect().height == 2
