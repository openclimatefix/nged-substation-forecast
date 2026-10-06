from datetime import UTC, datetime
from pathlib import Path

import patito as pt
import polars as pl
import pytest
from contracts.common import UTC_DATETIME_DTYPE
from contracts.power_schemas import CleanedPowerTimeSeries, PowerTimeSeries, TimeSeriesMetadata
from deltalake import CommitProperties, DeltaTable, write_deltalake
from nged_data import cleaning
from nged_data.cleaning import (
    CleaningProvenance,
    current_git_sha,
    flag_nged_power,
    read_cleaning_provenance,
)

T0 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
T1 = datetime(2026, 1, 1, 0, 30, tzinfo=UTC)


def _metadata(substation_types: dict[int, str]) -> pt.DataFrame[TimeSeriesMetadata]:
    return (
        pt.DataFrame(
            [
                {
                    "time_series_id": time_series_id,
                    "time_series_name": f"Series {time_series_id}",
                    "time_series_type": "Disaggregated Demand",
                    "units": "MW",
                    "licence_area": "EMids",
                    "substation_number": time_series_id,
                    "substation_type": substation_type,
                    "latitude": 52.0,
                    "longitude": -1.0,
                    "h3_res_5": 599423199024775167,
                }
                for time_series_id, substation_type in substation_types.items()
            ]
        )
        .set_model(TimeSeriesMetadata)
        .cast()
        .validate()
    )


def _power(rows: list[tuple[int, datetime, float]]) -> pt.LazyFrame[PowerTimeSeries]:
    frame = pl.DataFrame(
        {
            "time_series_id": [row[0] for row in rows],
            "time": [row[1] for row in rows],
            "power": [row[2] for row in rows],
        },
        schema_overrides={"time_series_id": pl.Int32, "power": pl.Float32},
    ).cast({"time": UTC_DATETIME_DTYPE})
    return pt.LazyFrame.from_existing(frame.lazy()).set_model(PowerTimeSeries)


def _flag(
    rows: list[tuple[int, datetime, float]], substation_types: dict[int, str]
) -> pl.DataFrame:
    flagged = flag_nged_power(_power(rows), _metadata(substation_types)).collect()
    return pl.DataFrame._from_pydf(flagged._df)


def test_substation_zero_is_flagged():
    result = _flag([(1, T0, 0.0)], {1: "Primary"})
    assert result["drop_reason"].to_list() == ["substation_zero"]


@pytest.mark.parametrize("substation_type", ["Primary", "BSP", "GSP"])
def test_substation_zero_flags_every_substation_type(substation_type: str):
    result = _flag([(1, T0, 0.0)], {1: substation_type})
    assert result["drop_reason"].to_list() == ["substation_zero"]


def test_substation_non_zero_reading_is_not_flagged():
    result = _flag([(1, T0, 0.5), (1, T1, -0.5)], {1: "Primary"})
    assert result["drop_reason"].to_list() == [None, None]


def test_generator_zero_is_not_flagged():
    result = _flag([(1, T0, 0.0)], {1: "HV Customer"})
    assert result["drop_reason"].to_list() == [None]


def test_zero_from_series_missing_from_roster_is_not_flagged():
    result = _flag([(1, T0, 0.0), (2, T0, 0.0)], {1: "Primary"})
    assert result["drop_reason"].to_list() == ["substation_zero", None]


def test_flag_nged_power_returns_exactly_the_input_rows_in_order():
    rows = [(1, T0, 0.0), (1, T1, 2.0), (2, T0, 0.0), (3, T0, 4.0)]
    result = _flag(rows, {1: "Primary", 2: "HV Customer"})
    assert result.columns == ["time_series_id", "time", "power", "drop_reason"]
    assert result.select("time_series_id", "time", "power").rows() == rows
    CleanedPowerTimeSeries.validate(result)


def _write_cleaned(path: Path, provenance: CleaningProvenance | None) -> None:
    write_deltalake(
        path,
        pl.DataFrame({"a": [1]}).to_arrow(),
        mode="overwrite",
        commit_properties=CommitProperties(
            custom_metadata=provenance.to_commit_metadata() if provenance else None
        ),
    )


def test_read_cleaning_provenance_round_trips_after_vacuum(tmp_path: Path):
    provenance = CleaningProvenance(
        raw_table_id="abc", raw_version=7, code_hash="hash", git_sha="deadbeef"
    )
    _write_cleaned(tmp_path / "t.delta", provenance)
    DeltaTable(tmp_path / "t.delta").vacuum(
        retention_hours=0, dry_run=False, enforce_retention_duration=False
    )
    assert read_cleaning_provenance(tmp_path / "t.delta") == provenance


def test_read_cleaning_provenance_returns_newest_write(tmp_path: Path):
    for version in (1, 2):
        _write_cleaned(
            tmp_path / "t.delta",
            CleaningProvenance(raw_table_id="abc", raw_version=version, code_hash="h", git_sha="s"),
        )
    provenance = read_cleaning_provenance(tmp_path / "t.delta")
    assert provenance is not None
    assert provenance.raw_version == 2


def test_read_cleaning_provenance_is_none_for_missing_table(tmp_path: Path):
    assert read_cleaning_provenance(tmp_path / "missing.delta") is None


def test_read_cleaning_provenance_is_none_for_missing_key(tmp_path: Path):
    _write_cleaned(tmp_path / "t.delta", None)
    assert read_cleaning_provenance(tmp_path / "t.delta") is None


def test_current_git_sha_prefers_git(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cleaning, "get_git_info", lambda: {"git_sha": "abc", "git_dirty": "false"})
    monkeypatch.setenv("GIT_SHA", "from_env")
    assert current_git_sha() == "abc"


def test_current_git_sha_falls_back_to_env_var(monkeypatch: pytest.MonkeyPatch):
    unknown = {"git_sha": cleaning.UNKNOWN, "git_dirty": cleaning.UNKNOWN}
    monkeypatch.setattr(cleaning, "get_git_info", lambda: unknown)
    monkeypatch.setenv("GIT_SHA", "from_env")
    assert current_git_sha() == "from_env"


@pytest.mark.parametrize("env_value", [None, ""])
def test_current_git_sha_is_unknown_without_git_or_env(
    monkeypatch: pytest.MonkeyPatch, env_value: str | None
):
    unknown = {"git_sha": cleaning.UNKNOWN, "git_dirty": cleaning.UNKNOWN}
    monkeypatch.setattr(cleaning, "get_git_info", lambda: unknown)
    if env_value is None:
        monkeypatch.delenv("GIT_SHA", raising=False)
    else:
        monkeypatch.setenv("GIT_SHA", env_value)
    assert current_git_sha() == cleaning.UNKNOWN
