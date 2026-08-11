from datetime import UTC, datetime
from pathlib import Path

import patito as pt
import polars as pl
import pytest
from contracts.common import UTC_DATETIME_DTYPE
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from nged_data import storage
from nged_data.storage import (
    _process_file_listing,
    _ProcessedFileListing,
    _RawFileListItem,
    select_new_rows,
    time_series_coverage,
    upsert_metadata,
)


def test_upsert_metadata_new_file(tmp_path: Path):
    metadata_path = tmp_path / "metadata.parquet"

    # Create dummy metadata
    metadata = (
        pt.DataFrame(
            [
                {
                    "time_series_id": 1,
                    "time_series_name": "Test Substation",
                    "time_series_type": "Disaggregated Demand",
                    "units": "MW",
                    "licence_area": "EMids",
                    "substation_number": 1,
                    "substation_type": "Primary",
                    "latitude": 52.0,
                    "longitude": -1.0,
                    "h3_res_5": 599423199024775167,
                }
            ]
        )
        .set_model(TimeSeriesMetadata)
        .cast()
        .validate()
    )

    upsert_metadata(metadata, str(metadata_path))

    assert metadata_path.exists()

    # Read back and verify
    read_metadata = pl.read_parquet(metadata_path)
    assert read_metadata.height == 1
    assert read_metadata["time_series_id"].item() == 1


def test_upsert_metadata_creates_missing_parent_dir(tmp_path: Path):
    """A first-ever run writes into a data root whose subdirectory doesn't exist yet; the create
    branch must make the parent dir rather than raising FileNotFoundError from write_parquet."""
    metadata_path = tmp_path / "NGED" / "metadata.parquet"  # parent NGED/ does not exist
    metadata = (
        pt.DataFrame(
            [
                {
                    "time_series_id": 1,
                    "time_series_name": "Test Substation",
                    "time_series_type": "Disaggregated Demand",
                    "units": "MW",
                    "licence_area": "EMids",
                    "substation_number": 1,
                    "substation_type": "Primary",
                    "latitude": 52.0,
                    "longitude": -1.0,
                    "h3_res_5": 599423199024775167,
                }
            ]
        )
        .set_model(TimeSeriesMetadata)
        .cast()
        .validate()
    )

    upsert_metadata(metadata, str(metadata_path))

    assert metadata_path.exists()


def test_upsert_metadata_merge(tmp_path: Path):
    metadata_path = tmp_path / "metadata.parquet"

    # Create initial metadata
    initial_metadata = (
        pt.DataFrame(
            [
                {
                    "time_series_id": 1,
                    "time_series_name": "Old Name",
                    "time_series_type": "Disaggregated Demand",
                    "units": "MW",
                    "licence_area": "EMids",
                    "substation_number": 1,
                    "substation_type": "Primary",
                    "latitude": 52.0,
                    "longitude": -1.0,
                    "h3_res_5": 599423199024775167,
                }
            ]
        )
        .set_model(TimeSeriesMetadata)
        .cast()
        .validate()
    )

    initial_metadata.write_parquet(metadata_path)

    # Create new metadata for same ID
    new_metadata = (
        pt.DataFrame(
            [
                {
                    "time_series_id": 1,
                    "time_series_name": "New Name",
                    "time_series_type": "Disaggregated Demand",
                    "units": "MW",
                    "licence_area": "EMids",
                    "substation_number": 1,
                    "substation_type": "Primary",
                    "latitude": 52.0,
                    "longitude": -1.0,
                    "h3_res_5": 599423199024775167,
                }
            ]
        )
        .set_model(TimeSeriesMetadata)
        .cast()
        .validate()
    )

    upsert_metadata(new_metadata, str(metadata_path))

    # Read back and verify
    read_metadata = pl.read_parquet(metadata_path)
    assert read_metadata.height == 1
    assert read_metadata["time_series_name"].item() == "New Name"


def test_upsert_metadata_returns_diff(tmp_path: Path):
    metadata_path = tmp_path / "metadata.parquet"

    # 1. Create initial metadata
    initial_data = [
        {
            "time_series_id": 1,
            "time_series_name": "ID 1 - Original",
            "time_series_type": "Disaggregated Demand",
            "units": "MW",
            "licence_area": "EMids",
            "substation_number": 1,
            "substation_type": "Primary",
            "latitude": 52.0,
            "longitude": -1.0,
            "h3_res_5": 599423199024775167,
        },
        {
            "time_series_id": 2,
            "time_series_name": "ID 2 - Original",
            "time_series_type": "Disaggregated Demand",
            "units": "MW",
            "licence_area": "EMids",
            "substation_number": 2,
            "substation_type": "Primary",
            "latitude": 52.0,
            "longitude": -1.0,
            "h3_res_5": 599423199024775167,
        },
    ]
    initial_metadata = pt.DataFrame(initial_data).set_model(TimeSeriesMetadata).cast().validate()
    initial_metadata.write_parquet(metadata_path)

    # 2. Create new metadata
    new_data = [
        # Identical to ID 1
        {
            "time_series_id": 1,
            "time_series_name": "ID 1 - Original",
            "time_series_type": "Disaggregated Demand",
            "units": "MW",
            "licence_area": "EMids",
            "substation_number": 1,
            "substation_type": "Primary",
            "latitude": 52.0,
            "longitude": -1.0,
            "h3_res_5": 599423199024775167,
        },
        # Updated ID 2
        {
            "time_series_id": 2,
            "time_series_name": "ID 2 - Updated",
            "time_series_type": "Disaggregated Demand",
            "units": "MW",
            "licence_area": "EMids",
            "substation_number": 2,
            "substation_type": "Primary",
            "latitude": 52.0,
            "longitude": -1.0,
            "h3_res_5": 599423199024775167,
        },
        # New ID 3
        {
            "time_series_id": 3,
            "time_series_name": "ID 3 - New",
            "time_series_type": "Disaggregated Demand",
            "units": "MW",
            "licence_area": "EMids",
            "substation_number": 3,
            "substation_type": "Primary",
            "latitude": 52.0,
            "longitude": -1.0,
            "h3_res_5": 599423199024775167,
        },
    ]
    new_metadata = pt.DataFrame(new_data).set_model(TimeSeriesMetadata).cast().validate()

    # 3. Call upsert_metadata
    stats = upsert_metadata(new_metadata, str(metadata_path))

    # 4. Assertions
    assert stats["metadata_n_new_TimeSeriesIDs"] == 1
    assert stats["metadata_n_updated_TimeSeriesIDs"] == 1
    assert set(stats["metadata_updated_TimeSeriesIDs"]) == {2}

    # Verify file content
    final_metadata = pl.read_parquet(metadata_path)
    assert final_metadata.height == 3
    assert (
        final_metadata.filter(pl.col("time_series_id") == 1)["time_series_name"].item()
        == "ID 1 - Original"
    )
    assert (
        final_metadata.filter(pl.col("time_series_id") == 2)["time_series_name"].item()
        == "ID 2 - Updated"
    )
    assert (
        final_metadata.filter(pl.col("time_series_id") == 3)["time_series_name"].item()
        == "ID 3 - New"
    )


def _roster(ids: list[int], name: str = "ID", **extra: object) -> pt.DataFrame[TimeSeriesMetadata]:
    """A valid roster covering ``ids``, plus any ``extra`` columns applied to every row."""
    rows = [
        {
            "time_series_id": i,
            "time_series_name": f"{name} {i}",
            "time_series_type": "Disaggregated Demand",
            "units": "MW",
            "licence_area": "EMids",
            "substation_number": i,
            "substation_type": "Primary",
            "latitude": 52.0,
            "longitude": -1.0,
            "h3_res_5": 599423199024775167,
            **extra,
        }
        for i in ids
    ]
    return pt.DataFrame(rows).set_model(TimeSeriesMetadata).cast().validate()


def test_upsert_metadata_rebuilds_when_the_stored_roster_will_not_parse(tmp_path: Path):
    """The reported bug: a roster left half-written by a killed process, or truncated by a full
    disk, no longer wedges the ingest. It is detected, reported and rewritten."""
    metadata_path = tmp_path / "metadata.parquet"
    metadata_path.write_bytes(b"not a parquet file")

    stats = upsert_metadata(_roster([1, 2]), str(metadata_path))

    assert "metadata_roster_rebuilt_reason" in stats
    assert "ComputeError" in stats["metadata_roster_rebuilt_reason"]
    assert set(pl.read_parquet(metadata_path)["time_series_id"]) == {1, 2}


def test_upsert_metadata_rebuilds_when_the_stored_roster_fails_its_contract(tmp_path: Path):
    """A stored roster missing a required column is *wrong*, not merely stale, so it is treated as
    absent rather than raising `DataFrameValidationError` out of the asset."""
    metadata_path = tmp_path / "metadata.parquet"
    _roster([1]).drop("substation_type").write_parquet(metadata_path)

    stats = upsert_metadata(_roster([2]), str(metadata_path))

    assert "DataFrameValidationError" in stats["metadata_roster_rebuilt_reason"]
    # The rebuild is from this run's snapshot alone, so id 1 is gone: a rebuilt roster can be thin.
    assert set(pl.read_parquet(metadata_path)["time_series_id"]) == {2}


def test_upsert_metadata_merges_a_snapshot_missing_the_optional_columns(tmp_path: Path):
    """`TimeSeriesMetadata` has four `allow_missing` fields, so a snapshot can be narrower than the
    stored roster and still validate. Merging the two must not raise, and a field the snapshot no
    longer carries must be *cleared*, so the roster keeps meaning "the latest snapshot of what NGED
    published"."""
    metadata_path = tmp_path / "metadata.parquet"
    _roster([1, 2], information="note").write_parquet(metadata_path)

    # This run's snapshot covers id 2 only, and carries no `information` column at all.
    snapshot = _roster([2], name="Renamed")
    assert "information" not in snapshot.columns
    upsert_metadata(snapshot, str(metadata_path))

    final = pl.read_parquet(metadata_path)
    assert final.filter(pl.col("time_series_id") == 2)["information"].item() is None
    assert final.filter(pl.col("time_series_id") == 1)["information"].item() == "note"
    assert final.filter(pl.col("time_series_id") == 2)["time_series_name"].item() == "Renamed 2"
    TimeSeriesMetadata.validate(final)


def test_upsert_metadata_ignores_the_stored_column_order(tmp_path: Path):
    """`hash_rows` is column-order sensitive, so a stored roster whose columns happen to sit in a
    different order must not be reported as wholly changed and rewritten every run."""
    metadata_path = tmp_path / "metadata.parquet"
    roster = _roster([1, 2])
    roster.select(sorted(roster.columns)).write_parquet(metadata_path)
    mtime_before = metadata_path.stat().st_mtime_ns

    stats = upsert_metadata(roster, str(metadata_path))

    assert stats["metadata_n_new_TimeSeriesIDs"] == 0
    assert stats["metadata_n_updated_TimeSeriesIDs"] == 0
    assert metadata_path.stat().st_mtime_ns == mtime_before  # not rewritten at all


def test_upsert_metadata_does_not_rebuild_on_a_panic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A Rust panic is evidence about the process, not about the file, so it must propagate and
    leave the stored roster alone — the asset's own guard degrades the run without rewriting.

    Unlike its siblings this passes before the fix too, because there was no handler to be too wide.
    It guards `_read_existing_roster` catching `Exception` and not `BaseException` — a one-word
    difference with no other visible symptom.
    """
    metadata_path = tmp_path / "metadata.parquet"
    _roster([1]).write_parquet(metadata_path)
    stored_before = metadata_path.read_bytes()

    class _Panic(BaseException):
        """Stands in for a pyo3 `PanicException`, which derives from `BaseException`."""

    def boom(*_: object, **__: object) -> pl.DataFrame:
        raise _Panic("rust panicked")

    monkeypatch.setattr(storage.pl, "read_parquet", boom)

    with pytest.raises(_Panic):
        upsert_metadata(_roster([2]), str(metadata_path))

    assert metadata_path.read_bytes() == stored_before


_EXAMPLE_OBJECT_KEY = (
    "timeseries/1774512000000_1774533600000/TimeSeries_23_20260326T080000Z_20260326T140000Z.json"
)
"""One real NGED object key, whose embedded epoch-millisecond window and id the parser extracts."""


def test_parse_file_listing_valid():
    raw_file_listing: list[_RawFileListItem] = [
        {
            "path": _EXAMPLE_OBJECT_KEY,
            "filesize_bytes": 1024,
        }
    ]

    result = _process_file_listing(raw_file_listing)

    assert result.height == 1
    assert result["time_series_id"][0] == 23
    assert result["path"][0] == _EXAMPLE_OBJECT_KEY
    assert result["filesize_bytes"][0] == 1024
    assert result["start_time"][0] == datetime(2026, 3, 26, 8, 0, 0, tzinfo=UTC)
    assert result["end_time"][0] == datetime(2026, 3, 26, 14, 0, 0, tzinfo=UTC)


def test_select_new_rows_file_listing(tmp_path: Path):
    """Regression: trailing comma made filtered_df a tuple, causing superfluous column_0 error."""
    delta_path = tmp_path / "power.delta"

    pl.DataFrame(
        {
            "time_series_id": pl.Series([1], dtype=pl.Int32),
            "time": pl.Series([datetime(2026, 1, 1, 12, 0, tzinfo=UTC)]).cast(UTC_DATETIME_DTYPE),
            "power": pl.Series([1.0], dtype=pl.Float32),
        }
    ).write_delta(delta_path)

    raw = pl.DataFrame(
        {
            "path": ["old.json", "new_ts1.json", "new_ts2.json"],
            "filesize_bytes": pl.Series([1000, 1000, 1000], dtype=pl.Int64),
            "time_series_id": pl.Series([1, 1, 2], dtype=pl.Int32),
            "start_time": pl.Series(
                [
                    datetime(2026, 1, 1, 6, 0, tzinfo=UTC),
                    datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
                    datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
                ]
            ).cast(UTC_DATETIME_DTYPE),
            "end_time": pl.Series(
                [
                    datetime(
                        2026, 1, 1, 12, 0, tzinfo=UTC
                    ),  # equals last_time for ts_id=1 → excluded
                    datetime(2026, 1, 1, 18, 0, tzinfo=UTC),  # > last_time for ts_id=1 → included
                    datetime(2026, 1, 1, 6, 0, tzinfo=UTC),  # ts_id=2 not in delta → included
                ]
            ).cast(UTC_DATETIME_DTYPE),
        }
    )
    file_listing = pt.DataFrame(raw).set_model(_ProcessedFileListing).validate()

    result = select_new_rows(file_listing, str(delta_path))

    assert result.height == 2
    assert set(result["path"].to_list()) == {"new_ts1.json", "new_ts2.json"}
    _ProcessedFileListing.validate(result)  # schema must survive filtering


def test_select_new_rows_power_time_series(tmp_path: Path):
    """select_new_rows must filter PowerTimeSeries rows newer than the Delta table max."""
    delta_path = tmp_path / "power.delta"
    T = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    pl.DataFrame(
        {
            "time_series_id": pl.Series([1], dtype=pl.Int32),
            "time": pl.Series([T]).cast(UTC_DATETIME_DTYPE),
            "power": pl.Series([1.0], dtype=pl.Float32),
        }
    ).write_delta(delta_path)

    input_power = PowerTimeSeries.validate(
        pl.DataFrame(
            {
                "time_series_id": pl.Series([1, 1], dtype=pl.Int32),
                "time": pl.Series([T, datetime(2026, 1, 1, 12, 30, tzinfo=UTC)]).cast(
                    UTC_DATETIME_DTYPE
                ),
                "power": pl.Series([1.0, 2.0], dtype=pl.Float32),
            }
        )
    )

    result = select_new_rows(input_power, str(delta_path))

    assert result.height == 1
    assert result["time"][0] == datetime(2026, 1, 1, 12, 30, tzinfo=UTC)
    PowerTimeSeries.validate(result)  # schema must survive filtering


def test_time_series_coverage(tmp_path: Path):
    """Returns the earliest and latest ``time`` per ``time_series_id`` from the Delta table."""
    delta_path = tmp_path / "power.delta"
    pl.DataFrame(
        {
            "time_series_id": pl.Series([1, 1, 2], dtype=pl.Int32),
            "time": pl.Series(
                [
                    datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
                    datetime(2026, 1, 1, 12, 30, tzinfo=UTC),
                    datetime(2026, 1, 2, 9, 0, tzinfo=UTC),
                ]
            ).cast(UTC_DATETIME_DTYPE),
            "power": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float32),
        }
    ).write_delta(delta_path)

    coverage = time_series_coverage(str(delta_path)).sort("time_series_id")

    assert coverage["time_series_id"].to_list() == [1, 2]
    assert coverage["first_time"].to_list() == [
        datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        datetime(2026, 1, 2, 9, 0, tzinfo=UTC),
    ]
    assert coverage["last_time"].to_list() == [
        datetime(2026, 1, 1, 12, 30, tzinfo=UTC),
        datetime(2026, 1, 2, 9, 0, tzinfo=UTC),
    ]


def test_time_series_coverage_absent_table(tmp_path: Path):
    """A missing Delta table yields an empty but correctly-typed frame, not an error."""
    coverage = time_series_coverage(str(tmp_path / "does_not_exist.delta"))
    assert coverage.is_empty()
    assert coverage.columns == ["time_series_id", "first_time", "last_time"]


def test_parse_file_listing_invalid():
    # Invalid path format
    raw_file_listing: list[_RawFileListItem] = [
        {
            "path": "invalid/path/format.json",
            "filesize_bytes": 1024,
        }
    ]

    # The function uses `_TimeSeriesJsonFileListing.validate(paths_df)`
    # If the regex fails, the columns will be null, and validation should fail.
    with pytest.raises(pt.exceptions.DataFrameValidationError):
        _process_file_listing(raw_file_listing)
