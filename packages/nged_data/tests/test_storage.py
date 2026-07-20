from datetime import datetime, timezone
from pathlib import Path

import patito as pt
import polars as pl
import pytest
from contracts.common import UTC_DATETIME_DTYPE
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from nged_data.storage import (
    _ProcessedFileListing,
    _RawFileListItem,
    _process_file_listing,
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


def test_parse_file_listing_valid():
    raw_file_listing: list[_RawFileListItem] = [
        {
            "path": "timeseries/1774512000000_1774533600000/TimeSeries_23_20260326T080000Z_20260326T140000Z.json",
            "filesize_bytes": 1024,
        }
    ]

    result = _process_file_listing(raw_file_listing)

    assert result.height == 1
    assert result["time_series_id"][0] == 23
    assert (
        result["path"][0]
        == "timeseries/1774512000000_1774533600000/TimeSeries_23_20260326T080000Z_20260326T140000Z.json"
    )
    assert result["filesize_bytes"][0] == 1024
    assert result["start_time"][0] == datetime(2026, 3, 26, 8, 0, 0, tzinfo=timezone.utc)
    assert result["end_time"][0] == datetime(2026, 3, 26, 14, 0, 0, tzinfo=timezone.utc)


def test_select_new_rows_file_listing(tmp_path: Path):
    """Regression: trailing comma made filtered_df a tuple, causing superfluous column_0 error."""
    UTC = timezone.utc
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
    UTC = timezone.utc
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
    UTC = timezone.utc
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
