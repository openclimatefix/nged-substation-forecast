from datetime import datetime, timezone
from pathlib import Path

import patito as pt
import polars as pl
import pytest
from contracts.power_schemas import TimeSeriesMetadata
from nged_data.storage import _process_file_listing, _RawFileListItem, upsert_metadata


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

    upsert_metadata(metadata, metadata_path)

    assert metadata_path.exists()

    # Read back and verify
    read_metadata = pl.read_parquet(metadata_path)
    assert read_metadata.height == 1
    assert read_metadata["time_series_id"].item() == 1


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

    upsert_metadata(new_metadata, metadata_path)

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
    metadata_diff = upsert_metadata(new_metadata, metadata_path)

    # 4. Assertions
    assert metadata_diff.height == 2
    assert set(metadata_diff["time_series_id"]) == {2, 3}

    # Verify ID 2 is updated
    assert (
        metadata_diff.filter(pl.col("time_series_id") == 2)["time_series_name"].item()
        == "ID 2 - Updated"
    )

    # Verify ID 3 is new
    assert (
        metadata_diff.filter(pl.col("time_series_id") == 3)["time_series_name"].item()
        == "ID 3 - New"
    )

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
