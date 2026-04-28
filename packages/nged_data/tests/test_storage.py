from pathlib import Path

import patito as pt
import polars as pl
from contracts.power_schemas import TimeSeriesMetadata
from nged_data.storage import upsert_metadata


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
