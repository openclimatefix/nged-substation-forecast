import json
import pytest
from pathlib import Path
from nged_data.io import load_nged_json


def test_load_nged_json_valid(tmp_path: Path):
    # Create a dummy JSON file
    data = {
        "TimeSeriesID": 123,
        "TimeSeriesName": "Test Substation",
        "TimeSeriesType": "PV",
        "Units": "MW",
        "LicenceArea": "EMids",
        "SubstationNumber": 123,
        "SubstationType": "Primary",
        "Latitude": 51.0,
        "Longitude": -1.0,
        "data": [
            {"endTime": "2026-01-01T00:00:00Z", "value": 10.0},
            {"endTime": "2026-01-01T00:30:00Z", "value": 11.0},
        ],
    }
    file_path = tmp_path / "test.json"
    with open(file_path, "w") as f:
        json.dump(data, f)

    metadata_df, time_series_df = load_nged_json(file_path)

    assert "time_series_id" in metadata_df.columns
    assert metadata_df["time_series_id"][0] == 123
    assert len(metadata_df) == 1

    assert "time_series_id" in time_series_df.columns
    assert "period_end_time" in time_series_df.columns
    assert "power" in time_series_df.columns
    assert len(time_series_df) == 2
    assert time_series_df["power"][0] == 10.0


def test_load_nged_json_empty_data(tmp_path: Path):
    # Create a dummy JSON file with empty data
    data = {
        "TimeSeriesID": 123,
        "TimeSeriesName": "Test Substation",
        "TimeSeriesType": "PV",
        "Units": "MW",
        "LicenceArea": "EMids",
        "SubstationNumber": 123,
        "SubstationType": "Primary",
        "Latitude": 51.0,
        "Longitude": -1.0,
        "data": [],
    }
    file_path = tmp_path / "test_empty.json"
    with open(file_path, "w") as f:
        json.dump(data, f)

    metadata_df, time_series_df = load_nged_json(file_path)

    assert len(metadata_df) == 1
    assert len(time_series_df) == 0
    assert set(time_series_df.columns) == {"time_series_id", "period_end_time", "power"}


def test_load_nged_json_missing_file():
    with pytest.raises(FileNotFoundError):
        load_nged_json(Path("non_existent.json"))
