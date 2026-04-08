import json
import pytest
from pathlib import Path
from nged_json_data.io import load_nged_json


def test_load_nged_json_valid(tmp_path: Path):
    # Create a dummy JSON file
    data = {
        "metadata_field": "value",
        "data": [
            {"timestamp": "2026-01-01T00:00:00Z", "MW": 10.0, "MVA": 12.0},
            {"timestamp": "2026-01-01T01:00:00Z", "MW": 11.0, "MVA": 13.0},
        ],
    }
    file_path = tmp_path / "test.json"
    with open(file_path, "w") as f:
        json.dump(data, f)

    metadata_df, time_series_df = load_nged_json(file_path)

    assert "metadata_field" in metadata_df.columns
    assert "data" not in metadata_df.columns
    assert len(metadata_df) == 1
    assert metadata_df["metadata_field"][0] == "value"

    assert "timestamp" in time_series_df.columns
    assert "MW" in time_series_df.columns
    assert "MVA" in time_series_df.columns
    assert len(time_series_df) == 2
    assert time_series_df["MW"][0] == 10.0


def test_load_nged_json_empty_data(tmp_path: Path):
    # Create a dummy JSON file with empty data
    data = {"metadata_field": "value", "data": []}
    file_path = tmp_path / "test_empty.json"
    with open(file_path, "w") as f:
        json.dump(data, f)

    metadata_df, time_series_df = load_nged_json(file_path)

    assert len(metadata_df) == 1
    assert len(time_series_df) == 0
    # If data is empty, we don't have columns
    assert len(time_series_df.columns) == 0


def test_load_nged_json_missing_file():
    with pytest.raises(FileNotFoundError):
        load_nged_json(Path("non_existent.json"))
