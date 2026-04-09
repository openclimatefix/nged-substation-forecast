import pytest
import polars as pl
import patito as pt
from pathlib import Path
from datetime import datetime
from unittest.mock import patch
from nged_json_data.storage import append_to_delta
from contracts.data_schemas import PowerTimeSeries


@pytest.fixture
def mock_delta_table():
    with patch("nged_json_data.storage.DeltaTable") as mock_dt:
        yield mock_dt


@pytest.fixture
def mock_write_deltalake():
    with patch("nged_json_data.storage.write_deltalake") as mock_write:
        yield mock_write


def test_append_to_delta_new_table(tmp_path: Path, mock_write_deltalake):
    delta_path = tmp_path / "delta"

    # Create a dummy Patito DataFrame
    df = pt.DataFrame[PowerTimeSeries](
        pl.DataFrame(
            {
                "time_series_id": ["1"],
                "end_time": [datetime(2026, 1, 1)],
                "MW": [10.0],
                "MVA": [12.0],
                "MVAr": [1.0],
            }
        )
    )

    append_to_delta(df, delta_path)

    mock_write_deltalake.assert_called_once()
    args, kwargs = mock_write_deltalake.call_args
    assert args[0] == delta_path
    assert kwargs["mode"] == "overwrite"


def test_append_to_delta_existing_table(tmp_path: Path, mock_delta_table, mock_write_deltalake):
    delta_path = tmp_path / "delta"
    delta_path.mkdir()

    # Mock existing keys
    mock_dt = mock_delta_table.return_value
    mock_dt.to_pyarrow_table.return_value = pl.DataFrame(
        {
            "time_series_id": ["1"],
            "end_time": [datetime(2026, 1, 1)],
        }
    ).to_arrow()

    # Create a dummy Patito DataFrame with new data
    df = pt.DataFrame[PowerTimeSeries](
        pl.DataFrame(
            {
                "time_series_id": ["1", "2"],
                "end_time": [datetime(2026, 1, 1), datetime(2026, 1, 2)],
                "MW": [10.0, 20.0],
                "MVA": [12.0, 22.0],
                "MVAr": [1.0, 2.0],
            }
        )
    )

    append_to_delta(df, delta_path)

    # Should have called write_deltalake with only the new data (time_series_id="2")
    mock_write_deltalake.assert_called_once()
    args, kwargs = mock_write_deltalake.call_args
    assert args[0] == delta_path
    assert kwargs["mode"] == "append"

    # Check that only the new data was written
    written_df = pl.from_arrow(args[1])
    assert len(written_df) == 1
    from typing import cast

    assert cast(pl.DataFrame, written_df).item(0, "time_series_id") == "2"
