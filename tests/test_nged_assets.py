import pytest
from unittest.mock import MagicMock
from pathlib import Path
from datetime import datetime
import dagster as dg
import polars as pl
from nged_substation_forecast.defs.nged_assets import nged_json_live_asset, nged_json_archive_asset
from contracts.settings import Settings


@pytest.fixture
def mock_settings(tmp_path: Path):
    settings = MagicMock(spec=Settings)
    settings.nged_data_path = tmp_path
    settings.data_quality = MagicMock()
    return settings


def test_nged_json_live_asset(mock_settings, tmp_path: Path):
    # Create dummy JSON file
    json_dir = tmp_path / "json" / "live" / "2026-01-26-00"
    json_dir.mkdir(parents=True, exist_ok=True)

    file_path = json_dir / "test.json"
    with open(file_path, "w") as f:
        f.write(
            '{"metadata_field": "value", "data": [{"period_end_time": "2026-01-01T00:00:00Z", "power": 10.0, "MVA": 12.0}]}'
        )

    # Build context
    context = dg.build_asset_context(partition_key="2026-01-26-00:00")

    # Mock load_nged_json, PowerTimeSeries.validate, append_to_delta
    with MagicMock() as mock_load, MagicMock() as mock_validate, MagicMock() as mock_append:
        with pytest.MonkeyPatch.context() as m:
            m.setattr("nged_substation_forecast.defs.nged_assets.load_nged_json", mock_load)
            m.setattr("contracts.data_schemas.PowerTimeSeries.validate", mock_validate)
            m.setattr("nged_substation_forecast.defs.nged_assets.append_to_delta", mock_append)

            mock_load.return_value = (MagicMock(), MagicMock())
            mock_validate.return_value = pl.DataFrame(
                {
                    "time_series_id": [1],
                    "period_end_time": [datetime(2026, 1, 1, 0, 0)],
                    "power": [10.0],
                }
            )

            nged_json_live_asset(context, mock_settings)

            mock_load.assert_called_once()
            mock_validate.assert_called_once()
            mock_append.assert_called_once()


def test_nged_json_archive_asset(mock_settings, tmp_path: Path):
    # Create dummy JSON file
    json_dir = tmp_path / "json" / "archive"
    json_dir.mkdir(parents=True, exist_ok=True)

    file_path = json_dir / "test.json"
    with open(file_path, "w") as f:
        f.write(
            '{"metadata_field": "value", "data": [{"period_end_time": "2026-01-01T00:00:00Z", "power": 10.0, "MVA": 12.0}]}'
        )

    # Build context
    context = dg.build_asset_context()

    # Mock load_nged_json, PowerTimeSeries.validate, append_to_delta
    with (
        MagicMock() as mock_load,
        MagicMock() as mock_validate,
        MagicMock() as mock_append,
    ):
        with pytest.MonkeyPatch.context() as m:
            m.setattr("nged_substation_forecast.defs.nged_assets.load_nged_json", mock_load)
            m.setattr("contracts.data_schemas.PowerTimeSeries.validate", mock_validate)
            m.setattr("nged_substation_forecast.defs.nged_assets.append_to_delta", mock_append)

            mock_load.return_value = (MagicMock(), MagicMock())
            mock_validate.return_value = pl.DataFrame(
                {
                    "time_series_id": [1],
                    "period_end_time": [datetime(2026, 1, 1, 0, 0)],
                    "power": [10.0],
                }
            )

            nged_json_archive_asset(context, mock_settings)

            mock_load.assert_called_once()
            mock_validate.assert_called_once()
            mock_append.assert_called_once()
