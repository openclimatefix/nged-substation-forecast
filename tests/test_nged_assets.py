import pytest
from unittest.mock import MagicMock
from pathlib import Path
import dagster as dg
from nged_substation_forecast.defs.nged_assets import nged_json_live_asset, nged_json_archive_asset
from contracts.settings import Settings


@pytest.fixture
def mock_settings(tmp_path: Path):
    settings = MagicMock(spec=Settings)
    settings.nged_data_path = tmp_path
    return settings


def test_nged_json_live_asset(mock_settings, tmp_path: Path):
    # Create dummy JSON file
    json_dir = tmp_path / "json" / "live" / "2026-01-26-00"
    json_dir.mkdir(parents=True, exist_ok=True)

    file_path = json_dir / "test.json"
    with open(file_path, "w") as f:
        f.write(
            '{"metadata_field": "value", "data": [{"timestamp": "2026-01-01T00:00:00Z", "MW": 10.0, "MVA": 12.0}]}'
        )

    # Build context
    context = dg.build_asset_context(partition_key="2026-01-26-00:00")

    # Mock load_nged_json, clean_power_data, append_to_delta
    with MagicMock() as mock_load, MagicMock() as mock_clean, MagicMock() as mock_append:
        with pytest.MonkeyPatch.context() as m:
            m.setattr("nged_substation_forecast.defs.nged_assets.load_nged_json", mock_load)
            m.setattr("nged_substation_forecast.defs.nged_assets.clean_power_data", mock_clean)
            m.setattr("nged_substation_forecast.defs.nged_assets.append_to_delta", mock_append)

            mock_load.return_value = (MagicMock(), MagicMock())
            mock_clean.return_value = MagicMock()

            nged_json_live_asset(context, mock_settings)

            mock_load.assert_called_once()
            mock_clean.assert_called_once()
            mock_append.assert_called_once()


def test_nged_json_archive_asset(mock_settings, tmp_path: Path):
    # Create dummy JSON file
    json_dir = tmp_path / "json" / "archive"
    json_dir.mkdir(parents=True, exist_ok=True)

    file_path = json_dir / "test.json"
    with open(file_path, "w") as f:
        f.write(
            '{"metadata_field": "value", "data": [{"timestamp": "2026-01-01T00:00:00Z", "MW": 10.0, "MVA": 12.0}]}'
        )

    # Build context
    context = dg.build_asset_context()

    # Mock load_nged_json, clean_power_data, append_to_delta, upsert_metadata
    with (
        MagicMock() as mock_load,
        MagicMock() as mock_clean,
        MagicMock() as mock_append,
        MagicMock() as mock_upsert,
    ):
        with pytest.MonkeyPatch.context() as m:
            m.setattr("nged_substation_forecast.defs.nged_assets.load_nged_json", mock_load)
            m.setattr("nged_substation_forecast.defs.nged_assets.clean_power_data", mock_clean)
            m.setattr("nged_substation_forecast.defs.nged_assets.append_to_delta", mock_append)
            m.setattr("nged_substation_forecast.defs.nged_assets.upsert_metadata", mock_upsert)

            mock_load.return_value = (MagicMock(), MagicMock())
            mock_clean.return_value = MagicMock()

            nged_json_archive_asset(context, mock_settings)

            mock_load.assert_called_once()
            mock_clean.assert_called_once()
            mock_append.assert_called_once()
            mock_upsert.assert_called_once()
