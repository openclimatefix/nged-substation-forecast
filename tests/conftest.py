"""Conftest for NGED Substation Forecast tests."""

from unittest.mock import MagicMock, patch

import pytest


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as requiring external data (CKAN token)"
    )
    config.addinivalue_line(
        "markers", "skip_zarr_refactor: skip test due to Zarr structure limitations"
    )


@pytest.fixture
def mock_ckan_primary_substation_locations():
    """Mock data for primary substation locations."""
    return [
        {
            "substation_number": 110375,
            "name": "Woodland Way",
            "latitude": 52.5200,
            "longitude": -1.0000,
            "region": "West Midlands",
        },
        {
            "substation_number": 110644,
            "name": "Test Substation B",
            "latitude": 52.5300,
            "longitude": -1.0100,
            "region": "West Midlands",
        },
        {
            "substation_number": 110772,
            "name": "Test Substation C",
            "latitude": 52.5400,
            "longitude": -1.0200,
            "region": "West Midlands",
        },
        {
            "substation_number": 110803,
            "name": "Test Substation D",
            "latitude": 52.5500,
            "longitude": -1.0300,
            "region": "West Midlands",
        },
        {
            "substation_number": 110804,
            "name": "Test Substation E",
            "latitude": 52.5600,
            "longitude": -1.0400,
            "region": "West Midlands",
        },
    ]


@pytest.fixture
def mock_ckan_csv_resources():
    """Mock data for CSV resources from CKAN."""
    return [
        {
            "table_id": "12345",
            "name": "aberaeron-primary-transformer-flows",
            "description": "Primary transformer flows for Aberaeron",
            "path": "aberaeron-primary-transformer-flows.csv",
            "resource_type": "csv",
            "format": "CSV",
        },
        {
            "table_id": "12346",
            "name": "abington-primary-transformer-flows",
            "description": "Primary transformer flows for Abington",
            "path": "abington-primary-transformer-flows.csv",
            "resource_type": "csv",
            "format": "CSV",
        },
    ]


@pytest.fixture
def mock_ckan_api(mock_ckan_primary_substation_locations, mock_ckan_csv_resources):
    """Mock CKAN API client that returns mock data."""
    with patch("nged_substation_forecast.defs.nged_assets.ckan") as mock_ckan:
        mock_ckan_instance = MagicMock()
        mock_ckan_instance.get_primary_substation_locations.return_value = (
            mock_ckan_primary_substation_locations
        )
        mock_ckan_instance.get_csv_resources_for_live_primary_substation_flows.return_value = (
            mock_ckan_csv_resources
        )
        mock_ckan.get_primary_substation_locations.return_value = (
            mock_ckan_primary_substation_locations
        )
        mock_ckan.get_csv_resources_for_live_primary_substation_flows.return_value = (
            mock_ckan_csv_resources
        )
        yield mock_ckan
