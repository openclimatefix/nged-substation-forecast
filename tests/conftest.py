"""Conftest for NGED Substation Forecast tests."""

from unittest.mock import MagicMock, patch

import pytest
from pathlib import Path

from dynamical_data.scripts.create_production_like_test_zarr import (
    create_production_like_ecmwf_zarr,
    create_broken_ecmwf_zarr,
)


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as requiring external data (CKAN token)"
    )
    config.addinivalue_line(
        "markers", "skip_zarr_refactor: skip test due to Zarr structure limitations"
    )
    config.addinivalue_line(
        "markers", "manual: mark test as requiring manual execution (e.g., local data)"
    )


@pytest.fixture
def production_like_zarr_path(tmp_path):
    """
    Fixture that generates a production-like test Zarr store in a temporary directory.
    Returns the Path to the generated Zarr store.
    """
    zarr_path = tmp_path / "ecmwf_production_like.zarr"
    create_production_like_ecmwf_zarr(zarr_path, seed=42)
    return zarr_path


@pytest.fixture
def broken_zarr_factory(tmp_path):
    """
    Fixture factory that generates a broken test Zarr store of a specific type.
    Returns a function that takes a broken_type and returns the Path to the Zarr store.
    """

    def _create_broken(broken_type: str) -> Path:
        zarr_path = tmp_path / f"broken_{broken_type}.zarr"
        create_broken_ecmwf_zarr(zarr_path, broken_type=broken_type, seed=42)
        return zarr_path

    return _create_broken


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
