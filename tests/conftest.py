"""Conftest for NGED Substation Forecast tests."""

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
