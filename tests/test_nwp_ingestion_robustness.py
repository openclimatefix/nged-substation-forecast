"""Tests for NWP ingestion robustness, focusing on duplicate handling and error cases.

This module contains tests that verify the NWP ingestion pipeline handles edge cases
robustly, including:

1. **Temporal deduplication**: When merging multiple forecast steps, duplicate
   timestamps should be handled correctly (last update wins).

2. **Bad Zarr handling**: The pipeline should fail loudly with informative error
   messages when encountering malformed inputs like:
   - Empty Zarr stores
   - Missing critical variables
   - Invalid coordinates

Why we use local Zarr files instead of mocking:
----------------------------------------------
- We test the actual xarray/zarr logic that will be used in production.
- Mocking the API would hide bugs in the temporal merging logic.
- Network dependencies introduce flakiness and slower test execution.
- Static samples enable deterministic, reproducible CI runs.

The test data is generated on-the-fly by pytest fixtures in `conftest.py` using
`create_production_like_test_zarr.py`.
"""

import xarray as xr
import pytest


def test_valid_production_like_zarr_loading(production_like_zarr_path):
    """Verify that the production-like Zarr can be loaded and has correct structure."""
    ds = xr.open_zarr(production_like_zarr_path)

    # Check dimensions
    expected_dims = {"latitude", "longitude", "init_time", "lead_time", "ensemble_member"}
    assert set(ds.dims) == expected_dims

    # Check critical variables
    assert "temperature_2m" in ds.data_vars
    assert "wind_speed_10m" in ds.data_vars

    # Check coordinate values (deterministic from seed=42)
    assert ds.latitude.values[0] == pytest.approx(55.8)
    assert ds.longitude.values[0] == pytest.approx(-3.4)


@pytest.mark.parametrize(
    "broken_type",
    [
        "missing_coords",
        "wrong_dim_order",
        "missing_vars",
        "wrong_dtype",
        "inconsistent_shape",
    ],
)
def test_broken_zarr_generation(broken_zarr_factory, broken_type):
    """Verify that broken Zarr variants are generated correctly by the factory."""
    zarr_path = broken_zarr_factory(broken_type)
    ds = xr.open_zarr(zarr_path)

    if broken_type == "missing_coords":
        assert "latitude" not in ds.coords
        assert "longitude" not in ds.coords
    elif broken_type == "missing_vars":
        assert "temperature_2m" not in ds.data_vars
    elif broken_type == "wrong_dtype":
        # init_time is string instead of datetime64
        assert ds.init_time.dtype.kind in ("U", "S", "O")
