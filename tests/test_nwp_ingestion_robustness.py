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
import numpy as np
import polars as pl
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from dynamical_data.processing import download_ecmwf, process_ecmwf_dataset
from xgboost_forecaster.model import XGBoostForecaster
from contracts.hydra_schemas import NwpModel, ModelConfig, ModelFeaturesConfig

# Path to the GeoJSON needed for H3 grid generation
GEOJSON_PATH = Path("packages/dynamical_data/england_scotland_wales.geojson")


@pytest.fixture
def h3_grid():
    """Fixture to provide a small dummy H3 grid for testing.

    We don't use the full GB grid because it's too slow to generate for tests.
    """
    return pl.DataFrame(
        {
            "h3_index": [123456789],  # Dummy index
            "nwp_lat": [55.9],
            "nwp_lng": [-3.2],
            "proportion": [1.0],
        },
        schema={
            "h3_index": pl.UInt64,
            "nwp_lat": pl.Float32,
            "nwp_lng": pl.Float32,
            "proportion": pl.Float32,
        },
    )


def test_valid_production_like_zarr_loading(production_like_zarr_path, h3_grid):
    """Verify that the production-like Zarr can be loaded and processed by the pipeline.

    We test the actual production ingestion functions (download_ecmwf and
    process_ecmwf_dataset) to ensure they work correctly with our synthetic data.
    """
    ds = xr.open_zarr(production_like_zarr_path)

    # 1. Test download_ecmwf (slicing and selection)
    init_time = ds.init_time.values[0]
    downloaded_ds = download_ecmwf(init_time, ds, h3_grid)

    assert isinstance(downloaded_ds, xr.Dataset)
    assert "temperature_2m" in downloaded_ds.data_vars

    # 2. Test process_ecmwf_dataset (aggregation and scaling)
    # Convert numpy datetime64 to python datetime for the function
    init_dt = datetime.fromtimestamp(init_time.astype("datetime64[s]").astype(int), tz=timezone.utc)

    processed_df = process_ecmwf_dataset(
        nwp_init_time=init_dt, loaded_ds=downloaded_ds, h3_grid=h3_grid
    )

    assert isinstance(processed_df, pl.DataFrame)
    assert not processed_df.is_empty()
    # Check for scaled columns (uint8)
    # Note: process_ecmwf_dataset returns Nwp-validated DF, which has original names but uint8 dtype
    assert "temperature_2m" in processed_df.columns
    assert processed_df["temperature_2m"].dtype == pl.UInt8


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
def test_broken_zarr_ingestion_fails_loudly(broken_zarr_factory, broken_type, h3_grid):
    """Verify that the pipeline fails loudly when encountering malformed Zarr data.

    We test that the actual production ingestion functions raise appropriate
    exceptions when given broken data.
    """
    zarr_path = broken_zarr_factory(broken_type)
    ds = xr.open_zarr(zarr_path)
    init_time = ds.init_time.values[0]

    # Depending on the broken type, different parts of the pipeline should fail
    with pytest.raises((KeyError, ValueError, AttributeError, TypeError, Exception)):
        downloaded_ds = download_ecmwf(init_time, ds, h3_grid)

        init_dt = datetime.fromtimestamp(
            init_time.astype("datetime64[s]").astype(int), tz=timezone.utc
        )
        process_ecmwf_dataset(nwp_init_time=init_dt, loaded_ds=downloaded_ds, h3_grid=h3_grid)


def test_temporal_deduplication_last_update_wins(tmp_path, h3_grid):
    """Verify that the pipeline correctly handles overlapping forecasts using 'last update wins'.

    This test simulates two overlapping forecasts initialized at different times
    and verifies that the deduplication logic selects the most recent forecast.
    """
    from dynamical_data.scripts.create_production_like_test_zarr import (
        create_production_like_ecmwf_zarr,
    )

    # 1. Create two overlapping forecasts
    # Forecast 1: Initialized at 00:00
    init_time_1 = "2026-03-01T00:00:00"
    zarr_path_1 = tmp_path / "forecast_1.zarr"
    create_production_like_ecmwf_zarr(zarr_path_1, init_time=init_time_1, seed=42)

    # Forecast 2: Initialized at 06:00 (more recent)
    init_time_2 = "2026-03-01T06:00:00"
    zarr_path_2 = tmp_path / "forecast_2.zarr"
    create_production_like_ecmwf_zarr(zarr_path_2, init_time=init_time_2, seed=43)

    # 2. Process both forecasts
    ds1 = xr.open_zarr(zarr_path_1)
    ds2 = xr.open_zarr(zarr_path_2)

    dt1 = datetime.fromisoformat(init_time_1).replace(tzinfo=timezone.utc)
    dt2 = datetime.fromisoformat(init_time_2).replace(tzinfo=timezone.utc)

    df1 = process_ecmwf_dataset(
        dt1, download_ecmwf(np.datetime64(init_time_1), ds1, h3_grid), h3_grid
    )
    df2 = process_ecmwf_dataset(
        dt2, download_ecmwf(np.datetime64(init_time_2), ds2, h3_grid), h3_grid
    )

    # 3. Combine and deduplicate using XGBoostForecaster logic
    # We use the _prepare_and_join_nwps method which implements the deduplication
    forecaster = XGBoostForecaster()

    # Mock the config to have a 0-hour delay for simpler testing
    forecaster.config = ModelConfig(
        power_fcst_model_name="test", nwp_availability_delay_hours=0, features=ModelFeaturesConfig()
    )

    # Use diagonal concat to handle potential column order differences
    nwps = {NwpModel.ECMWF_ENS_0_25DEG: pl.concat([df1, df2], how="diagonal").lazy()}

    # We want to test the deduplication when collapse_lead_times=True
    # We need a nwp_cutoff that is after both init times + delay
    nwp_cutoff = datetime.fromisoformat("2026-03-01T12:00:00").replace(tzinfo=timezone.utc)

    combined_lf = forecaster._prepare_and_join_nwps(
        nwps, nwp_cutoff=nwp_cutoff, collapse_lead_times=True
    )

    combined_df = cast(pl.DataFrame, combined_lf.collect())

    # 4. Verify that for overlapping valid times, the one from init_time_2 is chosen
    # Find a valid time that exists in both
    common_valid_times = set(df1["valid_time"]).intersection(set(df2["valid_time"]))
    assert common_valid_times, "No common valid times found for testing deduplication"

    for vt in common_valid_times:
        # In the combined DF, the init_time for this valid_time should be dt2
        # We use .filter() on the collected DataFrame
        init_times = combined_df.filter(pl.col("valid_time") == vt).select("init_time").unique()
        assert len(init_times) == 1
        assert init_times.item() == dt2
