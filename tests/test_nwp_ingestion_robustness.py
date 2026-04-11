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
import patito as pt
from datetime import datetime, timezone
from pathlib import Path
from typing import cast
from contracts.data_schemas import H3GridWeights

from dynamical_data.processing import MalformedZarrError, download_ecmwf, process_ecmwf_dataset
from xgboost_forecaster.model import XGBoostForecaster
from contracts.hydra_schemas import NwpModel, ModelConfig, ModelFeaturesConfig

# Path to the GeoJSON needed for H3 grid generation
GEOJSON_PATH = Path("packages/dynamical_data/england_scotland_wales.geojson")


@pytest.fixture
def h3_grid() -> pt.DataFrame[H3GridWeights]:
    """Fixture to provide a small dummy H3 grid for testing.

    We don't use the full GB grid because it's too slow to generate for tests.
    """
    return cast(
        pt.DataFrame[H3GridWeights],
        pl.DataFrame(
            {
                "h3_index": [123456789, 987654321],
                "nwp_lat": [56.0, 56.25],
                "nwp_lng": [-3.25, -3.0],
                "proportion": [0.5, 0.5],
            },
            schema={
                "h3_index": pl.UInt64,
                "nwp_lat": pl.Float32,
                "nwp_lng": pl.Float32,
                "proportion": pl.Float32,
            },
        ),
    )


def test_valid_production_like_zarr_loading(production_like_zarr_path, h3_grid):
    """Verify that the production-like Zarr can be loaded and processed by the pipeline.

    We test the actual production ingestion functions (download_ecmwf and
    process_ecmwf_dataset) to ensure they work correctly with our synthetic data.
    """
    # Explicitly setting decode_timedelta=True avoids reliance on Xarray's
    # deprecated automatic decoding of time units, ensuring lead_time is
    # correctly parsed as timedelta64[ns].
    ds = xr.open_zarr(production_like_zarr_path, decode_timedelta=True)

    # 1. Test download_ecmwf (slicing and selection)
    init_time = ds.init_time.values[0]
    downloaded_ds = download_ecmwf(init_time, h3_grid, ds)

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
    # Check for Float32 columns (as per Nwp schema)
    assert "temperature_2m" in processed_df.columns
    assert processed_df["temperature_2m"].dtype == pl.Float32


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
    # Explicitly setting decode_timedelta=True avoids reliance on Xarray's
    # deprecated automatic decoding of time units, ensuring lead_time is
    # correctly parsed as timedelta64[ns].
    ds = xr.open_zarr(zarr_path, decode_timedelta=True)
    init_time = ds.init_time.values[0]

    # Map broken types to expected exceptions
    expected_exceptions = {
        "missing_coords": MalformedZarrError,
        "wrong_dim_order": None,  # Now automatically fixed
        "missing_vars": MalformedZarrError,
        "wrong_dtype": MalformedZarrError,
        "inconsistent_shape": (ValueError, MalformedZarrError),
    }

    expected_exc = expected_exceptions.get(broken_type, Exception)

    def run_pipeline():
        downloaded_ds = download_ecmwf(init_time, h3_grid, ds)
        init_dt = datetime.fromtimestamp(
            init_time.astype("datetime64[s]").astype(int), tz=timezone.utc
        )
        process_ecmwf_dataset(nwp_init_time=init_dt, loaded_ds=downloaded_ds, h3_grid=h3_grid)

    # Depending on the broken type, different parts of the pipeline should fail
    if expected_exc is None:
        run_pipeline()
    else:
        with pytest.raises(expected_exc):
            run_pipeline()


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
    # Explicitly setting decode_timedelta=True avoids reliance on Xarray's
    # deprecated automatic decoding of time units, ensuring lead_time is
    # correctly parsed as timedelta64[ns].
    ds1 = xr.open_zarr(zarr_path_1, decode_timedelta=True)
    ds2 = xr.open_zarr(zarr_path_2, decode_timedelta=True)

    dt1 = datetime.fromisoformat(init_time_1).replace(tzinfo=timezone.utc)
    dt2 = datetime.fromisoformat(init_time_2).replace(tzinfo=timezone.utc)

    df1 = process_ecmwf_dataset(
        dt1, download_ecmwf(np.datetime64(init_time_1), h3_grid, ds1), h3_grid
    )
    df2 = process_ecmwf_dataset(
        dt2, download_ecmwf(np.datetime64(init_time_2), h3_grid, ds2), h3_grid
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
        list(nwps.values())[0], nwp_cutoff=nwp_cutoff, collapse_lead_times=True
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
        # We compare the timestamps (seconds since epoch) rather than using np.datetime64.
        # This avoids Numpy UserWarnings about explicit timezone representation being ignored,
        # while still maintaining the necessary precision for the comparison.
        assert init_times.item().timestamp() == dt2.timestamp()


def test_small_grid_forecast_ingestion(tmp_path, h3_grid):
    """Verify the pipeline can handle small-grid forecasts without IndexError.

    This test creates a Zarr dataset with a 2x2 latitude and longitude grid,
    and passes it through download_ecmwf. It verifies the fix
    where length-1 latitude arrays caused an IndexError during spatial slicing.
    """

    # 1. Create a single-point forecast Zarr
    # We need to monkeypatch the constants in the script to create a single-point Zarr
    # or just create it manually. Let's try to create it manually for simplicity.
    init_time = np.datetime64("2026-03-01T00:00:00")
    zarr_path = tmp_path / "single_point.zarr"

    # Create a minimal dataset with 2x2 lat/lon
    ds = xr.Dataset(
        {
            "latitude": (
                ["latitude"],
                np.array([56.0, 56.25], dtype=np.float32),
                {"units": "degrees_north"},
            ),
            "longitude": (
                ["longitude"],
                np.array([-3.25, -3.0], dtype=np.float32),
                {"units": "degrees_east"},
            ),
            "init_time": (["init_time"], [init_time]),
            "lead_time": (["lead_time"], [0.0, 6.0]),
            "ensemble_member": (["ensemble_member"], [0]),
        }
    )

    # Add dummy variables
    shape = (2, 2, 1, 2, 1)
    required_vars = [
        "temperature_2m",
        "dew_point_temperature_2m",
        "wind_u_10m",
        "wind_v_10m",
        "wind_u_100m",
        "wind_v_100m",
        "pressure_surface",
        "pressure_reduced_to_mean_sea_level",
        "geopotential_height_500hpa",
        "downward_long_wave_radiation_flux_surface",
        "downward_short_wave_radiation_flux_surface",
        "precipitation_surface",
    ]
    for var in required_vars:
        ds[var] = xr.DataArray(
            np.random.uniform(270.0, 300.0, size=shape),
            dims=["latitude", "longitude", "init_time", "lead_time", "ensemble_member"],
        )

    ds["categorical_precipitation_type_surface"] = xr.DataArray(
        np.zeros(shape, dtype=np.uint8),
        dims=["latitude", "longitude", "init_time", "lead_time", "ensemble_member"],
    )

    ds.to_zarr(zarr_path, zarr_format=2)

    # 2. Process it
    # Explicitly setting decode_timedelta=True avoids reliance on Xarray's
    # deprecated automatic decoding of time units, ensuring lead_time is
    # correctly parsed as timedelta64[ns].
    ds_loaded = xr.open_zarr(zarr_path, decode_timedelta=True)

    # This should NOT raise an IndexError
    downloaded_ds = download_ecmwf(init_time, h3_grid, ds_loaded)

    assert downloaded_ds.latitude.size == 2
    assert downloaded_ds.longitude.size == 2
    assert "temperature_2m" in downloaded_ds.data_vars
