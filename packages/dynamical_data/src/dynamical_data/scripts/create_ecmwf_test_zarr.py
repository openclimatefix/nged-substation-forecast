"""Script to create a deterministic Zarr sample for testing ECMWF ingestion.

This script creates a small, deterministic Zarr dataset with known temporal structure
to enable robust, deterministic CI testing for the NWP ingestion pipeline. The test
data specifically includes:

1. **Multiple forecast steps** (step=0 and step=1): This is critical to test the
   temporal merging logic in `download_and_scale_ecmwf` function. Without multiple
   steps, we cannot verify that the pipeline correctly deduplicates timestamps when
   merging forecast horizons.

2. **Duplicate timestamp entries**: Testing deduplication logic to ensure the last
   update is preserved when the same (init_time, valid_time, ensemble_member, h3_index)
   combination appears multiple times.

3. **Malformed data columns**: Tests validation logic to ensure the pipeline fails
   loudly with informative error messages when encountering unexpected column structures.

Why we need a deterministic Zarr sample:
----------------------------------------
- Network calls to the real API are slow, flaky, and introduce non-determinism.
- Using a static sample allows fast, reliable CI runs.
- We can mathematically verify that duplicates are merged correctly.
- We can test edge cases (empty files, missing columns) that are hard to generate
  with a live API.

This script belongs in packages/dynamical_data/scripts/ (NOT exploration_scripts/)
to keep it close to the code it tests and maintain proper package structure.
"""

import json
import os
from datetime import datetime, timedelta

import h3
import numpy as np
import polars as pl
import xarray as xr


def create_test_ecmwf_zarr(zarr_path: str, output_dir: str) -> None:
    """Create a small deterministic Zarr dataset for testing ECMWF ingestion.

    This function creates a minimal Xarray dataset with the structure expected by
    `download_and_scale_ecmwf`, including:

    - Two consecutive forecast steps (step=0 and step=1) to test temporal merging
    - Multiple H3 cells to test spatial filtering
    - Deterministic feature values for reproducibility
    - Deliberate duplicate entries to test deduplication logic

    The resulting Zarr store can be used by tests in `test_nwp_ingestion_robustness.py`
    to verify that:
    1. Multiple forecast steps are merged without duplication
    2. Duplicate entries are correctly deduplicated (last update wins)
    3. Invalid data raises appropriate errors

    Args:
        zarr_path: Path within the data project where the Zarr store should be created.
        output_dir: Directory to write the Zarr output (used for relative paths).
    """
    # Ensure output directory exists
    os.makedirs(zarr_path, exist_ok=True)

    # Create a deterministic H3 index for testing (central UK location)
    # Edinburgh, Scotland - fixed for reproducibility
    test_h3_index = h3.latlng_to_cell(55.9533, -3.1883, 5)

    # Define deterministic time structure
    # init_time: When the forecast was initialized
    init_time = datetime(2026, 3, 1, 0, 0, 0)

    # Step 0 and Step 1 (consecutive forecast horizons)
    # These need to merge without duplication
    steps = [0, 1]

    # Valid times for each step (30-minute granularity as ECWMF uses)
    # step=0 covers 30-minute intervals on the init day
    # step=1 covers the next 30-minute intervals
    valid_times_0 = [
        init_time + timedelta(minutes=30 * i) for i in range(0, 2)
    ]  # Only 2 timestamps for small test
    valid_times_1 = [init_time + timedelta(minutes=30 * i) for i in range(2, 4)]

    # Create synthetic deterministic values
    # Using small integer values cast to uint8 scaling later
    # These mimic the 0-255 scaled representation
    # Deterministic feature arrays use small values that will scale appropriately
    np.random.seed(42)  # For reproducibility

    # Step 0 data - 2 timestamps at Edinburgh
    data_step_0 = {
        "valid_time": valid_times_0 + valid_times_0,  # Include a duplicate for dedup test
        "init_time": [init_time] * len(valid_times_0) * 2,
        "lead_time_hours": [0.5, 1.0, 0.5, 1.0],
        "h3_index": [test_h3_index, test_h3_index] * 2,
        "ensemble_member": [0, 0] * 2,
        # Deterministic feature values (will be scaled)
        "temperature_2m": [280, 282] * 2,  # 7-9°C scaled
        "dew_point_temperature_2m": [275, 277] * 2,  # 2-4°C scaled
        "wind_speed_10m": [300, 310] * 2,  # 2-3 m/s scaled
        "wind_direction_10m": [180, 200] * 2,  # S to SSW
        "wind_speed_100m": [350, 360] * 2,
        "wind_direction_100m": [185, 205] * 2,
        "pressure_surface": [1015, 1014] * 2,
        "precipitation_surface": [50, 60] * 2,
        "downward_short_wave_radiation_flux_surface": [100, 50] * 2,
        "downward_long_wave_radiation_flux_surface": [200, 195] * 2,
    }

    # Step 1 data - 2 more timestamps (next hour)
    data_step_1 = {
        "valid_time": valid_times_1 + valid_times_1,  # Include a duplicate for dedup test
        "init_time": [init_time] * len(valid_times_1) * 2,
        "lead_time_hours": [1.5, 2.0, 1.5, 2.0],
        "h3_index": [test_h3_index, test_h3_index] * 2,
        "ensemble_member": [0, 0] * 2,
        "temperature_2m": [283, 284] * 2,  # 10-11°C scaled
        "dew_point_temperature_2m": [278, 279] * 2,
        "wind_speed_10m": [315, 320] * 2,
        "wind_direction_10m": [210, 215] * 2,
        "wind_speed_100m": [365, 370] * 2,
        "wind_direction_100m": [210, 215] * 2,
        "pressure_surface": [1013, 1012] * 2,
        "precipitation_surface": [40, 30] * 2,
        "downward_short_wave_radiation_flux_surface": [100, 80] * 2,
        "downward_long_wave_radiation_flux_surface": [190, 185] * 2,
    }

    # Combine both steps
    # This tests the merge logic in download_and_scale_ecmwf
    combined_data = {key: data_step_0[key] + data_step_1[key] for key in data_step_0.keys()}

    # Create Polars DataFrame
    df = pl.DataFrame(combined_data)

    # Intentionally add a duplicate entry to test deduplication
    # This entry has the same (valid_time, init_time, h3_index, ensemble_member)
    # but a slightly different temperature value
    duplicate_row = pl.DataFrame(
        {
            "valid_time": [valid_times_0[0]],  # Same timestamp as first row
            "init_time": [init_time],
            "lead_time_hours": [0.5],
            "h3_index": [test_h3_index],
            "ensemble_member": [0],
            # Use the last value (310) as the one that should be preserved after dedup
            "temperature_2m": [310],
            "dew_point_temperature_2m": [300],
            "wind_speed_10m": [350],
            "wind_direction_10m": [210],
            "wind_speed_100m": [400],
            "wind_direction_100m": [215],
            "pressure_surface": [1018],
            "precipitation_surface": [90],
            "downward_short_wave_radiation_flux_surface": [130],
            "downward_long_wave_radiation_flux_surface": [225],
        }
    )

    df = pl.concat([df, duplicate_row])

    # Convert to Xarray Dataset with proper structure
    ds = xr.Dataset(
        {
            "temperature_2m": (["time"], df["temperature_2m"].to_list()),
            "dew_point_temperature_2m": (["time"], df["dew_point_temperature_2m"].to_list()),
            "wind_speed_10m": (["time"], df["wind_speed_10m"].to_list()),
            "wind_direction_10m": (["time"], df["wind_direction_10m"].to_list()),
            "wind_speed_100m": (["time"], df["wind_speed_100m"].to_list()),
            "wind_direction_100m": (["time"], df["wind_direction_100m"].to_list()),
            "pressure_surface": (["time"], df["pressure_surface"].to_list()),
            "precipitation_surface": (["time"], df["precipitation_surface"].to_list()),
            "downward_short_wave_radiation_flux_surface": (
                ["time"],
                df["downward_short_wave_radiation_flux_surface"].to_list(),
            ),
            "downward_long_wave_radiation_flux_surface": (
                ["time"],
                df["downward_long_wave_radiation_flux_surface"].to_list(),
            ),
        },
        coords={
            "time": df["valid_time"].to_list(),
            "init_time": (["time"], df["init_time"].to_list()),
            "lead_time_hours": (["time"], df["lead_time_hours"].to_list()),
            "h3_index": (["time"], df["h3_index"].to_list()),
            "ensemble_member": (["time"], df["ensemble_member"].to_list()),
        },
    )

    # Add H3 coordinate as a dimensionless coord for spatial indexing
    ds = ds.assign_coords(h3_index=(["h3"], [test_h3_index, f"{test_h3_index}2"]))

    # Write to Zarr with compression
    ds.to_zarr(
        zarr_path,
        mode="w",
        zarr_format=2,
    )

    # Write metadata for test verification
    metadata = {
        "test_zarr_info": {
            "init_time": init_time.isoformat(),
            "total_rows": len(df),
            "expected_unique_rows_after_dedup": len(df) - 4,  # Remove 4 duplicates
            "h3_indices": [test_h3_index, f"{test_h3_index}2"],
            "steps_included": steps,
            "created_at": datetime.now().isoformat(),
        }
    }

    with open(os.path.join(zarr_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Created deterministic ECMWF test Zarr at: {zarr_path}")
    print(f"  Total rows (including duplicates): {len(df)}")
    print(f"  Expected unique rows after dedup: {len(df) - 4}")
    print(f"  H3 indices included: {metadata['test_zarr_info']['h3_indices']}")
    print(f"  Forecast steps included: {steps}")
    print("  Metadata saved to metadata.json")


def create_broken_zarr_samples(broken_dir: str) -> None:
    """Create broken Zarr samples for testing robust error handling.

    These samples test the pipeline's ability to fail loudly with informative
    error messages for malformed inputs.

    Args:
        broken_dir: Directory to write broken Zarr stores.
    """
    from datetime import datetime

    os.makedirs(broken_dir, exist_ok=True)

    # Define local test values for broken samples
    init_time = datetime(2026, 3, 1, 0, 0, 0)
    test_h3_index = h3.latlng_to_cell(55.9533, -3.1883, 5)
    valid_times_0 = [init_time + timedelta(minutes=30 * i) for i in range(0, 2)]

    # 1. Empty Zarr (no data variables)
    empty_ds = xr.Dataset(
        coords={
            "valid_time": [],
            "init_time": [],
            "lead_time_hours": [],
            "h3_index": [],
            "ensemble_member": [],
        }
    )
    empty_ds.to_zarr(os.path.join(broken_dir, "empty.zarr"), mode="w")

    # 2. Missing critical variable (temperature_2m)
    missing_var_ds = xr.Dataset(
        {
            "dew_point_temperature_2m": (["time"], [275.0, 276.0]),
            "wind_speed_10m": (["time"], [300.0, 310.0]),
            # Note: temperature_2m is missing
        },
        coords={
            "valid_time": [init_time + timedelta(minutes=30 * i) for i in range(2)],
            "init_time": [init_time] * 2,
            "lead_time_hours": [0.5, 1.0],
            "h3_index": [test_h3_index] * 2,
            "ensemble_member": [0, 0],
        },
    )
    missing_var_ds.to_zarr(os.path.join(broken_dir, "missing_var.zarr"), mode="w")

    # 3. Invalid coordinates (lead_time as string instead of numeric)
    invalid_coords_ds = xr.Dataset(
        {
            "temperature_2m": (["time"], [280.0, 282.0]),
            "dew_point_temperature_2m": (["time"], [275.0, 277.0]),
            "wind_speed_10m": (["time"], [300.0, 310.0]),
            "wind_direction_10m": (["time"], [180.0, 200.0]),
            "wind_speed_100m": (["time"], [350.0, 360.0]),
            "wind_direction_100m": (["time"], [185.0, 205.0]),
            "pressure_surface": (["time"], [1015.0, 1014.0]),
            "precipitation_surface": (["time"], [50.0, 60.0]),
            "downward_short_wave_radiation_flux_surface": (["time"], [100.0, 50.0]),
            "downward_long_wave_radiation_flux_surface": (["time"], [200.0, 195.0]),
        },
        coords={
            "time": valid_times_0,
            "init_time": (["time"], [init_time] * 2),
            "lead_time_hours": (["time"], ["invalid", "also_invalid"]),  # String instead of numeric
            "h3_index": (["time"], [test_h3_index] * 2),
            "ensemble_member": (["time"], [0, 0]),
        },
    )
    invalid_coords_ds.to_zarr(os.path.join(broken_dir, "invalid_coords.zarr"), mode="w")

    print(f"Created broken Zarr samples in: {broken_dir}")
    print("  - empty.zarr: No data variables")
    print("  - missing_var.zarr: Missing temperature_2m")
    print("  - invalid_coords.zarr: Invalid lead_time_hours coordinate type")


def get_script_paths() -> tuple[str, str]:
    """Get the default paths for test Zarr files relative to script location.

    Returns:
        A tuple of (main_zarr_path, broken_dir_path) paths.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Create main test Zarr
    main_zarr_path = os.path.join(project_root, "example_data", "ecmwf_sample.zarr")

    # Create broken Zarr samples for robustness testing
    broken_dir = os.path.join(project_root, "example_data", "broken_ecmwf_zarr")

    return main_zarr_path, broken_dir


def main() -> None:
    """Create test Zarr samples for NWP ingestion testing.

    This function creates deterministic test data including:
    - Main test Zarr with temporal duplicates for deduplication testing
    - Broken Zarr samples for robustness testing (empty, missing vars, invalid coords)
    """
    # Get default paths
    main_zarr_path, broken_dir = get_script_paths()

    # Create main test Zarr
    create_test_ecmwf_zarr(main_zarr_path, main_zarr_path)

    # Create broken Zarr samples for robustness testing
    create_broken_zarr_samples(broken_dir)

    print("\nCreated all test Zarr samples:")
    print(f"  Test sample: {main_zarr_path}")
    print(f"  Broken samples: {broken_dir}/")


if __name__ == "__main__":
    main()
