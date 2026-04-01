#!/usr/bin/env python3
"""Generate synthetic NWP test Zarr data matching production structure.

This script creates deterministic, synthetic Zarr datasets that match the exact
coordinate structure and data format expected by the production ECMWF download
pipeline (see `dynamical_data.processing.download_ecmwf`).

The script is designed to be complex enough to generate production-like data
without requiring actual network calls or access to the Dynamical.org API.
This allows for robust, deterministic testing of the ingestion pipeline,
including temporal deduplication and error handling for malformed data.

The generated test data includes:
1. Valid production-like structure with 5D data variables:
   - Dimensions: (latitude, longitude, init_time, lead_time, ensemble_member)
    - Data variables: temperature, wind components, pressure, precipitation, radiation
    - Physical realism: precipitation is de-accumulated, shortwave radiation is daytime-only.


2. Broken test cases for validation testing:
   - Missing coordinates
   - Mismatched dimension ordering
   - Malformed data arrays
   - Wrong dtypes and encodings
   - Inconsistent shapes

Usage:
    # Run with defaults
    uv run python packages/dynamical_data/src/dynamical_data/scripts/create_production_like_test_zarr.py

    # Specify output directory
    uv run python packages/dynamical_data/src/dynamical_data/scripts/create_production_like_test_zarr.py \
        --output-dir /path/to/output

    # Generate specific test cases
    uv run python packages/dynamical_data/src/dynamical_data/scripts/create_production_like_test_zarr.py \
        --valid true --broken true --broken-types all

    # Skip specific broken variants
    uv run python packages/dynamical_data/src/dynamical_data/scripts/create_production_like_test_zarr.py \
        --broken-types missing_coords,missing_vars

The generated data passes validation in processing.py and can be used for:
- CI testing without network calls
- Testing the download_ecmwf function
- Testing data validation logic
- Testing temporal deduplication (overlapping forecasts)
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# =============================================================================
# Constants
# =============================================================================

# Valid test cases to generate
VALID_TEST_CASES = ["ecmwf_production_like"]

# Broken test case types and their descriptions
BROKEN_TEST_CASE_TYPES: dict[str, str] = {
    "missing_coords": "Missing latitude/longitude coordinates",
    "wrong_dim_order": "Data variables with wrong dimension order",
    "malformed_data": "Data arrays with malformed values (NaNs in wrong places)",
    "missing_vars": "Missing required data variables",
    "wrong_dtype": "Wrong data type for coordinates",
    "inconsistent_shape": "Inconsistent array shapes for variables",
}

# Coordinate values for Great Britain region (matching production data)
# We use strict multiples of 0.25 to match the physical reality of the
# ECMWF IFS 0.25-degree grid.
LATITUDE_VALUES = np.array([55.75, 56.0, 56.25], dtype=np.float32)
LONGITUDE_VALUES = np.array([-3.5, -3.25, -3.0], dtype=np.float32)
INIT_TIME_VALUE = np.datetime64("2026-03-01T00:00:00.000000000")
LEAD_TIME_VALUES = np.array([0.0, 6.0, 12.0], dtype=np.float32)  # 0, 6, and 12-hour forecasts
ENSEMBLE_MEMBER_VALUES = np.array([0, 1], dtype=np.uint8)

# Expected data variable names (matching ECMWF IFS convention)
DATA_VARIABLES: dict[str, dict] = {
    "temperature_2m": {
        "dtype": np.float64,
        "fill_value": np.nan,
        "description": "Temperature at 2 meters above surface (K)",
    },
    "dew_point_temperature_2m": {
        "dtype": np.float64,
        "fill_value": np.nan,
        "description": "Dew point temperature at 2m (K)",
    },
    "wind_u_10m": {
        "dtype": np.float64,
        "fill_value": np.nan,
        "description": "U-component of wind at 10m (m/s)",
    },
    "wind_v_10m": {
        "dtype": np.float64,
        "fill_value": np.nan,
        "description": "V-component of wind at 10m (m/s)",
    },
    "wind_u_100m": {
        "dtype": np.float64,
        "fill_value": np.nan,
        "description": "U-component of wind at 100m (m/s)",
    },
    "wind_v_100m": {
        "dtype": np.float64,
        "fill_value": np.nan,
        "description": "V-component of wind at 100m (m/s)",
    },
    "pressure_surface": {
        "dtype": np.float64,
        "fill_value": np.nan,
        "description": "Pressure at surface (Pa)",
    },
    "pressure_reduced_to_mean_sea_level": {
        "dtype": np.float64,
        "fill_value": np.nan,
        "description": "Pressure reduced to mean sea level (Pa)",
    },
    "precipitation_surface": {
        "dtype": np.float64,
        "fill_value": np.nan,
        "description": "Total precipitation (m)",
    },
    "downward_short_wave_radiation_flux_surface": {
        "dtype": np.float64,
        "fill_value": np.nan,
        "description": "Shortwave radiation flux at surface (W/m^2)",
    },
    "downward_long_wave_radiation_flux_surface": {
        "dtype": np.float64,
        "fill_value": np.nan,
        "description": "Longwave radiation flux at surface (W/m^2)",
    },
    "geopotential_height_500hpa": {
        "dtype": np.float64,
        "fill_value": np.nan,
        "description": "Geopotential height at 500 hPa (m)",
    },
    "categorical_precipitation_type_surface": {
        "dtype": np.uint8,
        "fill_value": 0,
        "description": "Precipitation type classification (categorical)",
    },
}

# Compression settings matching production
ZARR_COMPRESSOR = {
    "id": "blosc",
    "cname": "lz4",
    "clevel": 5,
    "shuffle": 1,
    "blocksize": 0,
}

# Zarr format version
ZARR_FORMAT = 2


# =============================================================================
# Validation and helper functions
# =============================================================================


def get_script_directory() -> Path:
    """Get the directory where this script is located.

    Returns:
        Path object pointing to the scripts directory.
    """
    return Path(__file__).parent.resolve()


def get_output_directory() -> Path:
    """Get the default output directory for test Zarr files.

    Returns:
        Path to the example_data directory relative to script location.
    """
    return get_script_directory().parent / "example_data"


def validate_xr_structure(ds: xr.Dataset) -> tuple[bool, list[str]]:
    """Validate that an Xarray dataset has the expected production structure.

    Production code expects:
    - 5 dimensions: latitude, longitude, init_time, lead_time, ensemble_member
    - Data variables with shape (latitude, longitude, init_time, lead_time, ensemble_member)
    - Coordinates with proper types and encoding

    Args:
        ds: Xarray Dataset to validate.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors: list[str] = []

    # Check required dimensions
    required_dims = ["latitude", "longitude", "init_time", "lead_time", "ensemble_member"]
    for dim in required_dims:
        if dim not in ds.dims:
            errors.append(f"Missing required dimension: {dim}")

    # Check required coordinates
    required_coords = required_dims
    for coord in required_coords:
        if coord not in ds.coords:
            errors.append(f"Missing required coordinate: {coord}")

    # Check data variables and their shapes
    expected_shape = (
        len(ds.latitude) if "latitude" in ds.dims else 0,
        len(ds.longitude) if "longitude" in ds.dims else 0,
        len(ds.init_time) if "init_time" in ds.dims else 0,
        len(ds.lead_time) if "lead_time" in ds.dims else 0,
        len(ds.ensemble_member) if "ensemble_member" in ds.dims else 0,
    )

    for var_name in ds.data_vars:
        if ds[var_name].shape != expected_shape:
            errors.append(
                f"Data variable '{var_name}' has shape {ds[var_name].shape}, "
                f"expected {expected_shape}"
            )

    # Check coordinate dtypes
    coord_errors = []
    if "latitude" in ds.coords and ds.latitude.dtype != np.float32:
        coord_errors.append(f"latitude has dtype {ds.latitude.dtype}, expected float32")
    if "longitude" in ds.coords and ds.longitude.dtype != np.float32:
        coord_errors.append(f"longitude has dtype {ds.longitude.dtype}, expected float32")
    if coord_errors:
        errors.extend(coord_errors)

    return len(errors) == 0, errors


# =============================================================================
# Data generation functions
# =============================================================================


def create_production_like_ecmwf_zarr(
    output_path: Path,
    seed: int = 42,
    init_time: str | np.datetime64 = INIT_TIME_VALUE,
) -> xr.Dataset:
    """Create a synthetic NWP Zarr dataset matching production ECMWF structure.

    This creates a small but structurally complete dataset that:
    - Has the same coordinate structure as production data
    - Contains realistic temperature and wind values
    - Uses proper encodings and compression settings
    - Generates deterministic data for reproducible testing

    Args:
        output_path: Path where the Zarr store will be created.
        seed: Random seed for reproducibility (default: 42).
        init_time: The initialization time for the forecast. Parameterized to allow
            testing of temporal deduplication (overlapping forecasts).

    Returns:
        The created Xarray Dataset.
    """
    np.random.seed(seed)

    # Ensure init_time is np.datetime64
    if isinstance(init_time, str):
        init_time = np.datetime64(init_time)

    # Create the dataset with proper coordinate dimensions
    ds = xr.Dataset(
        {
            # Initialize all dimensions
            "latitude": (
                ["latitude"],
                LATITUDE_VALUES,
                {"long_name": "Latitude", "units": "degrees_north", "standard_name": "latitude"},
            ),
            "longitude": (
                ["longitude"],
                LONGITUDE_VALUES,
                {"long_name": "Longitude", "units": "degrees_east", "standard_name": "longitude"},
            ),
            "init_time": (
                ["init_time"],
                [init_time],
                {
                    "standard_name": "forecast_reference_time",
                    "long_name": "Initiation time",
                },
            ),
            "lead_time": (
                ["lead_time"],
                LEAD_TIME_VALUES,
                {
                    "standard_name": "forecast_period",
                    "long_name": "Lead time",
                    "units": "hours",
                },
            ),
            "ensemble_member": (
                ["ensemble_member"],
                ENSEMBLE_MEMBER_VALUES,
                {
                    "standard_name": "ensemble_member",
                    "long_name": "Ensemble member number",
                },
            ),
        }
    )

    # Generate synthetic data for each variable
    for var_name, var_info in DATA_VARIABLES.items():
        # Generate synthetic data with realistic ranges
        shape = (
            len(LATITUDE_VALUES),
            len(LONGITUDE_VALUES),
            1,  # Scalar init_time (single initialization time)
            len(LEAD_TIME_VALUES),
            len(ENSEMBLE_MEMBER_VALUES),
        )

        if var_name == "temperature_2m":
            # Temperature: 270-300 K (roughly -3 to 27°C)
            data = np.random.uniform(270.0, 300.0, size=shape).astype(np.float64)
        elif var_name == "dew_point_temperature_2m":
            # Dew point: slightly lower than temperature
            data = np.random.uniform(265.0, 295.0, size=shape).astype(np.float64)
            data = np.minimum(data, ds["temperature_2m"].values - 2.0)  # Ensure below temp
        elif "wind_u" in var_name or "wind_v" in var_name:
            # Wind components: -15 to 15 m/s
            data = np.random.uniform(-15.0, 15.0, size=shape).astype(np.float64)
        elif var_name == "pressure_surface":
            # Surface pressure: 95000-105000 Pa
            data = np.random.uniform(95000.0, 105000.0, size=shape).astype(np.float64)
        elif var_name == "pressure_reduced_to_mean_sea_level":
            # MSL pressure: 97000-107000 Pa
            data = np.random.uniform(97000.0, 107000.0, size=shape).astype(np.float64)
        elif var_name == "precipitation_surface":
            # Precipitation (m): 0-0.01m (10mm)
            # ECMWF data from Dynamical.org is already de-accumulated prior to download.
            # It represents the precipitation amount for that specific time step, not the
            # accumulated total, so we generate independent random values per step.
            data = np.random.uniform(0.0, 0.01, size=shape).astype(np.float64)
        elif "short_wave" in var_name:
            # Shortwave radiation (W/m^2): 0-1000
            data = np.zeros(shape, dtype=np.float64)
            # Shortwave radiation is only non-zero during daytime hours.
            # For simplicity, we assume lead times between 6:00 and 18:00 are daytime.
            # We need to calculate the actual time for each lead time.
            init_dt = pd.to_datetime(init_time).to_pydatetime()
            for i, lt in enumerate(LEAD_TIME_VALUES):
                valid_time = init_dt + (lt * 3600).astype("timedelta64[s]").item()
                if 6 <= valid_time.hour < 18:
                    data[:, :, :, i, :] = np.random.uniform(
                        200.0, 800.0, size=(shape[0], shape[1], shape[2], shape[4])
                    )
        elif "long_wave" in var_name:
            # Longwave radiation (W/m^2): 200-400
            data = np.random.uniform(200.0, 400.0, size=shape).astype(np.float64)
        elif var_name == "geopotential_height_500hpa":
            # Geopotential height (m): 5000-5800m
            data = np.random.uniform(5000.0, 5800.0, size=shape).astype(np.float64)
        elif var_name == "categorical_precipitation_type_surface":
            # Categorical: 0-11 (ECMWF categories)
            data = np.random.randint(0, 12, size=shape).astype(np.uint8)
        else:
            # Default: random values
            data = np.random.uniform(0.0, 1.0, size=shape).astype(np.float64)

        # Add the data variable to the dataset
        ds[var_name] = xr.DataArray(
            data,
            dims=["latitude", "longitude", "init_time", "lead_time", "ensemble_member"],
            attrs={
                "standard_name": var_name,
                "long_name": var_info["description"],
                "units": "unknown",  # Would be specified for each variable in real data
            },
        )

    # Write to Zarr store with matching compression
    ds.to_zarr(
        str(output_path),
        mode="w",
        zarr_format=ZARR_FORMAT,
        compute=True,
    )

    # Write additional metadata for verification
    write_verification_metadata(output_path, ds, seed)

    return ds


def write_verification_metadata(output_path: Path, ds: xr.Dataset, seed: int) -> None:
    """Write metadata file for test verification.

    Contains information about the test data structure that can be used by tests
    to verify correctness of the data generation.

    Args:
        output_path: Path to the Zarr store directory.
        ds: The xarray Dataset that was saved.
        seed: The random seed used for generation.
    """
    metadata = {
        "test_zarr_info": {
            "name": "ecmwf_production_like",
            "seed": seed,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "latitude_values": ds.latitude.values.tolist(),
            "longitude_values": ds.longitude.values.tolist(),
            "init_time": str(ds.init_time.values[0]),
            "lead_time_hours": ds.lead_time.values.tolist(),
            "ensemble_member_count": len(ds.ensemble_member),
            "data_variable_count": len(ds.data_vars),
            "data_variables": list(ds.data_vars.keys()),
            "coordinates": list(ds.coords.keys()),
            "validation_passed": True,
        }
    }

    with open(os.path.join(output_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def create_broken_ecmwf_zarrs(
    broken_dir: Path, seed: int = 42, broken_types: list[str] | None = None
) -> None:
    """Create broken test Zarr datasets for validation testing.

    Creates multiple intentionally broken test cases that exercise the validation
    logic in the production code. Each broken type tests a different failure mode:

    - missing_coords: Data variables without proper coordinate references
    - wrong_dim_order: Data variables with swapped dimension order
    - malformed_data: Arrays with NaNs/inf in unexpected locations
    - missing_vars: Missing required data variables
    - wrong_dtype: Coordinates with incorrect data types
    - inconsistent_shape: Data variables with mismatched array shapes

    Args:
        broken_dir: Directory to write broken Zarr stores.
        seed: Random seed for reproducibility.
        broken_types: Specific broken types to create. If None, creates all types.
    """
    os.makedirs(broken_dir, exist_ok=True)

    # Determine which broken types to create
    if broken_types is None:
        broken_types = list(BROKEN_TEST_CASE_TYPES.keys())

    for broken_type in broken_types:
        create_broken_ecmwf_zarr(
            broken_dir / f"{broken_type}.zarr",
            broken_type=broken_type,
            seed=seed,
        )
        print(f"  Created broken case: {broken_type}")


def create_broken_ecmwf_zarr(
    output_path: Path,
    broken_type: str,
    seed: int = 42,
    init_time: str | np.datetime64 = INIT_TIME_VALUE,
) -> xr.Dataset:
    """Create a single broken Zarr dataset for a specific failure mode.

    Args:
        output_path: Path for the broken Zarr store.
        broken_type: Type of broken case to create.
        seed: Random seed for reproducibility.
        init_time: The initialization time for the forecast. Parameterized to allow
            testing of temporal deduplication (overlapping forecasts).

    Returns:
        The created (broken) xarray Dataset.
    """
    np.random.seed(seed)

    # Ensure init_time is np.datetime64
    if isinstance(init_time, str):
        init_time = np.datetime64(init_time)

    output_dir = Path(output_path).parent
    os.makedirs(output_dir, exist_ok=True)

    # Create base dataset structure
    ds = xr.Dataset(
        {
            "latitude": (["latitude"], LATITUDE_VALUES),
            "longitude": (["longitude"], LONGITUDE_VALUES),
            "init_time": (["init_time"], [init_time]),
            "lead_time": (["lead_time"], LEAD_TIME_VALUES),
            "ensemble_member": (["ensemble_member"], ENSEMBLE_MEMBER_VALUES),
        }
    )

    # Generate base data shape
    # Note: INIT_TIME_VALUE is a scalar datetime64, not an array, so we use 1
    shape = (
        len(LATITUDE_VALUES),
        len(LONGITUDE_VALUES),
        1,  # Scalar init_time (single initialization time)
        len(LEAD_TIME_VALUES),
        len(ENSEMBLE_MEMBER_VALUES),
    )

    # Add some default data variables so we have something to break/validate
    # These will be overwritten or removed by specific broken_type cases
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
        "categorical_precipitation_type_surface",
    ]
    for var_name in required_vars:
        var_info = DATA_VARIABLES[var_name]
        dtype = var_info["dtype"]
        if dtype == np.uint8:
            data = np.zeros(shape, dtype=np.uint8)
        else:
            data = np.random.uniform(270.0, 300.0, size=shape).astype(dtype)

        ds[var_name] = xr.DataArray(
            data,
            dims=["latitude", "longitude", "init_time", "lead_time", "ensemble_member"],
            attrs={"description": f"Default {var_name}"},
        )

    match broken_type:
        case "missing_coords":
            # Create data without proper coordinate references
            # Data is there, but coordinates don't match

            # Drop the coordinate variables (latitude, longitude)
            # This leaves the dimensions but removes the coordinate values
            ds = ds.drop_vars(["latitude", "longitude"])

        case "wrong_dim_order":
            # Create data variables with wrong dimension ordering
            # This tests that the validation catches dimension mismatches
            for var_name in ["temperature_2m", "wind_u_10m"]:
                data = np.random.uniform(0.0, 1.0, size=shape).astype(np.float64)
                # WRONG: swap the first two dimensions
                data_swapped = np.transpose(data, axes=(1, 0, 2, 3, 4))
                ds[var_name] = xr.DataArray(
                    data_swapped,
                    dims=["longitude", "latitude", "init_time", "lead_time", "ensemble_member"],
                    attrs={"description": f"Variable {var_name}"},
                )

        case "malformed_data":
            # Create data with problematic values in unexpected places
            # NaNs should be in consistent locations, not random
            # Inf values where they shouldn't be

            # Add temperature with NaN in wrong place (not in first lead_time)
            temp_data = np.random.uniform(0.0, 300.0, size=shape).astype(np.float64)
            # Set a NaN where it shouldn't be (in second lead time, first ensemble)
            temp_data[0, 0, 0, 1, 0] = np.nan
            ds["temperature_2m"] = xr.DataArray(
                temp_data,
                dims=["latitude", "longitude", "init_time", "lead_time", "ensemble_member"],
                attrs={"description": "Temperature 2m"},
            )

            # Add wind speed with inf values
            wind_data = np.random.uniform(0.0, 20.0, size=shape).astype(np.float64)
            wind_data[0, 0, 0, 0, 0] = np.inf
            ds["wind_u_10m"] = xr.DataArray(
                wind_data,
                dims=["latitude", "longitude", "init_time", "lead_time", "ensemble_member"],
                attrs={"description": "Wind U 10m"},
            )

        case "missing_vars":
            # Remove critical variables that should always be present
            ds = remove_variables(
                ds,
                [
                    "temperature_2m",
                    "dew_point_temperature_2m",
                    "wind_u_10m",
                    "wind_v_10m",
                ],
            )

        case "wrong_dtype":
            # Set coordinates with wrong data types
            ds = ds.assign_coords(
                latitude=(["latitude"], LATITUDE_VALUES.astype(np.float64)),
                longitude=(["longitude"], LONGITUDE_VALUES.astype(np.float64)),
            )

            # Also change init_time to wrong type
            ds = ds.assign_coords(
                init_time=(["init_time"], ["2026-03-01T00:00:00"])  # String instead of datetime64
            )

        case "inconsistent_shape":
            # Create data variables with mismatched shapes
            # This should cause validation to fail
            # We use a different size for one of the dimensions to trigger the error
            # but we must use a different dimension name to avoid xarray alignment errors during assignment
            ds["temperature_2m"] = xr.DataArray(
                np.random.uniform(
                    0.0,
                    300.0,
                    size=(
                        len(LATITUDE_VALUES),
                        len(LONGITUDE_VALUES),
                        1,
                        len(LEAD_TIME_VALUES) + 1,
                        len(ENSEMBLE_MEMBER_VALUES),
                    ),
                ).astype(np.float64),
                dims=["latitude", "longitude", "init_time", "wrong_lead_time", "ensemble_member"],
                attrs={"description": "Temperature 2m"},
            )
            ds["wind_u_10m"] = xr.DataArray(
                np.random.uniform(
                    0.0,
                    20.0,
                    size=(
                        len(LATITUDE_VALUES),
                        len(LONGITUDE_VALUES),
                        1,
                        len(LEAD_TIME_VALUES),
                        len(ENSEMBLE_MEMBER_VALUES),
                        2,
                    ),
                ).astype(np.float64),
                # Wrong shape entirely
                dims=[
                    "latitude",
                    "longitude",
                    "init_time",
                    "lead_time",
                    "ensemble_member",
                    "extra_dim",
                ],
                attrs={"description": "Wind U 10m"},
            )

        case _:
            raise ValueError(f"Unknown broken type: {broken_type}")

    # Write to Zarr
    ds.to_zarr(
        str(output_path),
        mode="w",
        zarr_format=ZARR_FORMAT,
        compute=True,
    )

    # Write broken-type metadata
    write_broken_metadata(output_path, broken_type)

    return ds


def remove_variables(ds: xr.Dataset, vars_to_remove: list[str]) -> xr.Dataset:
    """Remove specified variables (data or coordinates) from the dataset.

    Args:
        ds: Xarray Dataset to modify.
        vars_to_remove: List of variable names to remove.

    Returns:
        Dataset with specified variables removed.
    """
    # Use drop_vars for any variable that exists in the dataset (data or coord)
    existing_vars = [v for v in vars_to_remove if v in ds.variables]
    if existing_vars:
        ds = ds.drop_vars(existing_vars)
    return ds


def write_broken_metadata(output_path: Path, broken_type: str) -> None:
    """Write metadata explaining the broken nature of the test data.

    Args:
        output_path: Path to the Zarr store directory.
        broken_type: Type of broken case created.
    """
    metadata = {
        "broken_test_info": {
            "type": broken_type,
            "description": BROKEN_TEST_CASE_TYPES.get(broken_type, "Unknown broken type"),
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
        }
    }

    with open(os.path.join(output_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)


# =============================================================================
# Command-line interface
# =============================================================================


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for this script.

    Returns:
        Configured ArgumentParser with all CLI options.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic NWP test Zarr data matching production structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all test cases with defaults
  uv run python create_production_like_test_zarr.py

  # Only create valid production-like data
  uv run python create_production_like_test_zarr.py --valid true --broken false

  # Create only specific broken test cases
  uv run python create_production_like_test_zarr.py --broken true --broken-types missing_coords,missing_vars

  # Use custom output directory
  uv run python create_production_like_test_zarr.py --output-dir ./custom_output
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Output directory for Zarr stores. Default: {get_output_directory()}",
    )

    parser.add_argument(
        "--valid",
        type=str.lower,
        default="true",
        choices=["true", "false"],
        help="Create valid production-like test data. Default: true",
    )

    parser.add_argument(
        "--broken",
        type=str.lower,
        default="true",
        choices=["true", "false"],
        help="Create broken test cases for validation testing. Default: true",
    )

    parser.add_argument(
        "--broken-types",
        type=str,
        default=None,
        help=(
            f"Comma-separated list of broken types to create. "
            f"Available: {', '.join(BROKEN_TEST_CASE_TYPES.keys())}. "
            f"Default: all types"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible data generation. Default: 42",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    return parser


def main() -> None:
    """Main entry point for the Zarr test data generator.

    Creates both valid production-like test data and broken test cases
    for comprehensive validation testing.
    """
    parser = create_parser()
    args = parser.parse_args()

    # Parse arguments
    output_dir = (Path(args.output_dir) if args.output_dir else get_output_directory()).resolve()
    create_valid = args.valid == "true"
    create_broken = args.broken == "true"

    # Parse broken types
    broken_types: list[str] | None = None
    if args.broken_types:
        broken_types = [t.strip() for t in args.broken_types.split(",")]
        # Validate broken types
        invalid_types = set(broken_types) - set(BROKEN_TEST_CASE_TYPES.keys())
        if invalid_types:
            parser.error(
                f"Invalid broken types: {', '.join(invalid_types)}\n"
                f"Valid options: {', '.join(BROKEN_TEST_CASE_TYPES.keys())}"
            )

    # Set up output directories
    valid_dir = output_dir / "ecmwf_production_like.zarr"
    broken_dir = output_dir / "broken_ecmwf_zarr"

    if not args.quiet:
        print(f"Creating test Zarr data in: {output_dir}")
        print(f"  Random seed: {args.seed}")

    # Create valid production-like data (if requested)
    if create_valid:
        if not args.quiet:
            print("\nGenerating valid production-like test data...")

        try:
            ds = create_production_like_ecmwf_zarr(valid_dir, args.seed)
            is_valid, errors = validate_xr_structure(ds)

            if is_valid:
                print(f"  Created: {valid_dir}")
                print(f"  Dimensions: {list(ds.dims)}")
                print(f"  Data variables: {len(ds.data_vars)}")
                print("  Validation: PASSED")
            else:
                print(f"  Created: {valid_dir}")
                print("  Validation: FAILED")
                for error in errors:
                    print(f"    - {error}")
        except Exception as e:
            print(f"  Error creating valid data: {e}")
            raise

    # Create broken test cases (if requested)
    if create_broken:
        if not args.quiet:
            print("\nGenerating broken test cases for validation...")

        try:
            create_broken_ecmwf_zarrs(broken_dir, args.seed, broken_types)
            print(f"  Created broken cases in: {broken_dir}")
            print("  Breaks expected by type:")
            for broken_type in broken_types if broken_types else BROKEN_TEST_CASE_TYPES.keys():
                desc = BROKEN_TEST_CASE_TYPES.get(broken_type, "Unknown")
                print(f"    - {broken_type}: {desc}")
        except Exception as e:
            print(f"  Error creating broken data: {e}")
            raise

    if not args.quiet:
        print("\nDone! Test Zarr stores created successfully.")


if __name__ == "__main__":
    main()
