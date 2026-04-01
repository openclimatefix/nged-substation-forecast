"""Script to create a production-like Zarr sample for testing ECMWF ingestion.

This script creates a small, deterministic Zarr dataset with the EXACT structure
expected by production `download_ecmwf` function, including:

1. **Proper latitude/longitude grid coordinates** - Enables spatial filtering via
   `ds.sel(latitude=..., longitude=...)` like production code.
2. **Multiple forecast steps** (lead_time dimension) - Tests temporal merging logic.
3. **Multiple ensemble members** - Tests ensemble handling.
4. **Minimal GB grid** - Covers a small region of Scotland (Edinburgh area) with
   ~0.25 degree resolution (like production).
5. **Deterministic feature values** - Enables mathematical verification of merges.

CRITICAL: The production `download_ecmwf` function expects Zarr datasets with
latitude/longitude dimensions for spatial slicing. A test Zarr without these
dimensions raises errors.
"""

from datetime import datetime

import numpy as np
import xarray as xr


def create_production_like_ecmwf_zarr(zarr_path: str) -> None:
    """Create a production-like Zarr dataset for testing ECMWF ingestion.

    Args:
        zarr_path: Path where the Zarr store should be created.
    """
    import json
    import os

    os.makedirs(zarr_path, exist_ok=True)

    # Create small lat/lon grid over Edinburgh, Scotland area
    lats = np.array([55.8, 55.9, 56.0], dtype=np.float32)  # 3 latitude points
    lons = np.array([-3.4, -3.2, -3.0], dtype=np.float32)  # 3 longitude points

    # Create init times and forecast parameters
    init_times = [datetime(2026, 3, 1, 0, 0, 0)]
    lead_times = np.array([0.5, 1.0], dtype=np.float32)
    ensemble_members = np.array([0, 1], dtype=np.uint8)

    # Create deterministic base values
    np.random.seed(42)
    temperature_base = 280.0
    wind_speed_base = 5.0
    pressure_base = 1015.0

    # Create 2D grids and add spatial variation
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    temperature_2d = temperature_base + lat_grid * 2.0 + lon_grid * 3.0
    wind_speed_10m_2d = wind_speed_base + np.abs(lat_grid * 0.5 + lon_grid * 0.3) + 0.1
    pressure_2d = pressure_base + lat_grid * 0.5 + lon_grid

    # Helper to expand 2D array to 5D (lat, lon, init, lead, ensemble)
    def expand_2d_to_5d(arr_2d: np.ndarray) -> np.ndarray:
        arr = arr_2d[:, :, np.newaxis]
        arr_lead = np.stack([arr.copy(), arr.copy() + np.random.normal(0, 0.5, arr.shape)], axis=2)
        arr_ens = np.stack(
            [arr_lead.copy(), arr_lead.copy() + np.random.normal(0, 0.3, arr_lead.shape)], axis=2
        )
        return arr_ens

    # Create all 5D data arrays
    temperature_2m_5d = expand_2d_to_5d(temperature_2d)
    dew_point_2m_5d = expand_2d_to_5d(temperature_2d - 5.0)
    wind_speed_10m_5d = expand_2d_to_5d(wind_speed_10m_2d)
    wind_dir_10m_5d = expand_2d_to_5d(
        ((180.0 + lat_grid * 10.0 + lon_grid * 15.0) % 360.0).astype(float)
    )
    wind_speed_100m_5d = expand_2d_to_5d(wind_speed_10m_2d * 1.1)
    wind_dir_100m_5d = wind_dir_10m_5d.copy()
    pressure_surface_5d = expand_2d_to_5d(pressure_2d)

    # Precipitation and radiation: NaN for lead_time=0, valid for lead_time=1
    precip_5d = np.full((3, 3, 1, 2, 2), np.nan, dtype=float)
    precip_5d[:, :, 0, 1, :] = np.random.uniform(0.1, 0.5, (3, 3, 2))

    sw_rad_5d = np.full((3, 3, 1, 2, 2), np.nan, dtype=float)
    sw_rad_5d[:, :, 0, 1, :] = np.random.uniform(50, 100, (3, 3, 2))

    lw_rad_5d = np.full((3, 3, 1, 2, 2), np.nan, dtype=float)
    lw_rad_5d[:, :, 0, 1, :] = np.random.uniform(150, 200, (3, 3, 2))

    # Categorical precipitation type
    precip_type_2d = np.zeros((3, 3), dtype=np.uint8)
    precip_type_2d[precip_5d[:, :, 0, 1, 0] > 0.2] = 1
    precip_type_5d = np.broadcast_to(precip_type_2d[..., None, None], (3, 3, 1, 2, 2)).astype(
        np.uint8
    )

    # Geopotential and pressure at reduced MSL
    geopotential_5d = expand_2d_to_5d(pressure_2d * 10.0)
    pressure_reduced_msl_5d = pressure_surface_5d.copy()

    # Create Xarray Dataset with production-like structure
    ds = xr.Dataset(
        {
            "temperature_2m": (
                ["latitude", "longitude", "init_time", "lead_time", "ensemble_member"],
                temperature_2m_5d,
            ),
            "dew_point_temperature_2m": (
                ["latitude", "longitude", "init_time", "lead_time", "ensemble_member"],
                dew_point_2m_5d,
            ),
            "wind_speed_10m": (
                ["latitude", "longitude", "init_time", "lead_time", "ensemble_member"],
                wind_speed_10m_5d,
            ),
            "wind_direction_10m": (
                ["latitude", "longitude", "init_time", "lead_time", "ensemble_member"],
                wind_dir_10m_5d,
            ),
            "wind_speed_100m": (
                ["latitude", "longitude", "init_time", "lead_time", "ensemble_member"],
                wind_speed_100m_5d,
            ),
            "wind_direction_100m": (
                ["latitude", "longitude", "init_time", "lead_time", "ensemble_member"],
                wind_dir_100m_5d,
            ),
            "pressure_surface": (
                ["latitude", "longitude", "init_time", "lead_time", "ensemble_member"],
                pressure_surface_5d,
            ),
            "precipitation_surface": (
                ["latitude", "longitude", "init_time", "lead_time", "ensemble_member"],
                precip_5d,
            ),
            "downward_short_wave_radiation_flux_surface": (
                ["latitude", "longitude", "init_time", "lead_time", "ensemble_member"],
                sw_rad_5d,
            ),
            "downward_long_wave_radiation_flux_surface": (
                ["latitude", "longitude", "init_time", "lead_time", "ensemble_member"],
                lw_rad_5d,
            ),
            "categorical_precipitation_type_surface": (
                ["latitude", "longitude", "init_time", "lead_time", "ensemble_member"],
                precip_type_5d,
            ),
            "geopotential_height_500hpa": (
                ["latitude", "longitude", "init_time", "lead_time", "ensemble_member"],
                geopotential_5d,
            ),
            "pressure_reduced_to_mean_sea_level": (
                ["latitude", "longitude", "init_time", "lead_time", "ensemble_member"],
                pressure_reduced_msl_5d,
            ),
        },
        coords={
            "latitude": lats,
            "longitude": lons,
            "init_time": init_times,
            "lead_time": lead_times,
            "ensemble_member": ensemble_members,
        },
    )

    # Write to Zarr
    ds.to_zarr(zarr_path, mode="w", zarr_format=2)

    # Write metadata
    metadata = {
        "test_zarr_info": {
            "init_time": init_times[0].isoformat(),
            "n_latitude_points": len(lats),
            "n_longitude_points": len(lons),
            "lead_times": lead_times.tolist(),
            "ensemble_members": ensemble_members.tolist(),
            "lat_range": [float(lats.min()), float(lats.max())],
            "lon_range": [float(lons.min()), float(lons.max())],
            "created_at": datetime.now().isoformat(),
        }
    }

    with open(os.path.join(zarr_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Created production-like ECMWF test Zarr at: {zarr_path}")
    print(
        f"  Dimensions: latitude ({len(lats)}), longitude ({len(lons)}), init_time (1), lead_time ({len(lead_times)}), ensemble_member ({len(ensemble_members)})"
    )
    print(f"  Lat range: {lats.min():.2f} to {lats.max():.2f}")
    print(f"  Lon range: {lons.min():.2f} to {lons.max():.2f}")
    print(f"  Lead times: {lead_times}")
    print(f"  Ensemble members: {ensemble_members}")
    print(f"  Data variables: {list(ds.data_vars.keys())}")


def create_broken_zarr_samples(broken_dir: str) -> None:
    """Create broken Zarr samples for robustness testing (production-like structure).

    Args:
        broken_dir: Directory to write broken Zarr stores.
    """
    import json
    import os

    os.makedirs(broken_dir, exist_ok=True)
    np.random.seed(42)

    lats = np.array([55.8, 55.9, 56.0], dtype=np.float32)
    lons = np.array([-3.4, -3.2, -3.0], dtype=np.float32)
    init_times = [datetime(2026, 3, 1, 0, 0, 0)]
    lead_times = np.array([0.5, 1.0], dtype=np.float32)
    ensemble_members = np.array([0, 1], dtype=np.uint8)

    def make_5d(arr_2d):
        arr = arr_2d[:, :, np.newaxis]
        arr_lead = np.stack([arr.copy(), arr.copy() + 1.0], axis=2)
        arr_ens = np.stack([arr_lead.copy(), arr_lead.copy() + 0.5], axis=2)
        return arr_ens

    # 1. Empty Zarr - zero spatial coverage
    empty_ds = xr.Dataset(
        {
            "temperature_2m": (
                ["latitude", "longitude", "init_time", "lead_time", "ensemble_member"],
                np.array([]).reshape(0, 0, 1, 2, 2),
            )
        },
        coords={
            "latitude": np.array([], dtype=np.float32),
            "longitude": np.array([], dtype=np.float32),
            "init_time": init_times,
            "lead_time": lead_times,
            "ensemble_member": ensemble_members,
        },
    )
    empty_ds.to_zarr(os.path.join(broken_dir, "empty.zarr"), mode="w")

    # 2. Missing temperature_2m variable
    missing_var_ds = xr.Dataset(
        {
            "dew_point_temperature_2m": (
                ["latitude", "longitude", "init_time", "lead_time", "ensemble_member"],
                make_5d(np.random.rand(3, 3)),
            )
        },
        coords={
            "latitude": lats,
            "longitude": lons,
            "init_time": init_times,
            "lead_time": lead_times,
            "ensemble_member": ensemble_members,
        },
    )
    missing_var_ds.to_zarr(os.path.join(broken_dir, "missing_var.zarr"), mode="w")

    # 3. Invalid lead_time type (string instead of float)
    wrong_lead_ds = xr.Dataset(
        {
            "temperature_2m": (
                ["latitude", "longitude", "init_time", "lead_time", "ensemble_member"],
                make_5d(np.random.rand(3, 3)),
            )
        },
        coords={
            "latitude": lats,
            "longitude": lons,
            "init_time": init_times,
            "lead_time": ["invalid", "also_invalid"],
            "ensemble_member": ensemble_members,
        },
    )
    wrong_lead_ds.to_zarr(os.path.join(broken_dir, "invalid_lead_type.zarr"), mode="w")

    metadata = {
        "broken_zarr_samples": {
            "empty.zarr": "Zero spatial coverage (0x0 grid)",
            "missing_var.zarr": "Missing temperature_2m variable",
            "invalid_lead_type.zarr": "lead_time as string instead of float",
            "created_at": datetime.now().isoformat(),
        }
    }

    with open(os.path.join(broken_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Created broken Zarr samples in: {broken_dir}")


def get_script_paths() -> tuple[str, str]:
    """Get the default paths for test Zarr files."""
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    main_zarr_path = os.path.join(project_root, "example_data", "ecmwf_production_like.zarr")
    broken_dir = os.path.join(project_root, "example_data", "broken_ecmwf_zarr")

    return main_zarr_path, broken_dir


def main() -> None:
    """Create production-like test Zarr samples for NWP ingestion testing."""
    main_zarr_path, broken_dir = get_script_paths()
    create_production_like_ecmwf_zarr(main_zarr_path)
    create_broken_zarr_samples(broken_dir)

    print("\nCreated production-like test Zarr:")
    print(f"  Path: {main_zarr_path}")


if __name__ == "__main__":
    main()
