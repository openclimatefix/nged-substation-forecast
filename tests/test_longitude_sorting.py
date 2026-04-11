import pytest
import xarray as xr
import numpy as np
import polars as pl
import patito as pt
from contracts.data_schemas import H3GridWeights
from dynamical_data.processing import download_ecmwf
from typing import cast


@pytest.fixture
def h3_grid() -> pt.DataFrame[H3GridWeights]:
    return cast(
        pt.DataFrame[H3GridWeights],
        pl.DataFrame(
            {
                "h3_index": [123456789, 123456790],
                "nwp_lat": [56.0, 57.0],
                "nwp_lng": [0.0, -5.0],
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


def test_longitude_descending_slicing(tmp_path, h3_grid):
    # Create a dataset with descending longitude
    init_time = np.datetime64("2026-03-01T00:00:00")

    # Descending longitude: [0, -5]
    ds = xr.Dataset(
        {
            "temperature_2m": (
                ["latitude", "longitude", "init_time", "lead_time", "ensemble_member"],
                np.random.rand(2, 2, 1, 1, 1),
            ),
            "latitude": (["latitude"], [56.0, 57.0]),
            "longitude": (["longitude"], [0.0, -5.0]),
            "init_time": (["init_time"], [init_time]),
            "lead_time": (["lead_time"], [0.0]),
            "ensemble_member": (["ensemble_member"], [0]),
        }
    )
    # Add other required variables
    for var in [
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
    ]:
        ds[var] = ds["temperature_2m"]

    # This should not fail or return empty
    downloaded_ds = download_ecmwf(init_time, h3_grid, ds)
    assert downloaded_ds.longitude.size > 0
