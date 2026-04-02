import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr
from dynamical_data.processing import process_ecmwf_dataset, download_ecmwf
from contracts.data_schemas import H3GridWeights


def create_mock_ds(lats, lons, init_time, lead_times, ensembles):
    shape = (len(lats), len(lons), 1, len(lead_times), len(ensembles))
    return xr.Dataset(
        data_vars={
            "temperature_2m": (
                ("latitude", "longitude", "init_time", "lead_time", "ensemble_member"),
                np.random.rand(*shape).astype(np.float32),
            ),
            "dew_point_temperature_2m": (
                ("latitude", "longitude", "init_time", "lead_time", "ensemble_member"),
                np.random.rand(*shape).astype(np.float32),
            ),
            "wind_u_10m": (
                ("latitude", "longitude", "init_time", "lead_time", "ensemble_member"),
                np.random.rand(*shape).astype(np.float32),
            ),
            "wind_v_10m": (
                ("latitude", "longitude", "init_time", "lead_time", "ensemble_member"),
                np.random.rand(*shape).astype(np.float32),
            ),
            "wind_u_100m": (
                ("latitude", "longitude", "init_time", "lead_time", "ensemble_member"),
                np.random.rand(*shape).astype(np.float32),
            ),
            "wind_v_100m": (
                ("latitude", "longitude", "init_time", "lead_time", "ensemble_member"),
                np.random.rand(*shape).astype(np.float32),
            ),
            "pressure_surface": (
                ("latitude", "longitude", "init_time", "lead_time", "ensemble_member"),
                np.random.rand(*shape).astype(np.float32),
            ),
            "pressure_reduced_to_mean_sea_level": (
                ("latitude", "longitude", "init_time", "lead_time", "ensemble_member"),
                np.random.rand(*shape).astype(np.float32),
            ),
            "geopotential_height_500hpa": (
                ("latitude", "longitude", "init_time", "lead_time", "ensemble_member"),
                np.random.rand(*shape).astype(np.float32),
            ),
            "downward_long_wave_radiation_flux_surface": (
                ("latitude", "longitude", "init_time", "lead_time", "ensemble_member"),
                np.random.rand(*shape).astype(np.float32),
            ),
            "downward_short_wave_radiation_flux_surface": (
                ("latitude", "longitude", "init_time", "lead_time", "ensemble_member"),
                np.random.rand(*shape).astype(np.float32),
            ),
            "precipitation_surface": (
                ("latitude", "longitude", "init_time", "lead_time", "ensemble_member"),
                np.random.rand(*shape).astype(np.float32),
            ),
            "categorical_precipitation_type_surface": (
                ("latitude", "longitude", "init_time", "lead_time", "ensemble_member"),
                np.zeros(shape, dtype=np.uint8),
            ),
        },
        coords={
            "latitude": lats,
            "longitude": lons,
            "init_time": [init_time.to_datetime64()],
            "lead_time": lead_times,
            "ensemble_member": ensembles,
        },
    )


def test_process_ecmwf_dataset_basic():
    h3_index = 0x85194AD7FFFFFFF
    h3_grid = H3GridWeights.validate(
        pl.DataFrame(
            {
                "h3_index": [h3_index],
                "nwp_lat": [51.5],
                "nwp_lng": [0.0],
                "len": [1],
                "total": [1],
                "proportion": [1.0],
            },
            schema={
                "h3_index": pl.UInt64,
                "nwp_lat": pl.Float64,
                "nwp_lng": pl.Float64,
                "len": pl.UInt32,
                "total": pl.UInt32,
                "proportion": pl.Float64,
            },
        )
    )

    init_time = pd.Timestamp("2024-04-01", tz="UTC")
    ds = create_mock_ds([51.5], [0.0], init_time, [pd.Timedelta(hours=0)], [0])

    result = process_ecmwf_dataset(init_time.to_pydatetime(), ds, h3_grid)
    assert len(result) == 1
    assert result["temperature_2m"].dtype == pl.UInt8


def test_process_ecmwf_dataset_empty_h3_grid():
    h3_grid = H3GridWeights.validate(
        pl.DataFrame(
            schema={
                "h3_index": pl.UInt64,
                "nwp_lat": pl.Float64,
                "nwp_lng": pl.Float64,
                "len": pl.UInt32,
                "total": pl.UInt32,
                "proportion": pl.Float64,
            }
        )
    )
    init_time = pd.Timestamp("2024-04-01", tz="UTC")
    ds = create_mock_ds([51.5], [0.0], init_time, [pd.Timedelta(hours=0)], [0])

    result = process_ecmwf_dataset(init_time.to_pydatetime(), ds, h3_grid)
    assert len(result) == 0


def test_download_ecmwf_empty_h3_grid():
    h3_grid = H3GridWeights.validate(
        pl.DataFrame(
            schema={
                "h3_index": pl.UInt64,
                "nwp_lat": pl.Float64,
                "nwp_lng": pl.Float64,
                "len": pl.UInt32,
                "total": pl.UInt32,
                "proportion": pl.Float64,
            }
        )
    )
    init_time = np.datetime64("2024-04-01T00:00:00")
    with pytest.raises(ValueError, match="h3_grid is empty"):
        download_ecmwf(init_time, h3_grid)


def test_download_ecmwf_longitude_validation():
    """Assert that download_ecmwf raises a ValueError if longitudes are out of bounds."""
    h3_grid = H3GridWeights.validate(
        pl.DataFrame(
            {
                "h3_index": [0x85194AD7FFFFFFF],
                "nwp_lat": [51.5],
                "nwp_lng": [-1.0],
                "len": [1],
                "total": [1],
                "proportion": [1.0],
            },
            schema={
                "h3_index": pl.UInt64,
                "nwp_lat": pl.Float64,
                "nwp_lng": pl.Float64,
                "len": pl.UInt32,
                "total": pl.UInt32,
                "proportion": pl.Float64,
            },
        )
    )

    init_time = pd.Timestamp("2024-04-01", tz="UTC")
    # Provide longitude 359.0 in the source dataset, which is out of the [-180, 180] range
    ds = create_mock_ds([51.5], [359.0], init_time, [pd.Timedelta(hours=0)], [0])

    with pytest.raises(ValueError, match=r"Dataset longitude must be in the range \[-180, 180\]"):
        download_ecmwf(init_time.to_datetime64(), h3_grid, ds=ds)
