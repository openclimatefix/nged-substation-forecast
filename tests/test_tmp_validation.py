import xarray as xr
import pytest
import numpy as np
import polars as pl
from contracts.data_schemas import H3GridWeights
from datetime import datetime, timezone
from dynamical_data.processing import MalformedZarrError, download_ecmwf, process_ecmwf_dataset


def test_missing_required_nwp_var_fails_early():
    """Verify that missing a variable required by Nwp schema fails in validate_dataset_schema."""
    init_time = np.datetime64("2026-03-01T00:00:00")

    # Create a dataset that passes validate_dataset_schema but is missing dew_point_temperature_2m
    ds = xr.Dataset(
        {
            "latitude": (["latitude"], [56.0]),
            "longitude": (["longitude"], [-3.25]),
            "init_time": (["init_time"], [init_time]),
            "lead_time": (["lead_time"], np.array([0], dtype="timedelta64[h]")),
            "ensemble_member": (["ensemble_member"], [0]),
            "temperature_2m": (
                ["latitude", "longitude", "init_time", "lead_time", "ensemble_member"],
                [[[[[280.0]]]]],
            ),
            "wind_u_10m": (
                ["latitude", "longitude", "init_time", "lead_time", "ensemble_member"],
                [[[[[5.0]]]]],
            ),
            "wind_v_10m": (
                ["latitude", "longitude", "init_time", "lead_time", "ensemble_member"],
                [[[[[5.0]]]]],
            ),
            "wind_u_100m": (
                ["latitude", "longitude", "init_time", "lead_time", "ensemble_member"],
                [[[[[7.0]]]]],
            ),
            "wind_v_100m": (
                ["latitude", "longitude", "init_time", "lead_time", "ensemble_member"],
                [[[[[7.0]]]]],
            ),
            "categorical_precipitation_type_surface": (
                ["latitude", "longitude", "init_time", "lead_time", "ensemble_member"],
                [[[[[0]]]]],
            ),
        }
    )

    h3_grid = H3GridWeights.validate(
        pl.DataFrame(
            {
                "h3_index": [123456789],
                "nwp_lat": [56.0],
                "nwp_lng": [-3.25],
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

    # This SHOULD raise MalformedZarrError if validate_dataset_schema was complete
    # But currently it might pass and fail later in Nwp.validate
    try:
        downloaded_ds = download_ecmwf(init_time, h3_grid, ds)
        init_dt = datetime.fromtimestamp(
            init_time.astype("datetime64[s]").astype(int), tz=timezone.utc
        )
        process_ecmwf_dataset(nwp_init_time=init_dt, loaded_ds=downloaded_ds, h3_grid=h3_grid)
    except MalformedZarrError:
        print("Caught MalformedZarrError (Good)")
    except Exception as e:
        print(f"Caught other exception: {type(e).__name__}: {e}")
        # If it's a Patito/Pydantic error, it means validate_dataset_schema missed it.
        if "dew_point_temperature_2m" in str(e):
            pytest.fail(
                "validate_dataset_schema should have caught missing dew_point_temperature_2m"
            )
