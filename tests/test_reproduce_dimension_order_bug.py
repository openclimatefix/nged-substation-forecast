import xarray as xr
import numpy as np
from dynamical_data.processing import validate_dataset_schema, REQUIRED_NWP_VARS


def test_fix_dimension_order_bug():
    """Verify that incorrect dimension order is fixed by validate_dataset_schema."""
    init_time = np.datetime64("2026-03-01T00:00:00")

    # Create a dataset with the WRONG dimension order for categorical_precipitation_type_surface
    # Expected: ('latitude', 'longitude', 'init_time', 'lead_time', 'ensemble_member')
    # Actual: ('init_time', 'lead_time', 'ensemble_member', 'latitude', 'longitude')

    wrong_dims = ("init_time", "lead_time", "ensemble_member", "latitude", "longitude")
    correct_dims = ("latitude", "longitude", "init_time", "lead_time", "ensemble_member")
    shape = (1, 1, 1, 1, 1)

    data_vars = {}
    for var in REQUIRED_NWP_VARS:
        if var == "categorical_precipitation_type_surface":
            data_vars[var] = (wrong_dims, np.zeros(shape))
        else:
            data_vars[var] = (correct_dims, np.zeros(shape))

    ds = xr.Dataset(
        {
            "latitude": (["latitude"], [56.0]),
            "longitude": (["longitude"], [-3.25]),
            "init_time": (["init_time"], [init_time]),
            "lead_time": (["lead_time"], np.array([0], dtype="timedelta64[h]")),
            "ensemble_member": (["ensemble_member"], [0]),
            **data_vars,
        }
    )

    # This should NOT raise MalformedZarrError anymore
    validate_dataset_schema(ds)

    # Verify the dimension order is now correct
    assert ds["categorical_precipitation_type_surface"].dims == correct_dims
