import xarray as xr
import pytest
import polars as pl
from contracts.data_schemas import H3GridWeights
from datetime import datetime, timezone
from dynamical_data.processing import download_ecmwf, process_ecmwf_dataset


def test_malformed_data_fails(broken_zarr_factory):
    """Verify that malformed data (NaNs in forbidden places) fails validation."""
    zarr_path = broken_zarr_factory("malformed_data")
    # Explicitly setting decode_timedelta=True avoids reliance on Xarray's
    # deprecated automatic decoding of time units, ensuring lead_time is
    # correctly parsed as timedelta64[ns].
    ds = xr.open_zarr(zarr_path, decode_timedelta=True)
    init_time = ds.init_time.values[0]

    h3_grid = H3GridWeights.validate(
        pl.DataFrame(
            {
                "h3_index": [123456789],
                "nwp_lat": [55.75],
                "nwp_lng": [-3.5],
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

    downloaded_ds = download_ecmwf(init_time, h3_grid, ds)
    init_dt = datetime.fromtimestamp(init_time.astype("datetime64[s]").astype(int), tz=timezone.utc)

    # This should raise a validation error from Patito/Nwp.validate
    with pytest.raises(Exception) as excinfo:
        process_ecmwf_dataset(nwp_init_time=init_dt, loaded_ds=downloaded_ds, h3_grid=h3_grid)

    print(f"Caught expected exception: {type(excinfo.value).__name__}: {excinfo.value}")
