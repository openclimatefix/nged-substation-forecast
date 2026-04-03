import xarray as xr
import numpy as np

# Create a dummy zarr store
ds = xr.Dataset(
    {"temp": (("time", "lat", "lon"), np.random.rand(100, 100, 100))},
    coords={
        "time": np.arange(100),
        "lat": np.arange(100),
        "lon": np.arange(100),
    },
)
ds.to_zarr("test.zarr", mode="w")

# Open with chunks=None
ds_opened = xr.open_zarr("test.zarr", chunks=None)
print(type(ds_opened.temp.data))

# Open with chunks={}
ds_opened_dask = xr.open_zarr("test.zarr", chunks={})
print(type(ds_opened_dask.temp.data))
