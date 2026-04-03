import xarray as xr
import numpy as np
import psutil
import os


def print_mem():
    process = psutil.Process(os.getpid())
    print(f"Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")


print_mem()
ds = xr.Dataset(
    {"temp": (("time", "lat", "lon"), np.random.rand(1000, 1000, 100))},
    coords={
        "time": np.arange(1000),
        "lat": np.arange(1000),
        "lon": np.arange(100),
    },
)
ds.to_zarr("test_large.zarr", mode="w")
print_mem()
del ds
print_mem()

ds_opened = xr.open_zarr("test_large.zarr", chunks=None)
print_mem()
# Slice it
ds_sliced = ds_opened.sel(time=slice(0, 10), lat=slice(0, 10), lon=slice(0, 10))
print_mem()
# Compute
res = ds_sliced.temp.compute()
print_mem()
