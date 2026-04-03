import xarray as xr
import numpy as np
import psutil
import os


def print_mem(label):
    process = psutil.Process(os.getpid())
    print(f"{label}: {process.memory_info().rss / 1024 / 1024:.2f} MB")


print_mem("Start")
ds = xr.Dataset(
    {"temp": (("time", "lat", "lon"), np.random.rand(1000, 1000, 100))},
    coords={
        "time": np.arange(1000),
        "lat": np.arange(1000),
        "lon": np.arange(100),
    },
)
ds.to_zarr("test_large2.zarr", mode="w")
print_mem("After write")
del ds

ds_opened = xr.open_zarr("test_large2.zarr", chunks=None, decode_timedelta=True)
print_mem("After open")
