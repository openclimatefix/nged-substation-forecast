import xarray as xr
import psutil
import os


def print_mem(label):
    process = psutil.Process(os.getpid())
    print(f"{label}: {process.memory_info().rss / 1024 / 1024:.2f} MB")


print_mem("Start")
ds_opened = xr.open_zarr("test_large2.zarr", chunks=None)
print_mem("After open")
ds_sliced = ds_opened.sel(lat=slice(0, 10), lon=slice(0, 10))
print_mem("After slice")
res = ds_sliced.temp.compute()
print_mem("After compute")
print(res.shape)
