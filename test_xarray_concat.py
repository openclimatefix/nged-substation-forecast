import xarray as xr
import psutil
import os


def print_mem():
    process = psutil.Process(os.getpid())
    print(f"Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")


print_mem()
ds_opened = xr.open_zarr("test_large.zarr", chunks=None)
print_mem()
ds_pos = ds_opened.sel(lon=slice(50, 100))
ds_neg = ds_opened.sel(lon=slice(0, 49))
ds_concat = xr.concat([ds_pos, ds_neg], dim="lon")
print_mem()
