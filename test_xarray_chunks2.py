import xarray as xr

ds_opened = xr.open_zarr("test.zarr", chunks=None)
print(type(ds_opened.temp.variable._data))
print(type(ds_opened.temp.data))
