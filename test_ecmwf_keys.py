import icechunk
import xarray as xr
import psutil
import os
from contracts.settings import Settings


def print_mem(label):
    process = psutil.Process(os.getpid())
    print(f"{label}: {process.memory_info().rss / 1024 / 1024:.2f} MB")


print_mem("Start")
_SETTINGS = Settings()
storage = icechunk.s3_storage(
    bucket=_SETTINGS.ecmwf_s3_bucket,
    prefix=_SETTINGS.ecmwf_s3_prefix,
    region="us-west-2",
    anonymous=True,
)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session("main")
ds = xr.open_zarr(session.store, chunks=None, decode_timedelta=True)
print_mem("After open_zarr")
ds_cropped = ds.sel(
    latitude=slice(60, 50), longitude=slice(-8, 2), init_time=ds.init_time.values[-1]
)
print_mem("After sel")
print(len(ds_cropped.data_vars.keys()))
print_mem("After keys")
