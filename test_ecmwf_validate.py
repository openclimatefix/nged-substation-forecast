import icechunk
import xarray as xr
import psutil
import os
from contracts.settings import Settings
from dynamical_data.processing import validate_dataset_schema


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
validate_dataset_schema(ds)
print_mem("After validate")
