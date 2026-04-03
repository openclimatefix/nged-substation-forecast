import icechunk
import xarray as xr
from contracts.settings import Settings

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
print(ds.temperature_2m.encoding.get("chunks"))
print(ds.temperature_2m.shape)
