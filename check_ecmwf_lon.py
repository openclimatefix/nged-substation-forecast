import xarray as xr
import icechunk

storage = icechunk.s3_storage(
    bucket="dynamical-ecmwf-ifs-ens",
    prefix="ecmwf-ifs-ens-forecast-15-day-0-25-degree/v0.1.0.icechunk/",
    region="us-west-2",
    anonymous=True,
)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session("main")
ds = xr.open_zarr(session.store, chunks=None, decode_timedelta=True)

print("Longitude min:", ds.longitude.min().values)
print("Longitude max:", ds.longitude.max().values)
print("Longitude values (first 10):", ds.longitude.values[:10])
print("Longitude values (last 10):", ds.longitude.values[-10:])
