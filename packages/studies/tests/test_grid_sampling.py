import numpy as np
import polars as pl
import xarray as xr
from pyproj import CRS, Transformer
from studies.grid_sampling import sample_nearest_cell

# A Lambert azimuthal equal-area grid centred on Great Britain, the projection UKV is served on. The
# grid is deliberately not square, and its x and y ranges differ, so a sampler that swapped the axes
# or transformed latitude and longitude in the wrong order lands on the wrong cell rather than on a
# symmetric twin of the right one.
CRS_LAEA = CRS.from_proj4("+proj=laea +lat_0=54.9 +lon_0=-2.5 +x_0=0 +y_0=0 +R=6371229 +units=m")
SPACING_M = 2_000.0
X_M = np.arange(-60_000.0, 80_000.0, SPACING_M)
Y_M = np.arange(-30_000.0, 40_000.0, SPACING_M)


def _field() -> xr.DataArray:
    # Each cell holds 1000 times its row index plus its column index, so a value names its cell.
    values = np.arange(len(Y_M))[:, None] * 1000.0 + np.arange(len(X_M))[None, :]
    return xr.DataArray(
        values,
        coords={"projection_y_coordinate": Y_M, "projection_x_coordinate": X_M},
        dims=("projection_y_coordinate", "projection_x_coordinate"),
    )


def test_each_site_reads_the_cell_it_sits_in():
    # Place each site a little off a known cell's centre, then express it in degrees.
    cells = {"A": (3, 50), "B": (30, 7), "C": (12, 61)}
    to_degrees = Transformer.from_crs(CRS_LAEA, "EPSG:4326", always_xy=True)
    rows = []
    for site, (row, column) in cells.items():
        longitude, latitude = to_degrees.transform(X_M[column] + 300.0, Y_M[row] - 400.0)
        rows.append({"site": site, "latitude": latitude, "longitude": longitude})

    sampled = sample_nearest_cell(field=_field(), sites=pl.DataFrame(rows), crs=CRS_LAEA)

    assert sampled == {site: row * 1000.0 + column for site, (row, column) in cells.items()}
