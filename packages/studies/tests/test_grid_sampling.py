import numpy as np
import polars as pl
import pytest
import xarray as xr
from pyproj import CRS, Transformer
from studies.grid_sampling import nearest_cells, nearest_grid_indices, sample_nearest_cell

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


def test_the_nearest_cell_is_nearest_on_the_ground_not_in_degrees():
    # At 53°N a degree of longitude is about 67 km and a degree of latitude about 111 km. The site
    # sits 0.06° north of one cell and 0.08° east of the other: nearer the first in degrees, but
    # nearer the second on the ground (about 5.4 km against 6.7 km).
    cells = pl.DataFrame({"cell_id": [7, 9], "latitude": [53.06, 53.0], "longitude": [0.0, 0.08]})
    sites = pl.DataFrame({"site": ["A"], "latitude": [53.0], "longitude": [0.0]})

    nearest = nearest_cells(sites=sites, cells=cells)

    assert nearest["cell_id"].to_list() == [9]
    assert nearest["distance_km"].item() == pytest.approx(5.35, abs=0.05)


def test_each_site_gets_its_own_nearest_cell():
    latitudes, longitudes = np.meshgrid(np.arange(52.0, 54.0, 0.05), np.arange(-1.0, 1.0, 0.05))
    cells = pl.DataFrame(
        {"latitude": latitudes.ravel(), "longitude": longitudes.ravel()}
    ).with_row_index(name="cell_id")
    sites = pl.DataFrame(
        {"site": ["A", "B"], "latitude": [52.51, 53.49], "longitude": [0.49, -0.74]}
    )

    nearest = nearest_cells(sites=sites, cells=cells).join(cells, on="cell_id").sort("site")

    assert nearest["latitude"].to_list() == pytest.approx([52.5, 53.5])
    assert nearest["longitude"].to_list() == pytest.approx([0.5, -0.75])


def test_no_cells_raises():
    empty = pl.DataFrame(
        {"cell_id": [], "latitude": [], "longitude": []},
        schema={"cell_id": pl.Int64, "latitude": pl.Float64, "longitude": pl.Float64},
    )
    sites = pl.DataFrame({"site": ["A"], "latitude": [53.0], "longitude": [0.0]})

    with pytest.raises(ValueError, match="no cells"):
        nearest_cells(sites=sites, cells=empty)


def test_grid_indices_name_each_sites_cell_on_a_grid_that_is_not_square():
    # Five latitudes and eight longitudes, so a split of the flat index that swapped the axes would
    # land on another cell, or off the grid.
    latitudes = np.array([52.0, 52.1, 52.2, 52.3, 52.4])
    longitudes = np.array([-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0])
    sites = pl.DataFrame(
        {"site": ["A", "B"], "latitude": [52.31, 52.02], "longitude": [-0.1, -0.69]}
    )

    indices = nearest_grid_indices(sites=sites, latitudes=latitudes, longitudes=longitudes)

    assert indices["lat_index"].to_list() == [3, 0]
    assert indices["lon_index"].to_list() == [6, 0]
