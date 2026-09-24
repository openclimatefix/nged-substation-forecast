"""Find the grid cell nearest each site, on a projected grid or among a set of cell centres."""

from typing import Final

import numpy as np
import polars as pl
import xarray as xr
from pyproj import CRS, Transformer

EARTH_RADIUS_KM: Final[float] = 6371.0
"""The mean radius of the Earth, which turns an angle on the sphere into a distance."""


def sample_nearest_cell(*, field: xr.DataArray, sites: pl.DataFrame, crs: CRS) -> dict[str, float]:
    """Return the value of the grid cell nearest each site.

    The site coordinates are transformed into the grid's own projection before the lookup, so the
    nearest cell is nearest in metres on the grid rather than in degrees. The grid's axes must be
    `projection_x_coordinate` and `projection_y_coordinate`, the CF names the Met Office uses.

    **A site outside the grid silently reads the edge cell**, because `method="nearest"` returns
    the edge cell for a point beyond the domain. A caller whose sites might fall outside it has to
    check the domain first.

    Args:
        field: A two-dimensional field on the projected axes.
        sites: The roster, carrying `site`, `latitude` and `longitude`.
        crs: The grid's projection.

    Returns:
        The value at each site's nearest grid cell, keyed by the site label.
    """
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    sampled: dict[str, float] = {}
    for row in sites.to_dicts():
        easting, northing = transformer.transform(row["longitude"], row["latitude"])
        sampled[str(row["site"])] = float(
            field.sel(
                projection_x_coordinate=easting,
                projection_y_coordinate=northing,
                method="nearest",
            ).item()
        )
    return sampled


def distance_matrix_km(*, sites: pl.DataFrame, cells: pl.DataFrame) -> np.ndarray:
    """Return the great-circle distance in km from every site to every cell centre.

    Args:
        sites: The roster, carrying `latitude` and `longitude` in degrees.
        cells: One row per cell, carrying `latitude` and `longitude` in degrees.

    Returns:
        An array of shape (number of sites, number of cells), in the row order of each frame.
    """
    cell_latitude = np.radians(cells["latitude"].to_numpy().astype(np.float64))
    cell_longitude = np.radians(cells["longitude"].to_numpy().astype(np.float64))
    site_latitude = np.radians(sites["latitude"].to_numpy().astype(np.float64))[:, None]
    site_longitude = np.radians(sites["longitude"].to_numpy().astype(np.float64))[:, None]
    half_chord = (
        np.sin((cell_latitude - site_latitude) / 2.0) ** 2
        + np.cos(site_latitude)
        * np.cos(cell_latitude)
        * np.sin((cell_longitude - site_longitude) / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(half_chord, 0.0, 1.0)))


def nearest_cells(*, sites: pl.DataFrame, cells: pl.DataFrame) -> pl.DataFrame:
    """Return the cell whose centre is nearest each site, by great-circle distance.

    **Distance is measured on the sphere, never in degrees.** At the trial area's latitude a degree
    of longitude is about 0.6 of a degree of latitude, so the nearest cell in degrees can be a cell
    further away on the ground. The cells can be any set of centres: a regular latitude-longitude
    grid flattened to one row per cell, or an unstructured grid such as ICON's triangles.

    **A site outside the cells' extent still gets a cell**, the nearest edge cell, so a caller whose
    sites might fall outside the extent reads `distance_km` before trusting the result.

    Args:
        sites: The roster, carrying `site`, `latitude` and `longitude`.
        cells: One row per cell, carrying `cell_id`, `latitude` and `longitude`.

    Returns:
        One row per site with `site`, the nearest `cell_id`, and `distance_km` to its centre.

    Raises:
        ValueError: If `cells` is empty, which leaves no cell to choose.
    """
    if cells.height == 0:
        msg = "no cells to choose from"
        raise ValueError(msg)
    distance_km = distance_matrix_km(sites=sites, cells=cells)
    nearest = distance_km.argmin(axis=1)
    return pl.DataFrame(
        {
            "site": sites["site"],
            "cell_id": cells["cell_id"].gather(nearest),
            "distance_km": distance_km[np.arange(sites.height), nearest],
        }
    )


def nearest_grid_indices(
    *, sites: pl.DataFrame, latitudes: np.ndarray, longitudes: np.ndarray
) -> pl.DataFrame:
    """Return the indices into a regular latitude-longitude grid's two axes of each site's cell.

    The grid is flattened latitude-major and handed to `nearest_cells`, and each flat index is
    split back into its two axis indices. **A split that swapped the axes would still return
    plausible distances**, so the result is checked against the cells it names before it is
    returned.

    Args:
        sites: The roster, carrying `site`, `latitude` and `longitude`.
        latitudes: The grid's latitude axis, in degrees.
        longitudes: The grid's longitude axis, in degrees.

    Returns:
        One row per site with `site`, `lat_index`, `lon_index` and `distance_km`.

    Raises:
        ValueError: If a returned index pair does not name the cell `nearest_cells` chose.
    """
    lat_grid, lon_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
    cells = pl.DataFrame(
        {"latitude": lat_grid.ravel(), "longitude": lon_grid.ravel()}
    ).with_row_index(name="cell_id")
    nearest = nearest_cells(sites=sites, cells=cells).join(
        cells.rename({"latitude": "cell_latitude", "longitude": "cell_longitude"}), on="cell_id"
    )
    indices = nearest.with_columns(
        lat_index=pl.col("cell_id") // len(longitudes),
        lon_index=pl.col("cell_id") % len(longitudes),
    )
    lat_index = indices["lat_index"].to_numpy()
    lon_index = indices["lon_index"].to_numpy()
    if not (
        np.array_equal(latitudes[lat_index], indices["cell_latitude"].to_numpy())
        and np.array_equal(longitudes[lon_index], indices["cell_longitude"].to_numpy())
    ):
        msg = "an axis index does not name the cell chosen"
        raise ValueError(msg)
    return indices.select("site", "lat_index", "lon_index", "distance_km").sort("site")
