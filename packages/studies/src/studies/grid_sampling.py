"""Sample a projected weather grid at a set of coordinates, by nearest cell."""

import polars as pl
import xarray as xr
from pyproj import CRS, Transformer


def sample_nearest_cell(*, field: xr.DataArray, sites: pl.DataFrame, crs: CRS) -> dict[str, float]:
    """Return the value of the grid cell nearest each site.

    The site coordinates are transformed into the grid's own projection before the lookup, so the
    nearest cell is nearest in metres on the grid rather than in degrees. The grid's axes must be
    `projection_x_coordinate` and `projection_y_coordinate`, the CF names the Met Office uses.

    **A site outside the grid is not an error.** `method="nearest"` returns the edge cell for a
    point beyond the domain, so a caller whose sites might fall outside it has to check the domain
    first.

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
