"""Check that a crop of the AIFS grid fully covers the H3 cell of every study site.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/841>. The forecast study reads
each generator as an area-weighted mean over the generator's H3 resolution-5 cell
(`geo.h3.compute_h3_grid_weights`, as `studies/beam_diffuse_split/ens_past_solar.py` does). That
read needs every 0.25-degree grid column and row the cell touches to be inside the crop.

For each of the nine anonymised study sites (PV `A` to `F`, wind `W1` to `W3`) the script finds the
grid points the site's cell overlaps, and compares them with the grid points inside the crop that
`fetch_dynamical_zarr.py` would write for `--extra-east-columns`. **It prints only each site's
label, whether the cell is fully covered, and the smallest margin in grid columns between the
cell's outermost overlapped grid point and the crop's edge.** A margin of 0 means the cell reaches
the edge column. Site coordinates, the crop's bounds, and grid indices stay in memory.

Run it with `uv run python studies/weather_downloads/check_aifs_crop_covers_sites.py
--extra-east-columns 1`.
"""

import argparse
import sys
from typing import Final

import h3.api.basic_int as h3
import numpy as np
import polars as pl
from fetch_dynamical_zarr import _cropped_dataset
from fetch_open_meteo_previous_runs import _pv_sites, _wind_sites
from geo.h3 import compute_h3_grid_weights

H3_RESOLUTION: Final[int] = 5
"""The H3 resolution the studies average over, as in `ens_past_solar.H3_RESOLUTION`."""

GRID_STEP_DEGREES: Final[float] = 0.25
"""The AIFS grid spacing."""

DATASET_ID: Final[str] = "ecmwf-aifs-single-forecast"
"""Either AIFS dataset has the same grid, so one is enough."""


def _margins(*, needed: np.ndarray, crop: np.ndarray) -> tuple[float, float]:
    """Return the margins, in grid columns, on the low and high side of `needed` inside `crop`.

    Args:
        needed: The grid coordinates (degrees) the site's cell overlaps.
        crop: The crop's grid coordinates (degrees), in either order.

    Returns:
        The distance from the smallest and the largest needed coordinate to the crop's matching
        edge, in grid columns. A negative value means the crop stops short of the cell.
    """
    low = (needed.min() - crop.min()) / GRID_STEP_DEGREES
    high = (crop.max() - needed.max()) / GRID_STEP_DEGREES
    return float(low), float(high)


def main() -> int:
    """Print, per study site, whether the crop covers the site's H3 cell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra-east-columns", type=int, default=0)
    arguments = parser.parse_args()
    crop = _cropped_dataset(dataset_id=DATASET_ID, extra_east_columns=arguments.extra_east_columns)
    crop_lats = crop["latitude"].to_numpy()
    crop_lons = crop["longitude"].to_numpy()

    sites = pl.concat([_pv_sites(), _wind_sites()]).sort("site")
    all_covered = True
    for site, latitude, longitude in sites.iter_rows():
        cell = h3.latlng_to_cell(latitude, longitude, H3_RESOLUTION)
        weights = compute_h3_grid_weights(nwp_grid_size_degrees=GRID_STEP_DEGREES, h3_index=[cell])
        lat_margins = _margins(needed=weights["nwp_lat"].to_numpy(), crop=crop_lats)
        lon_margins = _margins(needed=weights["nwp_lon"].to_numpy(), crop=crop_lons)
        margin = round(min(*lat_margins, *lon_margins))
        covered = margin >= 0
        all_covered &= covered
        print(f"{site}: fully covered={'yes' if covered else 'no'}, smallest margin={margin}")
    return 0 if all_covered else 1


if __name__ == "__main__":
    sys.exit(main())
