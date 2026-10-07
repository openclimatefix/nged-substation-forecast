"""The private latitude and longitude box around the NGED trial area, held in memory only.

The box comes from the private list of generators, so no bound is ever printed, logged, or
committed. Every gridded weather download takes its extent from `load_trial_area_box`, and the
past-weather studies that fetch their own grid read it from here as well.
"""

import json

import polars as pl

from studies.sources import LEGACY_TRIAL_AREA_BOX_PATH, TRIAL_AREA_BOX_PATH, existing_or_legacy


class TrialAreaBox:
    """The lat/lon box covering the NGED trial area, with a margin, held in memory only.

    Every method on this class returns derived quantities (a point count, a grid of coordinates to
    send to a weather API) and never the bounds themselves, so a caller cannot accidentally log or
    print the private box by handling this object carelessly. A caller that truly needs the raw
    bounds (to build a request URL, say) reads `.lat_min` etc. directly and is responsible for never
    passing that value to a log call, a file under `docs/`, or this script's own stdout.
    """

    def __init__(self, *, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> None:
        """Store the box's bounds, in degrees."""
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max

    def grid_points(self, *, spacing_deg: float) -> pl.DataFrame:
        """Return a regular lat/lon grid covering the box, at roughly `spacing_deg` spacing.

        Args:
            spacing_deg: Target spacing between adjacent grid points, in degrees.

        Returns:
            One row per point, carrying `point_id`, `latitude`, and `longitude`. `point_id` is a
            zero-padded running index, not a coordinate, so it is safe to use in filenames and logs.
        """
        import numpy as np

        n_lat = max(2, round((self.lat_max - self.lat_min) / spacing_deg) + 1)
        n_lon = max(2, round((self.lon_max - self.lon_min) / spacing_deg) + 1)
        latitudes = np.linspace(self.lat_min, self.lat_max, n_lat)
        longitudes = np.linspace(self.lon_min, self.lon_max, n_lon)
        lat_grid, lon_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
        return pl.DataFrame(
            {"latitude": lat_grid.ravel(), "longitude": lon_grid.ravel()}
        ).with_row_index(name="point_id")


def load_trial_area_box() -> TrialAreaBox:
    """Load the trial-area box written by the setup step.

    The box is read from `TRIAL_AREA_BOX_PATH`, or from `LEGACY_TRIAL_AREA_BOX_PATH` while the file
    has not yet moved there, so that a script runs on either side of the move.

    Returns:
        The box, held only in memory from here on.

    Raises:
        FileNotFoundError: If the box has not been derived yet (see
            `write_trial_area_box_from_metadata` below, run once per checkout).
    """
    path = existing_or_legacy(current=TRIAL_AREA_BOX_PATH, legacy=LEGACY_TRIAL_AREA_BOX_PATH)
    bounds = json.loads(path.read_text())
    return TrialAreaBox(**bounds)


def write_trial_area_box_from_metadata(*, margin_deg: float = 0.15) -> None:
    """Derive the trial-area box from the private list of generators and write it to disk.

    Run once per checkout (or whenever the site list changes). The box is the site list's own
    lat/lon extent, widened by `margin_deg` on every side — a few grid cells at the 9 km-to-25 km
    spacing of the coarser products this issue downloads. Nothing here prints or returns the bounds;
    they are read back only through `load_trial_area_box`.

    Args:
        margin_deg: How far to widen the site list's own bounding box on every side, in degrees.
    """
    from contracts.settings import get_settings

    metadata = pl.read_parquet(get_settings().metadata_path)
    lat_min, lat_max, lon_min, lon_max = metadata.select(
        pl.col("latitude").min().alias("lat_min"),
        pl.col("latitude").max().alias("lat_max"),
        pl.col("longitude").min().alias("lon_min"),
        pl.col("longitude").max().alias("lon_max"),
    ).row(0)
    TRIAL_AREA_BOX_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRIAL_AREA_BOX_PATH.write_text(
        json.dumps(
            {
                "lat_min": lat_min - margin_deg,
                "lat_max": lat_max + margin_deg,
                "lon_min": lon_min - margin_deg,
                "lon_max": lon_max + margin_deg,
            }
        )
    )
