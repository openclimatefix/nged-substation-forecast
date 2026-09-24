"""Where every download for issue #841 lands, and the private trial-area box.

One-off throwaway module for the downloads in
<https://github.com/openclimatefix/nged-substation-forecast/issues/841>, which feed the two
past-weather studies (#809) and the forecast study (#810). It duplicates the small `data/`
resolution helper in `studies/beam_diffuse_split/sources.py` rather than importing it, so this
directory stays self-contained the way every other study directory is.
"""

import json
import os
from pathlib import Path
from typing import Final

import polars as pl
from contracts.settings import PROJECT_ROOT
from dotenv import load_dotenv


def _main_checkout(root: Path) -> Path:
    """Return the repository's main working tree, given any working tree's root.

    Copied from `studies.beam_diffuse_split.sources._main_checkout`: see that function's docstring
    for why a linked worktree must not get its own empty `data/`.
    """
    marker = root / ".git"
    if not marker.is_file():
        return root
    pointer = marker.read_text().removeprefix("gitdir:").strip()
    if not pointer:
        return root
    git_dir = Path(pointer)
    if git_dir.parent.name != "worktrees":
        return root
    return git_dir.parent.parent.parent


REPO_DATA_DIR: Final[Path] = Path(
    os.environ.get("DATA_PATH_INTERNAL") or _main_checkout(PROJECT_ROOT) / "data"
)
"""Where every download and every built frame lands. See `sources.REPO_DATA_DIR` for the full
rationale; this is the same resolution, duplicated rather than imported."""

WEATHER_DOWNLOADS_DIR: Final[Path] = REPO_DATA_DIR / "studies" / "weather"
"""One subdirectory per product, e.g. `ECMWF-IFS-HRES`, `NORA3`, `ICON-DREAM-EU`."""

load_dotenv(PROJECT_ROOT / ".env")


def open_meteo_api_key() -> str | None:
    """Return the Open-Meteo commercial API key from `.env`/the environment, or `None` if unset.

    A caller with a key switches from the free `<name>-api.open-meteo.com` host to
    `customer-<name>-api.open-meteo.com` and appends `&apikey=<key>` to the request, which lifts
    the free tier's daily/hourly/minutely rate limits entirely (confirmed against both the
    Previous Runs and Historical Forecast customer hosts). Never log or print the returned value.
    """
    return os.environ.get("OPEN_METEO_API_KEY")


_TRIAL_AREA_BOX_PATH: Final[Path] = WEATHER_DOWNLOADS_DIR / "_trial_area_box.json"
"""Where the trial-area box's bounds are kept.

**This file is never read by anything outside this process's private working state, and its
contents must never be logged, printed, committed, or quoted back in a report.** The bounds are
derived from the private generator roster (`packages/contracts` `TimeSeriesMetadata`), and NGED's
generator locations must never appear in anything published — see CLAUDE.md.
"""


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

    Returns:
        The box, held only in memory from here on.

    Raises:
        FileNotFoundError: If the box has not been derived yet (see
            `write_trial_area_box_from_roster` below, run once per checkout).
    """
    bounds = json.loads(_TRIAL_AREA_BOX_PATH.read_text())
    return TrialAreaBox(**bounds)


def write_trial_area_box_from_roster(*, margin_deg: float = 0.15) -> None:
    """Derive the trial-area box from the private generator roster and write it to disk.

    Run once per checkout (or whenever the roster changes). The box is the roster's own lat/lon
    extent, widened by `margin_deg` on every side — a few grid cells at the 9 km-to-25 km spacing
    of the coarser products this issue downloads. Nothing here prints or returns the bounds; they
    are read back only through `load_trial_area_box`.

    Args:
        margin_deg: How far to widen the roster's own bounding box on every side, in degrees.
    """
    from contracts.settings import get_settings

    metadata = pl.read_parquet(get_settings().metadata_path)
    lat_min, lat_max, lon_min, lon_max = metadata.select(
        pl.col("latitude").min().alias("lat_min"),
        pl.col("latitude").max().alias("lat_max"),
        pl.col("longitude").min().alias("lon_min"),
        pl.col("longitude").max().alias("lon_max"),
    ).row(0)
    WEATHER_DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    _TRIAL_AREA_BOX_PATH.write_text(
        json.dumps(
            {
                "lat_min": lat_min - margin_deg,
                "lat_max": lat_max + margin_deg,
                "lon_min": lon_min - margin_deg,
                "lon_max": lon_max + margin_deg,
            }
        )
    )


if __name__ == "__main__":
    write_trial_area_box_from_roster()
