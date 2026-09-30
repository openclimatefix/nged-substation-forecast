"""Download CERRA's own 2D latitude and longitude arrays, for the whole domain.

One-off throwaway script for the downloads in
<https://github.com/openclimatefix/nged-substation-forecast/issues/841>. The cropped CERRA parquets
carry only `(y_index, x_index)`, so a study that must re-derive which cell is nearest each site
needs the grid itself. One time step of one variable is requested (about 10 MB), and only the
`latitude` and `longitude` fields are kept.

**The output is a whole-domain grid, not a site list, so it holds no generator coordinate.** It is
still private data under `data/`, and its values are never printed or logged.

Run with `uv run --with cdsapi --with netCDF4 python studies/weather_downloads/fetch_cerra_grid.py`.
Credentials are read by `cdsapi` from the environment or `~/.cdsapirc`; nothing is stored here.
"""

import sys
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import xarray as xr
from fetch_cerra import HEIGHT_LEVELS_DATASET, _build_request, download_one_chunk
from lineage import write_lineage_note, write_readme
from paths import WEATHER_DOWNLOADS_DIR

OUTPUT_DIR: Final[Path] = WEATHER_DOWNLOADS_DIR / "CERRA"
OUTPUT_PATH: Final[Path] = OUTPUT_DIR / "cerra_grid.parquet"
SCRATCH_PATH: Final[Path] = WEATHER_DOWNLOADS_DIR.parent.parent / "_scratch" / "cerra" / "grid.nc"
GRID_DATE: Final[str] = "2020-01-01"
"""Any served date works, because the grid is fixed; one day at 00:00 is the smallest request."""


def _write_readme() -> None:
    """Document the grid file next to its lineage note."""
    write_readme(
        product_dir=OUTPUT_DIR,
        filename="README_cerra_grid.md",
        product_name="CERRA whole-domain latitude and longitude grid",
        source_web_page="https://cds.climate.copernicus.eu/datasets/reanalysis-cerra-height-levels",
        script_path="studies/weather_downloads/fetch_cerra_grid.py",
        lineage_filenames=["lineage_cerra_grid.json"],
        columns={
            "y_index": "Row index into CERRA's native Lambert grid, as in the cropped parquets.",
            "x_index": "Column index into the same grid.",
            "latitude": "Cell latitude, degrees north.",
            "longitude": "Cell longitude, degrees east, normalised to [-180, 180).",
        },
        missing_value_convention="No missing values: every cell of the domain has a coordinate.",
        gotchas=[
            (
                "The grid is fixed in time, so one time step of one variable was requested. "
                "The wind values are discarded."
            ),
            (
                "The file is private data. It covers the whole domain and holds no site "
                "coordinate, but it lets a reader turn the `y_index` and `x_index` of any "
                "cropped file back into a location, so it must not be published."
            ),
        ],
        external_docs={
            "CERRA height levels": "https://cds.climate.copernicus.eu/datasets/"
            "reanalysis-cerra-height-levels",
        },
    )


def main() -> int:
    """Fetch one small field, then write the grid's index and coordinate columns."""
    if OUTPUT_PATH.exists():
        print(f"{OUTPUT_PATH.name} exists, skipping the download")
        _write_readme()
        return 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    request = _build_request(
        variable="wind_speed", height_level="100_m", start_date=GRID_DATE, end_date=GRID_DATE
    )
    request["month"] = ["01"]
    request["day"] = ["01"]
    request["time"] = ["00:00"]
    download_one_chunk(dataset=HEIGHT_LEVELS_DATASET, request=request, scratch_path=SCRATCH_PATH)
    with xr.open_dataset(SCRATCH_PATH) as ds:
        latitude = np.asarray(ds["latitude"].values)
        longitude = np.asarray(ds["longitude"].values)
    longitude = ((longitude + 180) % 360) - 180
    if latitude.ndim != 2 or latitude.shape != longitude.shape:
        msg = f"expected 2D latitude and longitude of one shape, got {latitude.shape}"
        raise RuntimeError(msg)
    y_index, x_index = np.indices(latitude.shape)
    frame = pl.DataFrame(
        {
            "y_index": y_index.ravel().astype(np.int32),
            "x_index": x_index.ravel().astype(np.int32),
            "latitude": latitude.ravel().astype(np.float64),
            "longitude": longitude.ravel().astype(np.float64),
        }
    )
    partial = OUTPUT_PATH.with_suffix(".parquet.tmp")
    frame.write_parquet(partial)
    write_lineage_note(
        product_dir=OUTPUT_DIR,
        filename="lineage_cerra_grid.json",
        source_address="https://cds.climate.copernicus.eu/datasets/reanalysis-cerra-height-levels",
        request_description=(
            "CERRA reanalysis-cerra-height-levels, wind_speed at 100 m, one analysis time step "
            f"({GRID_DATE} 00:00), whole domain. Only the 2D latitude and longitude are kept, "
            "with longitude normalised to [-180, 180) as `fetch_cerra.py` does."
        ),
        variables=["latitude", "longitude"],
        extra={"rows": frame.height, "grid_shape": list(latitude.shape)},
    )
    partial.rename(OUTPUT_PATH)
    SCRATCH_PATH.unlink()
    _write_readme()
    print(f"wrote {frame.height} grid cells to {OUTPUT_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
