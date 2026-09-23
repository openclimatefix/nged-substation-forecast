"""Download 100 m and 10 m wind from five weather products at each metered wind generator.

One-off throwaway script for the study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/826>.

**Every product is asked for its 100 m wind speed and direction and its 10 m wind speed, and the
ICON products for their 80 m wind as well.** ERA5 and UKV publish 100 m wind natively. The three
ICON products publish 80 m and 120 m, and Open-Meteo's 100 m value for them is the 120 m speed
scaled by 0.98, which a tree cannot tell apart from the 120 m speed itself. So the study shows each
ICON product its native 80 m wind as the hub-height column, which keeps every arm to the same number
of columns, and keeps the 100 m column for a sensitivity check.

**Every product is read from the nearest land cell.** At one wind generator the nearest ICON global
cell is influenced by the sea, with a 10 m speed 16% higher than the land cell's; every other
product returns the same values from either choice at all three generators.

**The window starts on 2024-08-12**, when Open-Meteo's own UKV downloader started. UKV's archive
before that date carries radiation but no hub-height wind. Coordinates are read at run time from
the private roster and sent in the query string; no coordinate and no identifier reaches the
written frame, whose rows are keyed by the anonymous wind labels.

Speeds stay in Open-Meteo's default unit, km/h. A tree is indifferent to the unit.

Run it with `uv run python studies/beam_diffuse_split/fetch_wind_point.py`.
"""

import logging
import sys
from pathlib import Path
from typing import Final

import polars as pl
from build_dataset import _wind_sites
from era5_grid import LAST_DATE, LAST_YEAR
from fetch_open_meteo_point import fetch_point_frame
from sources import HISTORICAL_FORECAST_URL, WEATHER_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("fetch_wind_point")

ARCHIVE_URL: Final[str] = "https://archive-api.open-meteo.com/v1/archive"
"""Open-Meteo's reanalysis endpoint, which serves ERA5 with the same query shape."""

WIND_VARIABLES: Final[tuple[str, ...]] = (
    "wind_speed_100m",
    "wind_direction_100m",
    "wind_speed_10m",
)
"""The columns every product is asked for."""

ICON_VARIABLES: Final[tuple[str, ...]] = ("wind_speed_80m", "wind_direction_80m")
"""The native hub-height columns the ICON products are asked for as well."""

FIRST_DATE: Final[str] = "2024-08-12"
"""The first day of the window: when Open-Meteo's own UKV downloader started."""

PRODUCTS: Final[dict[str, tuple[str, str]]] = {
    "era5": ("era5", ARCHIVE_URL),
    "ukv": ("ukmo_uk_deterministic_2km", HISTORICAL_FORECAST_URL),
    "icon_d2": ("icon_d2", HISTORICAL_FORECAST_URL),
    "icon_eu": ("icon_eu", HISTORICAL_FORECAST_URL),
    "icon_global": ("icon_global", HISTORICAL_FORECAST_URL),
}
"""Each product's `models=` value and endpoint."""


def output_path_for(*, product: str) -> Path:
    """Return where one product's wind download is written.

    Args:
        product: A key of `PRODUCTS`.

    Returns:
        The parquet path.
    """
    return WEATHER_DATA_DIR / product.upper().replace("_", "-") / f"wind_{product}.parquet"


def main() -> int:
    """Download every product's wind at every wind generator, one year per request."""
    sites = _wind_sites()
    first_year = int(FIRST_DATE[:4])
    for product, (models_parameter, base_url) in PRODUCTS.items():
        frame = (
            pl.concat(
                fetch_point_frame(
                    sites=sites,
                    variables=(
                        (*WIND_VARIABLES, *ICON_VARIABLES)
                        if product.startswith("icon")
                        else WIND_VARIABLES
                    ),
                    models_parameter=models_parameter,
                    first_date=FIRST_DATE if year == first_year else f"{year}-01-01",
                    last_date=LAST_DATE if year == LAST_YEAR else f"{year}-12-31",
                    base_url=base_url,
                    cell_selection="land",
                )
                for year in range(first_year, LAST_YEAR + 1)
            )
            .unique(subset=["site", "time"], keep="first")
            .sort("site", "time")
        )
        path = output_path_for(product=product)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
        _LOG.info("%s: %d rows for %d sites to %s", product, frame.height, sites.height, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
