"""Download 10 m, 80 m, and 100 m wind from five weather products at each metered wind generator.

One-off throwaway script for the study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/826>.

**Every product is asked for its 100 m wind speed and direction and its 10 m wind speed, and the
ICON products for their 80 m wind as well.** ERA5 and UKV publish 100 m wind natively. The three
ICON products publish 80 m and 120 m, and Open-Meteo's 100 m value for them is the 120 m speed
scaled by 0.98, which a tree cannot tell apart from the 120 m speed itself. So the study shows each
ICON product its native 80 m wind, rather than both 80 m and 120 m, as the hub-height column, which
keeps every arm to the same number of columns, and keeps the 100 m column for a sensitivity check.

**Every product is read from the nearest land cell.** At one wind generator the nearest ICON global
cell is influenced by the sea, with a 10 m speed 16% higher than the land cell's; every other
product returns the same values from either choice at all three generators.

**The window starts on 2024-08-12**, when Open-Meteo's own UKV downloader started. UKV's archive
before that date carries radiation but no hub-height wind. Coordinates are read at run time from
the private site list and sent in the query string; no coordinate and no identifier reaches the
written frame, whose rows are keyed by the anonymous wind labels.

Speeds stay in Open-Meteo's default unit, km/h. A tree is indifferent to the unit.

Run it with `uv run python studies/past_weather/fetch_wind_point.py`.
"""

import logging
import sys
from typing import Final

import polars as pl
from fetch_open_meteo_point import fetch_point_frame
from studies.era5_grid import FIRST_DATE_OVERRIDE, LAST_DATE, LAST_YEAR
from studies.pv_dataset import wind_sites
from studies.wind_product_frames import PRODUCTS, output_path_for

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("fetch_wind_point")


WIND_VARIABLES: Final[tuple[str, ...]] = (
    "wind_speed_100m",
    "wind_direction_100m",
    "wind_speed_10m",
)
"""The columns every product is asked for."""

EIGHTY_METRE_VARIABLES: Final[tuple[str, ...]] = ("wind_speed_80m", "wind_direction_80m")
"""The 80 m wind every product but ERA5 is asked for as well.

It is native for the ICON products, and Open-Meteo also serves it for UKV, where it is not a
rescaling of the 100 m value. ERA5 publishes no 80 m wind.
"""

FIRST_DATE: Final[str] = FIRST_DATE_OVERRIDE or "2024-08-12"
"""The first day of the window: when Open-Meteo's own UKV downloader started.

`ERA5_FIRST_DATE` overrides it.
"""


def main() -> int:
    """Download wind at every wind generator, one year per request.

    Downloads every product, or only those named on the command line.
    """
    sites = wind_sites()
    first_year = int(FIRST_DATE[:4])
    products = {name: PRODUCTS[name] for name in sys.argv[1:]} or PRODUCTS
    for product, (models_parameter, base_url) in products.items():
        frame = pl.concat(
            fetch_point_frame(
                sites=sites,
                variables=(
                    (*WIND_VARIABLES, *EIGHTY_METRE_VARIABLES)
                    if product != "era5"
                    else WIND_VARIABLES
                ),
                models_parameter=models_parameter,
                first_date=FIRST_DATE if year == first_year else f"{year}-01-01",
                last_date=LAST_DATE if year == LAST_YEAR else f"{year}-12-31",
                base_url=base_url,
                cell_selection="land",
            )
            for year in range(first_year, LAST_YEAR + 1)
        ).sort("site", "time")
        path = output_path_for(product=product)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
        _LOG.info("%s: %d rows for %d sites to %s", product, frame.height, sites.height, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
