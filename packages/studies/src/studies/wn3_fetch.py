"""The names of the WeatherNext 3 Icechunk store that the fetch and the input builder share.

`studies/weather_downloads/fetch_weathernext3.py` writes the store and
`studies/nwp_forecast_comparison/build_wn3_inputs.py` reads it, so both take the store's location
and its array names from here.
"""

from typing import Final

STORE_PREFIX: Final[str] = "weathernext3_statistics_uk"
"""The directory of the Icechunk repository inside the output bucket."""


MAIN_BRANCH: Final[str] = "main"
"""The branch colleagues read. Only `validate_weathernext3.py --publish` moves it."""


INIT_TIME: Final[str] = "init_time"
LEAD_TIME: Final[str] = "lead_time"
LATITUDE: Final[str] = "latitude"
LONGITUDE: Final[str] = "longitude"
RUN_WRITTEN: Final[str] = "run_written"
SOURCE_INIT_TIME: Final[str] = "source_init_time"
"""Array names in the output. The last two hold one value per `init_time` slot."""
