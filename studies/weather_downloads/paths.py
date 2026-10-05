"""Where every download for issue #841 lands, and the command that writes the trial-area box.

One-off throwaway module for the downloads in
<https://github.com/openclimatefix/nged-substation-forecast/issues/841>, which feed the two
past-weather studies (#809) and the forecast study (#810). It takes `REPO_DATA_DIR` from
`studies.sources`, the one place that resolves the `data/` directory, and the box itself lives in
`studies.trial_area`. Running this file derives the box from the private roster.
"""

import os
from pathlib import Path
from typing import Final

from studies.sources import REPO_DATA_DIR
from studies.trial_area import write_trial_area_box_from_roster

WEATHER_DOWNLOADS_DIR: Final[Path] = REPO_DATA_DIR / "studies" / "weather"
"""One subdirectory per product, e.g. `ECMWF-IFS-HRES`, `NORA3`, `ICON-DREAM-EU`."""


def open_meteo_api_key() -> str | None:
    """Return the Open-Meteo commercial API key from the environment, or `None` if unset.

    The key is read from `OPEN_METEO_TOKEN`, then from `OPEN_METEO_API_KEY`. Both are exported from
    `~/.bashrc`; neither belongs in `.env`, because Dagster reads that file.

    A caller with a key switches from the free `<name>-api.open-meteo.com` host to
    `customer-<name>-api.open-meteo.com` and appends `&apikey=<key>` to the request, which lifts
    the free tier's daily/hourly/minutely rate limits entirely (confirmed against both the
    Previous Runs and Historical Forecast customer hosts). Never log or print the returned value.
    """
    return os.environ.get("OPEN_METEO_TOKEN") or os.environ.get("OPEN_METEO_API_KEY")


if __name__ == "__main__":
    write_trial_area_box_from_roster()
