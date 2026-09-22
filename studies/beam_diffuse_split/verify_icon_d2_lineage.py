"""Check Open-Meteo's ICON-D2 mirror against DWD's own files, and say which lead it serves.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/809>.

**The write-ups call ICON-D2 an analysis, and nothing had measured that.** `verify_ukv_lineage.py`
settles the question for UKV against the Met Office's own files; ICON-D2 was added later and its
lead was assumed from the same argument - Open-Meteo ingests every run and a later run overwrites
an earlier one for the same valid time - without being checked. ICON-D2 runs every 3 hours rather
than hourly, and its radiation is accumulated rather than instantaneous, so the argument does not
transfer and the shortest lead the archive can hold is a 1-to-3-hour forecast rather than T+0.

**The check is a single day, because DWD keeps only about a day of runs.** That is enough to
separate a short lead from a long one, which is the question, and it cannot speak to whether the
archive's earlier years were built the same way.

DWD publishes `ASWDIR_S` and `ASWDIFD_S` as means since the run started, so an hour's mean is
recovered by differencing consecutive steps weighted by their lead. Global horizontal irradiance is
the sum of the direct and diffuse downward components, which is what Open-Meteo serves as
`shortwave_radiation`.

The coordinate is a public place, never a metered generator's.

Run it with `uv run --no-project --with cfgrib --with xarray --with numpy --with requests python
studies/beam_diffuse_split/verify_icon_d2_lineage.py`.
"""

import bz2
import datetime as dt
import logging
import sys
import tempfile
from pathlib import Path
from typing import Final

import numpy as np
import requests
import xarray as xr

_LOG = logging.getLogger(__name__)

PROBE_LATITUDE: Final[float] = 52.95
PROBE_LONGITUDE: Final[float] = -1.15
"""A public place inside ICON-D2's domain, chosen so no metered generator's location is written."""

DWD_ROOT: Final[str] = "https://opendata.dwd.de/weather/nwp/icon-d2/grib"
OPEN_METEO_URL: Final[str] = "https://historical-forecast-api.open-meteo.com/v1/forecast"

COMPONENTS: Final[tuple[str, ...]] = ("aswdir_s", "aswdifd_s")
"""The two downward short-wave components whose sum is global horizontal irradiance."""

VALID_HOURS: Final[tuple[int, ...]] = (11, 12, 13, 14)
"""The valid hours compared, chosen in the middle of the day where the signal is largest."""

REQUEST_TIMEOUT: Final[int] = 120


def _grib_url(*, run: dt.datetime, lead: int, component: str) -> str:
    """Return the URL of one regular-latitude-longitude field.

    Args:
        run: The run's initialisation time, in UTC.
        lead: The lead in whole hours.
        component: One of `COMPONENTS`.

    Returns:
        The address of the bzip2-compressed GRIB2 file.
    """
    stamp = run.strftime("%Y%m%d%H")
    name = (
        f"icon-d2_germany_regular-lat-lon_single-level_{stamp}_{lead:03d}_2d_{component}.grib2.bz2"
    )
    return f"{DWD_ROOT}/{run:%H}/{component}/{name}"


def _sample(*, run: dt.datetime, lead: int, component: str) -> float | None:
    """Download one field and read it at the probe coordinate.

    Args:
        run: The run's initialisation time, in UTC.
        lead: The lead in whole hours.
        component: One of `COMPONENTS`.

    Returns:
        The value in W m⁻², or `None` if DWD no longer serves that file.
    """
    response = requests.get(
        _grib_url(run=run, lead=lead, component=component), timeout=REQUEST_TIMEOUT
    )
    if response.status_code != 200:
        return None
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as handle:
        handle.write(bz2.decompress(response.content))
        path = Path(handle.name)
    try:
        with xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""}) as data:
            variable = next(iter(data.data_vars))
            # One file bundles the four 15-minute steps inside its hour, so the whole hour has to
            # be picked out rather than taken as the file's only value.
            point = data[variable].sel(
                latitude=PROBE_LATITUDE, longitude=PROBE_LONGITUDE, method="nearest"
            )
            if "step" in point.dims:
                point = point.sel(step=np.timedelta64(lead, "h"))
            return float(point.to_numpy())
    finally:
        path.unlink(missing_ok=True)


def _hourly_mean(*, run: dt.datetime, lead: int) -> float | None:
    """Recover one hour's mean global horizontal irradiance from two consecutive steps.

    DWD averages these fields from the run's start, so the mean over the hour ending at `lead` is
    `lead * A(lead) - (lead - 1) * A(lead - 1)`, summed over the two components.

    Args:
        run: The run's initialisation time, in UTC.
        lead: The lead in whole hours of the hour's end.

    Returns:
        The hourly mean in W m⁻², or `None` if either step is missing.
    """
    total = 0.0
    for component in COMPONENTS:
        now = _sample(run=run, lead=lead, component=component)
        before = _sample(run=run, lead=lead - 1, component=component)
        if now is None or before is None:
            return None
        total += lead * now - (lead - 1) * before
    return total


def _open_meteo(*, day: dt.date) -> dict[int, float]:
    """Read Open-Meteo's served ICON-D2 hourly irradiance for one day.

    Args:
        day: The day to read, in UTC.

    Returns:
        Hour of day to the served value in W m⁻².

    Raises:
        RuntimeError: If the response carries no hourly block.
    """
    response = requests.get(
        OPEN_METEO_URL,
        params={
            "latitude": PROBE_LATITUDE,
            "longitude": PROBE_LONGITUDE,
            "start_date": day.isoformat(),
            "end_date": day.isoformat(),
            "hourly": "shortwave_radiation",
            "models": "icon_d2",
            "timezone": "UTC",
        },
        timeout=REQUEST_TIMEOUT,
    )
    payload = response.json()
    if "hourly" not in payload:
        msg = f"Open-Meteo returned no hourly block: {payload}"
        raise RuntimeError(msg)
    hourly = payload["hourly"]
    return {
        dt.datetime.fromisoformat(stamp).hour: value
        for stamp, value in zip(hourly["time"], hourly["shortwave_radiation"], strict=True)
        if value is not None
    }


def main() -> int:
    """Compare every available run against Open-Meteo, and name the best-matching lead."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    day = dt.datetime.now(tz=dt.UTC).date()
    served = _open_meteo(day=day)
    _LOG.info("Open-Meteo served %d hours for %s", len(served), day)

    rows: list[tuple[str, int, float, int]] = []
    for run_hour in (0, 3, 6, 9):
        run = dt.datetime.combine(day, dt.time(run_hour), tzinfo=dt.UTC)
        errors: list[float] = []
        leads: list[int] = []
        for hour in VALID_HOURS:
            lead = hour - run_hour
            if lead < 1 or hour not in served:
                continue
            native = _hourly_mean(run=run, lead=lead)
            if native is None:
                continue
            errors.append(native - served[hour])
            leads.append(lead)
            _LOG.info(
                "run %02dZ lead %2d valid %02d:00  DWD %8.2f  Open-Meteo %8.2f  diff %8.2f",
                run_hour,
                lead,
                hour,
                native,
                served[hour],
                native - served[hour],
            )
        if errors:
            rows.append(
                (
                    f"{run_hour:02d}Z",
                    int(np.mean(leads)),
                    float(np.sqrt(np.mean(np.square(errors)))),
                    len(errors),
                )
            )

    if not rows:
        _LOG.error("no run could be compared; DWD may have rotated its files")
        return 1

    print("\n| Run | Mean lead (h) | RMSE vs Open-Meteo (W m⁻²) | Hours compared |")
    print("|---|---|---|---|")
    for label, lead, rmse, count in rows:
        print(f"| {label} | {lead} | {rmse:.2f} | {count} |")
    best = min(rows, key=lambda row: row[2])
    print(
        f"\nBest match: the {best[0]} run at a mean lead of {best[1]} hours, "
        f"RMSE {best[2]:.2f} W m⁻²."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
