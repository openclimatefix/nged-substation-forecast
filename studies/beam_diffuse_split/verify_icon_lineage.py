"""Check Open-Meteo's ICON archive against DWD's own files, and say which run it serves each hour.

One-off throwaway script for the study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/809>.

**The served lead of an ICON product decides how a comparison with UKV's T+0 analysis reads.**
Open-Meteo ingests every run and a later run overwrites an earlier one for the same valid time, so
the archive should hold, for each hour, the freshest run that covers it. That is an argument, and
this script measures it: for every daytime hour it reconstructs the hourly mean from every run DWD
still serves, and reports which run matches what Open-Meteo served.

**The check covers the runs DWD still publishes, about one day.** That is enough to establish the
mapping from valid hour to run, which is what the served lead follows from; it cannot speak to
whether the archive's earlier years were built the same way.

DWD publishes `ASWDIR_S` and `ASWDIFD_S` as means since the run started, so an hour's mean is
recovered by differencing consecutive steps weighted by their lead. Global horizontal irradiance is
the sum of the direct and diffuse downward components, which is what Open-Meteo serves as
`shortwave_radiation`. ICON global is published by DWD only on its icosahedral grid, so it is not
covered here.

The coordinate is a public place, never a metered generator's.

Run it with `uv run --with cfgrib python studies/beam_diffuse_split/verify_icon_lineage.py --model
icon-eu`.
"""

import argparse
import bz2
import datetime as dt
import logging
import re
import sys
import tempfile
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np
import requests
import xarray as xr

_LOG = logging.getLogger(__name__)

PROBE_LATITUDE: Final[float] = 52.95
PROBE_LONGITUDE: Final[float] = -1.15
"""A public place inside every ICON domain here, chosen so no metered generator's location is
written."""

OPEN_METEO_URL: Final[str] = "https://historical-forecast-api.open-meteo.com/v1/forecast"

COMPONENTS: Final[tuple[str, ...]] = ("aswdir_s", "aswdifd_s")
"""The two downward short-wave components whose sum is global horizontal irradiance."""

DAYTIME_HOURS: Final[tuple[int, int]] = (8, 16)
"""The first and last UTC label hours compared, where the signal is large enough to tell runs
apart."""

MAX_LEAD_HOURS: Final[int] = 9
"""The longest lead reconstructed for a valid hour, which reaches three 3-hourly runs back."""

REQUEST_TIMEOUT: Final[int] = 120


class IconModel(NamedTuple):
    """Where one ICON product's native files live, and how Open-Meteo names it.

    Attributes:
        dwd_root: The product's GRIB directory on DWD's open-data server.
        file_template: The file name, with `{stamp}`, `{lead}` and `{component}` to fill.
        upper_case_component: Whether DWD writes the component name in upper case.
        open_meteo_model: The value of Open-Meteo's `models=` parameter.
    """

    dwd_root: str
    file_template: str
    upper_case_component: bool
    open_meteo_model: str


MODELS: Final[dict[str, IconModel]] = {
    "icon-d2": IconModel(
        dwd_root="https://opendata.dwd.de/weather/nwp/icon-d2/grib",
        file_template=(
            "icon-d2_germany_regular-lat-lon_single-level_{stamp}_{lead:03d}_2d_{component}"
            ".grib2.bz2"
        ),
        upper_case_component=False,
        open_meteo_model="icon_d2",
    ),
    "icon-eu": IconModel(
        dwd_root="https://opendata.dwd.de/weather/nwp/icon-eu/grib",
        file_template=(
            "icon-eu_europe_regular-lat-lon_single-level_{stamp}_{lead:03d}_{component}.grib2.bz2"
        ),
        upper_case_component=True,
        open_meteo_model="icon_eu",
    ),
}
"""Every ICON product this script can check, keyed by its `--model` name."""


def _latest_runs(*, model: IconModel) -> list[dt.datetime]:
    """Return the run DWD currently holds for each 3-hourly slot, oldest first.

    Args:
        model: The product.

    Returns:
        Each run's initialisation time, in UTC.
    """
    runs: list[dt.datetime] = []
    for slot in range(0, 24, 3):
        listing = requests.get(
            f"{model.dwd_root}/{slot:02d}/{COMPONENTS[0]}/", timeout=REQUEST_TIMEOUT
        ).text
        stamps = set(re.findall(r"_(\d{10})_\d{3}_", listing))
        runs += [dt.datetime.strptime(stamp, "%Y%m%d%H").replace(tzinfo=dt.UTC) for stamp in stamps]
    return sorted(runs)


def _sample(*, model: IconModel, run: dt.datetime, lead: int, component: str) -> float | None:
    """Download one field and read it at the probe coordinate.

    Args:
        model: The product.
        run: The run's initialisation time, in UTC.
        lead: The lead in whole hours.
        component: One of `COMPONENTS`.

    Returns:
        The value in W m⁻², or `None` if DWD no longer serves that file.
    """
    name = model.file_template.format(
        stamp=run.strftime("%Y%m%d%H"),
        lead=lead,
        component=component.upper() if model.upper_case_component else component,
    )
    response = requests.get(
        f"{model.dwd_root}/{run:%H}/{component}/{name}", timeout=REQUEST_TIMEOUT
    )
    if response.status_code != 200:
        return None
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as handle:
        handle.write(bz2.decompress(response.content))
        path = Path(handle.name)
    try:
        with xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""}) as data:
            variable = next(iter(data.data_vars))
            point = data[variable].sel(
                latitude=PROBE_LATITUDE, longitude=PROBE_LONGITUDE, method="nearest"
            )
            # An ICON-D2 file bundles the four 15-minute steps inside its hour, so the whole hour
            # has to be picked out rather than taken as the file's only value.
            if "step" in point.dims:
                point = point.sel(step=np.timedelta64(lead, "h"))
            return float(point.to_numpy())
    finally:
        path.unlink(missing_ok=True)


def _hourly_mean(*, model: IconModel, run: dt.datetime, lead: int) -> float | None:
    """Recover one hour's mean global horizontal irradiance from two consecutive steps.

    DWD averages these fields from the run's start, so the mean over the hour ending at `lead` is
    `lead * A(lead) - (lead - 1) * A(lead - 1)`, summed over the two components.

    Args:
        model: The product.
        run: The run's initialisation time, in UTC.
        lead: The lead in whole hours of the hour's end.

    Returns:
        The hourly mean in W m⁻², or `None` if either step is missing.
    """
    total = 0.0
    for component in COMPONENTS:
        now = _sample(model=model, run=run, lead=lead, component=component)
        before = _sample(model=model, run=run, lead=lead - 1, component=component)
        if now is None or before is None:
            return None
        total += lead * now - (lead - 1) * before
    return total


def _open_meteo(*, model: IconModel, first: dt.date, last: dt.date) -> dict[dt.datetime, float]:
    """Read Open-Meteo's served hourly irradiance over a span of days.

    Args:
        model: The product.
        first: The first day to read, in UTC.
        last: The last day to read, in UTC.

    Returns:
        Each served hour, as its end, to the served value in W m⁻².

    Raises:
        RuntimeError: If the response carries no hourly block.
    """
    payload = requests.get(
        OPEN_METEO_URL,
        params={
            "latitude": PROBE_LATITUDE,
            "longitude": PROBE_LONGITUDE,
            "start_date": first.isoformat(),
            "end_date": last.isoformat(),
            "hourly": "shortwave_radiation",
            "models": model.open_meteo_model,
            "timezone": "UTC",
        },
        timeout=REQUEST_TIMEOUT,
    ).json()
    if "hourly" not in payload:
        msg = f"Open-Meteo returned no hourly block: {payload}"
        raise RuntimeError(msg)
    hourly = payload["hourly"]
    return {
        dt.datetime.fromisoformat(stamp).replace(tzinfo=dt.UTC): value
        for stamp, value in zip(hourly["time"], hourly["shortwave_radiation"], strict=True)
        if value is not None
    }


def main() -> int:
    """For each daytime hour, find the run whose reconstruction matches what Open-Meteo served."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    model = MODELS[parser.parse_args().model]

    runs = _latest_runs(model=model)
    _LOG.info("DWD holds runs %s", [f"{run:%d %H}Z" for run in runs])
    served = _open_meteo(model=model, first=runs[0].date(), last=runs[-1].date())
    first_hour, last_hour = DAYTIME_HOURS

    print("\n| Valid hour (UTC) | Served (W m⁻²) | Run: lead (h), difference (W m⁻²) | Best run |")
    print("|---|---|---|---|")
    freshest_is_best = 0
    compared = 0
    for valid in sorted(served):
        if not first_hour <= valid.hour <= last_hour:
            continue
        candidates = [
            run for run in runs if 1 <= (valid - run).total_seconds() / 3600 <= MAX_LEAD_HOURS
        ]
        differences: dict[dt.datetime, float] = {}
        for run in candidates:
            lead = int((valid - run).total_seconds() // 3600)
            native = _hourly_mean(model=model, run=run, lead=lead)
            if native is not None:
                differences[run] = native - served[valid]
        if len(differences) < 2:
            continue
        best = min(differences, key=lambda run: abs(differences[run]))
        compared += 1
        freshest_is_best += best == max(differences)
        cells = ", ".join(
            f"{run:%H}Z: {int((valid - run).total_seconds() // 3600)}, {difference:+.1f}"
            for run, difference in sorted(differences.items())
        )
        print(
            f"| {valid:%Y-%m-%d %H}:00 | {served[valid]:.1f} | {cells} | {best:%H}Z, lead "
            f"{int((valid - best).total_seconds() // 3600)} |"
        )

    print(
        f"\nThe freshest available run was the best match at {freshest_is_best} of {compared} "
        "hours."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
