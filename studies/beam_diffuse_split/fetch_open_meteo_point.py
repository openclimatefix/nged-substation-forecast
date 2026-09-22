"""Download one Open-Meteo forecast model's irradiance at each PV site, and check what arrived.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/800>.

**The script is parameterised by `--model` rather than written for UKV**, because Open-Meteo
normalises variable names across models: `shortwave_radiation`, `direct_radiation`, and
`diffuse_radiation` are the same request whichever model serves them, so only the `models=` value,
the output path, and the
native temporal convention differ. Those three live in `sources.OPEN_METEO_MODELS`, and a second
model is one entry there rather than a second script.

Requests go to the historical-forecast endpoint, which is a different service from the `archive-api`
endpoint `fetch_era5_open_meteo.py` uses for ERA5, with its own call-weight accounting. Coordinates
are read at run time from the private roster and sent in the query string. **No coordinate and no
identifier reaches the written frame**: rows are keyed by the anonymised site label
`build_dataset._pv_sites` assigns.

Two checks run over the downloaded frame before it is written, and each raises with its measured
number rather than printing for a human to read. What they establish, and what they deliberately do
not, is on each `_check_*` function.

Run it with `uv run --with netcdf4 python studies/beam_diffuse_split/fetch_open_meteo_point.py
--model ukv`. The extra dependencies are `build_dataset`'s, which this script imports the site
roster from.
"""

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Final

import numpy as np
import polars as pl
from build_dataset import _pv_sites
from era5_grid import LAST_DATE, LAST_YEAR
from sources import (
    HISTORICAL_FORECAST_URL,
    OPEN_METEO_MODELS,
    OpenMeteoModel,
    point_output_path_for,
)
from studies.served_column_checks import (
    check_direct_is_not_a_separation_model,
    check_hourly_value_is_a_backward_mean,
)
from studies.solar import (
    cos_zenith,
    cos_zenith_hour_mean,
    extraterrestrial_horizontal,
    zenith,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("fetch_open_meteo_point")

REQUEST_TIMEOUT_SECONDS: Final[float] = 600.0
MAX_ATTEMPTS: Final[int] = 5

HOURLY_VARIABLES: Final[tuple[str, ...]] = ("shortwave_radiation", "direct_radiation")
"""The two horizontal fluxes the arms consume, in Open-Meteo's normalised names.

The served `diffuse_radiation` is not requested. `build_dataset._add_separation_models` derives the
diffuse flux as global minus direct for every source, so a served diffuse column would reach no arm
whatever it held.
"""

INSTANT_SUFFIX: Final[str] = "_instant"
"""What Open-Meteo appends to ask for the snapshot behind a backward-looking hourly mean."""


def _requested_variables(*, model: OpenMeteoModel) -> tuple[str, ...]:
    """Return the `hourly=` variable list for one model.

    The `_instant` snapshots are worth their share of the call weight only where the model's own
    output is instantaneous, because that is the case where Open-Meteo's hourly value is a
    reconstruction rather than a de-accumulation, and the two columns then differ. An `accumulated`
    model returns without them, and `main` then skips the reconstruction check, which would have
    nothing to reconstruct against.

    Args:
        model: The registry entry being fetched.

    Returns:
        Every variable name to request.
    """
    if model.native_radiation != "instantaneous":
        return HOURLY_VARIABLES
    return HOURLY_VARIABLES + tuple(f"{name}{INSTANT_SUFFIX}" for name in HOURLY_VARIABLES)


def _first_date_of(*, model: OpenMeteoModel, year: int) -> str:
    """Return the first date to request in `year`, as `YYYY-MM-DD`."""
    archive_year = int(model.archive_starts[:4])
    return model.archive_starts if year == archive_year else f"{year}-01-01"


def _last_date_of(*, year: int) -> str:
    """Return the last date to request in `year`, as `YYYY-MM-DD`."""
    return LAST_DATE if year == LAST_YEAR else f"{year}-12-31"


def _get_json(*, url: str) -> Any:
    """Fetch one URL, retrying a transport failure but never an API refusal.

    **A rate-limit refusal must not be retried.** Open-Meteo answers an exceeded call budget with a
    JSON body carrying `error: true` and a `reason`, at an HTTP status a bare `except` reads as one
    more transient failure — so a retry loop copied from `fetch_era5_open_meteo.py` would sleep
    five times and then report a network problem that never happened.

    Args:
        url: The request to make.

    Returns:
        The decoded JSON body, which is a list when several coordinates were requested.

    Raises:
        RuntimeError: If the API refused the request, or every attempt failed.
    """
    for attempt in range(MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as refusal:
            # The body is read for its `reason` and never echoed whole: Open-Meteo's parameter
            # errors quote the offending value back, and the values here are meter coordinates.
            reason = json.loads(refusal.read() or b"{}").get("reason", "no reason given")
            msg = f"Open-Meteo refused the request with HTTP {refusal.code}: {reason}"
            raise RuntimeError(msg) from refusal
        # urllib raises several unrelated types for a transient failure, and a retry is the right
        # response to all of them.
        except Exception:  # noqa: BLE001
            _LOG.warning("attempt %d failed, retrying", attempt + 1)
            time.sleep(5.0 * (attempt + 1))
        else:
            if isinstance(payload, dict) and payload.get("error"):
                msg = f"Open-Meteo refused the request: {payload.get('reason')}"
                raise RuntimeError(msg)
            return payload
    msg = f"Open-Meteo failed after {MAX_ATTEMPTS} attempts"
    raise RuntimeError(msg)


def fetch_point_frame(
    *,
    sites: pl.DataFrame,
    variables: tuple[str, ...],
    models_parameter: str,
    first_date: str,
    last_date: str,
) -> pl.DataFrame:
    """Fetch one date range at every site's own coordinates, in a single request.

    Every site goes in one request, as `fetch_era5_open_meteo.py` does for the ERA5 grid cells.
    Open-Meteo returns the blocks in the order the coordinates were sent, so the anonymised labels
    are zipped back on by position rather than by matching coordinates, and no coordinate is
    carried past this function.

    Args:
        sites: The roster, carrying `site`, `latitude`, and `longitude`.
        variables: Open-Meteo's names for the hourly variables to request.
        models_parameter: The value of the API's `models=` query parameter.
        first_date: First date to request, as `YYYY-MM-DD`.
        last_date: Last date to request, as `YYYY-MM-DD`.

    Returns:
        One row per (site, time), with one column per requested variable and no null rows.

    Raises:
        RuntimeError: If the response does not carry one block per site.
    """
    url = (
        f"{HISTORICAL_FORECAST_URL}"
        f"?latitude={','.join(str(value) for value in sites['latitude'])}"
        f"&longitude={','.join(str(value) for value in sites['longitude'])}"
        f"&start_date={first_date}&end_date={last_date}"
        f"&hourly={','.join(variables)}"
        f"&models={models_parameter}&timezone=UTC&cell_selection=nearest"
    )
    payload = _get_json(url=url)
    blocks = payload if isinstance(payload, list) else [payload]
    if len(blocks) != sites.height:
        msg = f"asked for {sites.height} sites and got {len(blocks)} blocks"
        raise RuntimeError(msg)

    frame = (
        pl.concat(
            pl.DataFrame(
                {"site": site, "time": block["hourly"]["time"]}
                | {name: block["hourly"][name] for name in variables},
                schema_overrides=dict.fromkeys(variables, pl.Float64),
            )
            for site, block in zip(sites["site"], blocks, strict=True)
        )
        .drop_nulls()
        .with_columns(pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M").dt.replace_time_zone("UTC"))
    )
    # The block count above is checked before `drop_nulls`, so a meter the model answered for but
    # filled entirely with nulls survives that check and then vanishes here. A limited-area model
    # does exactly that outside its domain, and a five-meter frame would otherwise be scored
    # without anyone noticing which meter left.
    if frame["site"].n_unique() != sites.height:
        present = set(frame["site"].unique().to_list())
        missing = sorted(set(sites["site"].to_list()) - present)
        msg = f"{models_parameter} returned only nulls for {missing}; it may not cover them"
        raise RuntimeError(msg)
    return frame


def _renamed(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Rename Open-Meteo's variable names to the flux names the rest of the experiment uses."""
    mapping = {
        "shortwave_radiation": "ghi_w_m2",
        "direct_radiation": "bhi_w_m2",
        "shortwave_radiation_instant": "ghi_instant_w_m2",
        "direct_radiation_instant": "bhi_instant_w_m2",
    }
    return frame.rename({old: new for old, new in mapping.items() if old in frame.columns})


def _solar_geometry(*, frame: pl.DataFrame, sites: pl.DataFrame) -> pl.DataFrame:
    """Add the solar geometry the checks need, without writing a coordinate to the frame.

    Geometry is evaluated at each hour's midpoint, which is what `build_dataset` does and what the
    period-ending convention requires.

    Args:
        frame: The downloaded rows, keyed by `site` and `time`.
        sites: The roster, carrying `site`, `latitude`, and `longitude`.

    Returns:
        `frame` with `solar_zenith_deg`, `cos_zenith_instant`, `cos_zenith_hour_mean`,
        `clearness_index` and `extraterrestrial_horizontal_w_m2` added.
    """
    coordinates = {
        str(row["site"]): (float(row["latitude"]), float(row["longitude"]))
        for row in sites.to_dicts()
    }
    frames: list[pl.DataFrame] = []
    for (site,), rows in frame.sort("site", "time").group_by(["site"], maintain_order=True):
        latitude, longitude = coordinates[str(site)]
        stamps = rows["time"]
        midpoint_zenith = zenith(
            stamps=stamps.dt.offset_by("-30m"), latitude=latitude, longitude=longitude
        )
        cos_instant = cos_zenith(
            zenith_deg=zenith(stamps=stamps, latitude=latitude, longitude=longitude)
        )
        cos_hour_mean = cos_zenith_hour_mean(stamps=stamps, latitude=latitude, longitude=longitude)
        extraterrestrial = extraterrestrial_horizontal(stamps=stamps, zenith_deg=midpoint_zenith)
        frames.append(
            rows.with_columns(
                solar_zenith_deg=pl.Series(midpoint_zenith),
                cos_zenith_instant=pl.Series(cos_instant),
                cos_zenith_hour_mean=pl.Series(cos_hour_mean),
                clearness_index=pl.Series(
                    np.where(
                        extraterrestrial > 0.0,
                        rows["ghi_w_m2"].to_numpy() / np.maximum(extraterrestrial, 1e-9),
                        0.0,
                    )
                ),
                extraterrestrial_horizontal_w_m2=pl.Series(extraterrestrial),
            )
        )
    return pl.concat(frames)


def main() -> int:
    """Download one model at every site, run both checks, and write the frame."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(OPEN_METEO_MODELS), required=True)
    arguments = parser.parse_args()
    model = OPEN_METEO_MODELS[arguments.model]
    if model.native_radiation == "unmeasured":
        msg = (
            f"{model.source}'s native temporal convention is unmeasured, so what its hourly column "
            "holds is unknown. Establish it and record it in sources.OPEN_METEO_MODELS first."
        )
        raise ValueError(msg)

    sites = _pv_sites()
    first_year = int(model.archive_starts[:4])
    _LOG.info(
        "fetching %s for %d sites, %s to %s, as %d requests",
        model.models_parameter,
        sites.height,
        model.archive_starts,
        LAST_DATE,
        LAST_YEAR - first_year + 1,
    )

    variables = _requested_variables(model=model)
    frame = _renamed(
        frame=pl.concat(
            fetch_point_frame(
                sites=sites,
                variables=variables,
                models_parameter=model.models_parameter,
                first_date=_first_date_of(model=model, year=year),
                last_date=_last_date_of(year=year),
            )
            for year in range(first_year, LAST_YEAR + 1)
        )
        .unique(subset=["site", "time"], keep="first")
        .sort("site", "time")
    )

    with_geometry = _solar_geometry(frame=frame, sites=sites)
    if model.native_radiation == "instantaneous":
        check_hourly_value_is_a_backward_mean(frame=with_geometry)
    check_direct_is_not_a_separation_model(frame=with_geometry)

    output_path = point_output_path_for(source=model.source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path)
    _LOG.info(
        "wrote %d rows covering %s to %s for %d sites to %s",
        frame.height,
        frame["time"].min(),
        frame["time"].max(),
        frame["site"].n_unique(),
        output_path,
    )
    if model.live_ingest_starts is not None:
        backfilled = frame.filter(pl.col("time").dt.date().cast(pl.Utf8) < model.live_ingest_starts)
        _LOG.info(
            "%d of %d rows predate %s, when Open-Meteo's downloader for this model first existed, "
            "and were backfilled from a source it does not name",
            backfilled.height,
            frame.height,
            model.live_ingest_starts,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
