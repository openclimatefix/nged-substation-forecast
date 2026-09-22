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

Run it with `uv run --no-project --with polars --with numpy --with pvlib --with deltalake --with
xarray --with netcdf4 python scripts/experiments/beam_diffuse_split/fetch_open_meteo_point.py
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
import pvlib
from build_dataset import SOLAR_CONSTANT_W_M2, _pv_sites
from era5_grid import LAST_DATE, LAST_YEAR
from sources import (
    HISTORICAL_FORECAST_URL,
    OPEN_METEO_MODELS,
    OpenMeteoModel,
    point_output_path_for,
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
        midpoint_zenith = _zenith(
            stamps=stamps.dt.offset_by("-30m"), latitude=latitude, longitude=longitude
        )
        cos_instant = np.clip(
            np.cos(np.radians(_zenith(stamps=stamps, latitude=latitude, longitude=longitude))),
            0.0,
            None,
        )
        cos_hour_mean = _cos_zenith_hour_mean(stamps=stamps, latitude=latitude, longitude=longitude)
        extraterrestrial = np.asarray(
            pvlib.irradiance.get_extra_radiation(
                datetime_or_doy=stamps.dt.ordinal_day().to_numpy(),
                solar_constant=SOLAR_CONSTANT_W_M2,
            )
        ) * np.clip(np.cos(np.radians(midpoint_zenith)), 0.0, None)
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


def _zenith(*, stamps: pl.Series, latitude: float, longitude: float) -> np.ndarray:
    """Return the apparent solar zenith angle in degrees at each stamp."""
    position = pvlib.solarposition.get_solarposition(
        time=stamps.to_numpy(), latitude=latitude, longitude=longitude
    )
    return position["apparent_zenith"].to_numpy().astype(np.float64)


COS_ZENITH_SUBSAMPLES: Final[int] = 60
"""How many samples the mean cosine of the solar zenith angle over an hour is taken from.

One sample a minute. The quantity is smooth in time except at sunrise and sunset, where the clip
at zero puts a corner in it, so the error a minute's spacing leaves is far below the 1 W m⁻²
rounding of the column the check compares against.
"""


def _cos_zenith_hour_mean(*, stamps: pl.Series, latitude: float, longitude: float) -> np.ndarray:
    """Return the mean cosine of the solar zenith angle over the hour *ending* at each stamp."""
    minutes = np.arange(COS_ZENITH_SUBSAMPLES) + 0.5 - COS_ZENITH_SUBSAMPLES
    samples = [
        np.clip(
            np.cos(
                np.radians(
                    _zenith(
                        stamps=stamps.dt.offset_by(f"{int(offset)}s"),
                        latitude=latitude,
                        longitude=longitude,
                    )
                )
            ),
            0.0,
            None,
        )
        for offset in np.round(minutes * 60.0)
    ]
    return np.mean(samples, axis=0)


MIN_ELEVATION_FOR_RECONSTRUCTION_DEGREES: Final[float] = 10.0
"""Rows at a lower sun than this are left out of the reconstruction test.

The reconstruction multiplies the default column by the ratio of the instantaneous to the hour-mean
cosine of the solar zenith angle, and that ratio grows without bound as the sun sets — so near the
horizon it amplifies the default column's 1 W m⁻² rounding into a residual of many W m⁻². That
residual is a fact about rounding rather than about which hour the label names, and leaving it in
would force a threshold loose enough to stop discriminating.
"""

MAX_INSTANT_RECONSTRUCTION_RMS_W_M2: Final[float] = 5.0
"""How far the reconstructed snapshot may sit from the served one.

Measured at one meter's coordinates over June 2026, the reconstruction lands at a root-mean-square
error of 0.65 W m⁻² for the global flux and 0.14 for the direct flux — the scale of the 1 W m⁻²
rounding the default columns carry. The threshold leaves room for other sites and seasons while
staying well inside the failures it exists to catch: assuming the hour *beginning* at the label
gives 68 W m⁻² and assuming an hour centred on it gives 26, so a half-hour error in either
direction is at least five times the threshold rather than a marginal call.
"""


def _check_hourly_value_is_a_backward_mean(*, frame: pl.DataFrame) -> None:
    """Assert the hourly column is a backward mean over the hour ending at its label.

    **The check is absolute rather than relative:** it needs no reference product, and cannot be
    satisfied by two sources being wrong the same way. A half-hour error in an hourly label is the
    fault it exists to catch.

    Open-Meteo's own downloader states the mechanism. UKV publishes radiation as an instantaneous
    snapshot, and Open-Meteo divides that snapshot by the ratio of the instantaneous cosine of the
    solar zenith angle to its mean over the preceding hour before storing it, so the stored value is
    a backward-looking hourly mean holding the clear-sky index fixed across the hour. Asking for the
    `_instant` column multiplies the same ratio back. Reconstructing one column from the other and
    the sun's geometry therefore pins both the conversion and which hour the label names.

    What the check settles is which of the two served columns the arms should read, which
    `sources.PointTemporalType` records and explains.

    Args:
        frame: The downloaded rows, carrying the geometry `_solar_geometry` adds.

    Raises:
        ValueError: If either flux fails to reconstruct.
    """
    daylight = frame.filter(
        (pl.col("ghi_w_m2") > 0.0)
        & (pl.col("solar_zenith_deg") < 90.0 - MIN_ELEVATION_FOR_RECONSTRUCTION_DEGREES)
    )
    factor = daylight["cos_zenith_instant"].to_numpy() / np.maximum(
        daylight["cos_zenith_hour_mean"].to_numpy(), 1e-9
    )
    for default_column, instant_column in (
        ("ghi_w_m2", "ghi_instant_w_m2"),
        ("bhi_w_m2", "bhi_instant_w_m2"),
    ):
        error = daylight[default_column].to_numpy() * factor - daylight[instant_column].to_numpy()
        root_mean_square = float(np.sqrt(np.mean(error**2)))
        _LOG.info(
            "backward-mean reconstruction of %s: RMS %.3f W m-2 on %d daylight rows",
            default_column,
            root_mean_square,
            daylight.height,
        )
        if root_mean_square > MAX_INSTANT_RECONSTRUCTION_RMS_W_M2:
            msg = (
                f"{default_column} does not reconstruct {instant_column} from the hour ending at "
                f"its label: RMS {root_mean_square:.2f} W m-2 against a threshold of "
                f"{MAX_INSTANT_RECONSTRUCTION_RMS_W_M2}. Either the label names a different hour "
                "or Open-Meteo has changed the conversion; settle which before training on it."
            )
            raise ValueError(msg)


CLEARNESS_BIN_WIDTH: Final[float] = 0.05
ZENITH_BIN_WIDTH_DEGREES: Final[float] = 5.0
MIN_ROWS_PER_BIN: Final[int] = 30
MIN_DIRECT_FRACTION_SPREAD: Final[float] = 0.05
"""How much the published direct fraction must vary inside one `(clearness, zenith)` bin.

A separation model's direct fraction is by construction a function of the clearness index and the
solar zenith angle, so inside a fine bin on those two it is very nearly constant whatever formula it
uses, and the spread it leaves comes only from the bin's own width. Measured over 2025 at one
meter's coordinates on these bin widths, the median within-bin spread is 0.118 for UKV's published
fraction and
0.016 for an Erbs fraction derived from the same global irradiance — so the threshold sits three
times above the separation-model floor and well below what UKV's own field gave.
"""

MIN_ZENITH_FOR_SPREAD_DEGREES: Final[float] = 80.0
"""Rows at a lower sun than this are left out of the spread test.

The clearness index is a ratio taken against a small extraterrestrial flux there, so it is mostly
noise, and binning on a noisy axis would inflate the within-bin spread of anything.
"""


def _check_direct_is_not_a_separation_model(*, frame: pl.DataFrame) -> None:
    """Assert the published direct fraction carries information beyond clearness and geometry.

    **This is the one check that reaches the era no sampling against the Met Office's own files
    can**, because it needs no reference product: Open-Meteo's UKV archive runs from 2022-03-01 and
    the Met Office's AWS bucket holds a rolling two years, so the earlier half can only be checked
    from the inside.

    What it would catch is the failure that would void arm C outright — a mirror that reconstructed
    the beam from global irradiance with a separation model rather than serving the model's own
    field. Arm C would then be a copy of arm B, and the headline contrast would be a measurement of
    floating-point noise.

    Args:
        frame: The downloaded rows, carrying the geometry `_solar_geometry` adds.

    Raises:
        ValueError: If the within-bin spread sits at the separation-model floor.
    """
    binned = (
        frame.filter(
            (pl.col("solar_zenith_deg") < MIN_ZENITH_FOR_SPREAD_DEGREES)
            & (pl.col("ghi_w_m2") > 20.0)
            & (pl.col("extraterrestrial_horizontal_w_m2") > 50.0)
        )
        .with_columns(
            direct_fraction=(pl.col("bhi_w_m2") / pl.col("ghi_w_m2")).clip(0.0, 1.0),
            clearness_bin=(pl.col("clearness_index") / CLEARNESS_BIN_WIDTH).floor(),
            zenith_bin=(pl.col("solar_zenith_deg") / ZENITH_BIN_WIDTH_DEGREES).floor(),
        )
        .group_by("clearness_bin", "zenith_bin")
        .agg(spread=pl.col("direct_fraction").std(ddof=0), rows=pl.len())
        .filter(pl.col("rows") >= MIN_ROWS_PER_BIN)
    )
    if binned.height == 0:
        msg = (
            f"no (clearness, zenith) bin holds {MIN_ROWS_PER_BIN} rows, so the spread test "
            "cannot run"
        )
        raise ValueError(msg)
    median_spread = binned.select(pl.col("spread").median()).item()
    _LOG.info(
        "direct-fraction spread: median %.4f within %d populated bins",
        median_spread,
        binned.height,
    )
    if median_spread < MIN_DIRECT_FRACTION_SPREAD:
        msg = (
            f"the published direct fraction varies by only {median_spread:.4f} inside a "
            f"(clearness, zenith) bin, against a threshold of {MIN_DIRECT_FRACTION_SPREAD}. That "
            "is what a separation model applied to global irradiance looks like, and it would make "
            "arm C a copy of arm B."
        )
        raise ValueError(msg)


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
        _check_hourly_value_is_a_backward_mean(frame=with_geometry)
    _check_direct_is_not_a_separation_model(frame=with_geometry)

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
