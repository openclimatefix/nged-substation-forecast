"""Download one Open-Meteo forecast model over a grid of points inside the trial-area box.

One-off throwaway script for the downloads in
<https://github.com/openclimatefix/nged-substation-forecast/issues/841>. Covers the three
Open-Meteo-served products the issue asks for first: ECMWF IFS HRES 9 km, DMI and KNMI
HARMONIE-AROME, and Meteo-France ARPEGE Europe. A further Open-Meteo model needs only a new entry
in `MODELS` below. Each entry names its own `hourly_variables` — every model defaults to
`RADIATION_VARIABLES`, and `ecmwf-ifs-hres` also carries `WIND_VARIABLES` for the wind-products
study, so re-fetching one model never silently starts requesting a variable that has not been
validated for it.

The trial-area box is never printed, logged, or written into the output: only the grid points'
`point_id` (a running index) travels into filenames and frames, exactly as
`studies/beam_diffuse_split/fetch_open_meteo_point.py` keys its per-site output on an anonymised
label rather than a coordinate.

Run it with `uv run python studies/weather_downloads/fetch_open_meteo_grid.py --model ecmwf-ifs-hres
--start-date 2024-08-01 --end-date 2024-08-31`.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import polars as pl
from lineage import write_lineage_note, write_readme
from paths import WEATHER_DOWNLOADS_DIR, load_trial_area_box, open_meteo_api_key

HISTORICAL_FORECAST_URL: Final[str] = (
    "https://customer-historical-forecast-api.open-meteo.com/v1/forecast"
    if open_meteo_api_key()
    else "https://historical-forecast-api.open-meteo.com/v1/forecast"
)
"""The commercial host once `OPEN_METEO_API_KEY` is set (see `paths.open_meteo_api_key`), which
lifts the free tier's daily/hourly/minutely rate limits entirely; the free host otherwise."""
# Printed at import time (never the key itself): `open_meteo_api_key()` reads `.env` from
# `PROJECT_ROOT` of whichever checkout this script runs in, so a silent fall-back to the free host
# and its rate limits is otherwise easy to miss when running from a checkout other than the one
# holding the key.
print(
    f"Using {'the commercial' if open_meteo_api_key() else 'the free'} "
    f"Open-Meteo host: {HISTORICAL_FORECAST_URL}",
    file=sys.stderr,
)
REQUEST_TIMEOUT_SECONDS: Final[float] = 600.0
MAX_ATTEMPTS: Final[int] = 5
RADIATION_VARIABLES: Final[tuple[str, ...]] = ("shortwave_radiation", "direct_radiation")
"""The default `hourly=` set: every model in `MODELS` gets these unless it names its own."""
WIND_VARIABLES: Final[tuple[str, ...]] = ("wind_speed_10m", "wind_speed_100m")
"""Added to `ecmwf-ifs-hres`'s own set below, for the wind-products study
(<https://github.com/openclimatefix/nged-substation-forecast/issues/841>). Kept off every other
model here: each one's `hourly_variables` is what was actually requested and validated for that
product, so giving a second model wind it has not been checked for would happen silently the next
time this script re-fetches it."""
GRID_SPACING_DEG: Final[float] = 0.05
"""Spacing used for every product here, chosen to land in the "a few dozen points" the issue
describes across the box, rather than resolving each product at its own native grid spacing. Read
from this one constant everywhere — the request, the lineage note, and the README all derive from
it, so it can never drift out of sync with what was actually fetched."""

POINT_ID_COLUMN_DESCRIPTION: Final[str] = (
    "Running index (0-based) into the trial-area box's regular lat/lon grid at "
    "`GRID_SPACING_DEG` spacing, in request order — not a coordinate, and not guaranteed to be a "
    "distinct model grid cell (adjacent points can share a cell where the model's own resolution "
    "is coarser than the grid spacing)."
)

VARIABLE_COLUMN_DESCRIPTIONS: Final[dict[str, str]] = {
    "shortwave_radiation": "Global horizontal irradiance, W/m^2.",
    "direct_radiation": "Direct (beam) horizontal irradiance, W/m^2.",
    "wind_speed_10m": "10 m wind speed, km/h (Open-Meteo's default unit), instantaneous at `time`.",
    "wind_speed_100m": "100 m wind speed, km/h (Open-Meteo's default unit), instantaneous at "
    "`time`.",
}
"""README column description for each variable name that can appear in a model's
`hourly_variables`. One entry per name `MODELS` can reference."""


@dataclass(frozen=True)
class OpenMeteoGridModel:
    """One Open-Meteo model this script can fetch, and its output directory name."""

    output_dir: str
    models_parameter: str
    label: str
    docs_url: str
    """The model's own technical documentation, for the README's "further reading"."""
    known_gotcha: str | None = None
    """A finding from the `data-validation` skill's checklist worth every reader knowing, or
    `None` where the check found nothing. Keep it to a sentence or two; the full numbers behind
    the claim live in this product's `lineage.json`."""
    hourly_variables: tuple[str, ...] = RADIATION_VARIABLES
    """The `hourly=` variables this model is fetched with. Defaults to the radiation-only set;
    override per model where a study needs more, as `ecmwf-ifs-hres` does for wind."""


MODELS: Final[dict[str, OpenMeteoGridModel]] = {
    "ecmwf-ifs-hres": OpenMeteoGridModel(
        output_dir="ECMWF-IFS-HRES",
        models_parameter="ecmwf_ifs",
        label="ECMWF IFS HRES 9 km",
        docs_url="https://www.ecmwf.int/en/forecasts/documentation-and-support",
        hourly_variables=RADIATION_VARIABLES + WIND_VARIABLES,
        known_gotcha=(
            "`models_parameter` must be `ecmwf_ifs`, not `ecmwf_ifs04` — `ecmwf_ifs04` nominally "
            "means `ECMWF IFS 0.4°` (~44 km, global), a different and much coarser product from "
            "the label's `ECMWF IFS HRES 9 km`. On the **commercial customer** host, `ecmwf_ifs04` "
            "returns `shortwave_radiation` and `wind_speed_100m` as entirely null (every value, "
            "every point, every hour), while `ecmwf_ifs` returns real data matching the API's own "
            "unrequested 'best match' default exactly. `fetch_open_meteo_previous_runs.py` also "
            "uses `ecmwf_ifs` for this same product."
        ),
    ),
    "dmi-harmonie-arome": OpenMeteoGridModel(
        output_dir="DMI-HARMONIE-AROME",
        models_parameter="dmi_harmonie_arome_europe",
        label="DMI HARMONIE-AROME (Europe domain)",
        docs_url="https://opendatadocs.dmi.googleapis.com/",
        known_gotcha=(
            "`direct_radiation` is an upstream defect and should not be used without checking "
            "against a second source: it is exactly 0 in 48% of daytime hours (2.9% for the "
            "KNMI HARMONIE-AROME domain over the same points/hours), including 84 whole UTC "
            "dates where it is zero all day while `shortwave_radiation` is not, and 74 rows "
            "where it exceeds `shortwave_radiation` (physically impossible). "
            "`shortwave_radiation` is unaffected and correlates well with KNMI and SARAH-3."
        ),
    ),
    "knmi-harmonie-arome": OpenMeteoGridModel(
        output_dir="KNMI-HARMONIE-AROME",
        models_parameter="knmi_harmonie_arome_europe",
        label="KNMI HARMONIE-AROME (Europe domain)",
        docs_url="https://english.knmidata.nl/",
    ),
    "arpege-europe": OpenMeteoGridModel(
        output_dir="ARPEGE-EUROPE",
        models_parameter="meteofrance_arpege_europe",
        label="Meteo-France ARPEGE Europe",
        docs_url="https://meteofrance.com/en/weather-forecast-and-services/weather-data",
        known_gotcha=(
            "Both variables step up at 2024-01-01, most likely a change of source inside "
            "Open-Meteo's archive rather than a real weather change: shortwave is roughly 20% "
            "low and direct roughly 40-45% low before that date relative to after it, measured "
            "against ECMWF-IFS-HRES over the same points/hours. There is also a 35-hour gap "
            "(both variables null at all points) from 2023-12-31 07:00 to 2024-01-01 17:00 "
            "UTC. Exclude data before 2024-01-01 from any study, or treat it as a separate, "
            "lower-quality product."
        ),
    ),
}
"""`dmi_harmonie_arome_dini` (DMI's own name for the model) is not a recognised `models=` value on
Open-Meteo's API; `dmi_harmonie_arome_europe` is the DMI-family identifier that answers at Great
Britain coordinates, so that is what is requested here. Confirmed by probing the live API, not by
Open-Meteo's docs, which do not list machine-readable identifiers."""


def _get_json(*, url: str) -> Any:
    """Fetch one URL, retrying a transport failure but never an API refusal.

    Copied from `studies/beam_diffuse_split/fetch_open_meteo_point.py`'s `_get_json`: see that
    function's docstring for why a rate-limit refusal must not be retried.
    """
    for attempt in range(MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as refusal:
            reason = json.loads(refusal.read() or b"{}").get("reason", "no reason given")
            msg = f"Open-Meteo refused the request with HTTP {refusal.code}: {reason}"
            raise RuntimeError(msg) from refusal
        except Exception:  # noqa: BLE001
            time.sleep(5.0 * (attempt + 1))
        else:
            if isinstance(payload, dict) and payload.get("error"):
                msg = f"Open-Meteo refused the request: {payload.get('reason')}"
                raise RuntimeError(msg)
            return payload
    msg = f"Open-Meteo failed after {MAX_ATTEMPTS} attempts"
    raise RuntimeError(msg)


POINTS_PER_REQUEST: Final[int] = 100
"""Grid points per GET request. The customer API host's nginx front end refuses a request whose
URI exceeds its buffer (confirmed: HTTP 414 Request-URI Too Large at the full 342-point URL,
~13 KB, against `customer-historical-forecast-api.open-meteo.com` — the free host tolerates the
same URL). 100 points keeps the URL to a few KB, comfortably under that limit, regardless of how
many points the trial-area box's grid produces."""


def _fetch_grid_batch(
    *,
    points: pl.DataFrame,
    variables: tuple[str, ...],
    models_parameter: str,
    start_date: str,
    end_date: str,
) -> pl.DataFrame:
    """Fetch one date range at up to `POINTS_PER_REQUEST` grid points, in a single request.

    Args:
        points: Carries `point_id`, `latitude`, `longitude` (see `TrialAreaBox.grid_points`).
        variables: Open-Meteo's names for the hourly variables to request.
        models_parameter: The value of the API's `models=` query parameter.
        start_date: First date to request, as `YYYY-MM-DD`.
        end_date: Last date to request, as `YYYY-MM-DD`.

    Returns:
        One row per (point_id, time), with no coordinate column.
    """
    url = (
        f"{HISTORICAL_FORECAST_URL}"
        f"?latitude={','.join(str(value) for value in points['latitude'])}"
        f"&longitude={','.join(str(value) for value in points['longitude'])}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly={','.join(variables)}"
        f"&models={models_parameter}&timezone=UTC"
    )
    api_key = open_meteo_api_key()
    if api_key:
        url += f"&apikey={api_key}"
    payload = _get_json(url=url)
    blocks = payload if isinstance(payload, list) else [payload]
    if len(blocks) != points.height:
        msg = f"asked for {points.height} points and got {len(blocks)} blocks"
        raise RuntimeError(msg)
    return pl.concat(
        pl.DataFrame(
            {"point_id": point_id, "time": block["hourly"]["time"]}
            | {name: block["hourly"][name] for name in variables},
            schema_overrides=dict.fromkeys(variables, pl.Float64),
        )
        for point_id, block in zip(points["point_id"], blocks, strict=True)
    ).with_columns(pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M").dt.replace_time_zone("UTC"))


def fetch_grid_frame(
    *,
    points: pl.DataFrame,
    variables: tuple[str, ...],
    models_parameter: str,
    start_date: str,
    end_date: str,
) -> pl.DataFrame:
    """Fetch one date range at every grid point, batched to `POINTS_PER_REQUEST` points a request.

    Args:
        points: Carries `point_id`, `latitude`, `longitude` (see `TrialAreaBox.grid_points`).
        variables: Open-Meteo's names for the hourly variables to request.
        models_parameter: The value of the API's `models=` query parameter.
        start_date: First date to request, as `YYYY-MM-DD`.
        end_date: Last date to request, as `YYYY-MM-DD`.

    Returns:
        One row per (point_id, time), with no coordinate column.
    """
    return pl.concat(
        _fetch_grid_batch(
            points=points[batch_start : batch_start + POINTS_PER_REQUEST],
            variables=variables,
            models_parameter=models_parameter,
            start_date=start_date,
            end_date=end_date,
        )
        for batch_start in range(0, points.height, POINTS_PER_REQUEST)
    )


def _write_docs_for_model(
    *,
    frame: pl.DataFrame,
    model: OpenMeteoGridModel,
    n_points: int,
    output_dir: Path,
    start_date: str,
    end_date: str,
) -> None:
    """Write `lineage.json` and `README.md` for one already-fetched (or already-cached) frame.

    Split out from `main` so the docs can be regenerated from an already-downloaded combined
    parquet — after a change to the columns/gotcha text, or to `GRID_SPACING_DEG` — without
    re-fetching from Open-Meteo. Every fact below about the fetched data (the null counts) is
    computed from `frame` itself, never hand-typed, so the docs cannot drift out of sync with
    what the file actually contains.

    Args:
        frame: The combined, already-written frame these docs describe.
        model: Which `MODELS` entry this frame belongs to.
        n_points: The point count this *frame* was actually fetched over — `frame["point_id"]
            .n_unique()` when regenerating docs for an already-downloaded file, not a fresh
            `box.grid_points(...).height` call, which reflects the trial-area box's *current*
            extent and can differ from what the file was fetched with if the box changed since.
        output_dir: The product's own directory under `data/studies/weather/`.
        start_date: First date actually fetched, `YYYY-MM-DD`.
        end_date: Last date actually fetched, `YYYY-MM-DD`.
    """
    null_counts = frame.select(model.hourly_variables).null_count().row(0, named=True)
    null_summary = "; ".join(
        f"`{column}`: {count} null{'s' if count != 1 else ''}"
        for column, count in null_counts.items()
    )
    any_nulls = any(count > 0 for count in null_counts.values())
    has_wind = any(name in model.hourly_variables for name in WIND_VARIABLES)

    write_lineage_note(
        product_dir=output_dir,
        source_address=HISTORICAL_FORECAST_URL,
        request_description=(
            f"{model.label}, models={model.models_parameter}, "
            f"hourly={','.join(model.hourly_variables)}, "
            f"{n_points} grid points at {GRID_SPACING_DEG} degree spacing inside the "
            f"trial-area box (a few grid cells' margin around the NGED generator roster's own "
            f"extent)"
        ),
        variables=list(model.hourly_variables),
        extra={
            "n_points": n_points,
            "grid_spacing_deg": GRID_SPACING_DEG,
            "date_range_fetched": [start_date, end_date],
            "row_count": frame.height,
            "note": (
                "Each radiation value is a mean over the hour ENDING at its `time` label "
                "(confirmed against clear-sky irradiance: values track best when shifted "
                "20-30 minutes earlier)"
                + (
                    "; each wind value is instantaneous at `time` (Open-Meteo's own "
                    "convention for `wind_speed_10m`/`wind_speed_100m`, unlike its "
                    "period-ending radiation)"
                    if has_wind
                    else ""
                )
                + f", and `time` is timezone-aware UTC. Null counts in this fetch — "
                f"{null_summary}." + (f" {model.known_gotcha}" if model.known_gotcha else "")
            ),
        },
    )
    write_readme(
        product_dir=output_dir,
        product_name=model.label,
        source_web_page="https://open-meteo.com/en/docs/historical-forecast-api",
        script_path="studies/weather_downloads/fetch_open_meteo_grid.py",
        lineage_filenames=["lineage.json"],
        columns={
            "point_id": POINT_ID_COLUMN_DESCRIPTION,
            "time": (
                "UTC, timezone-aware. Marks the END of the hour each radiation value averages "
                "over (confirmed against clear-sky irradiance); wind values at this same `time` "
                "are instantaneous, not averaged."
                if has_wind
                else "UTC, timezone-aware. Marks the END of the hour each value averages over "
                "(confirmed against clear-sky irradiance)."
            ),
        }
        | {name: VARIABLE_COLUMN_DESCRIPTIONS[name] for name in model.hourly_variables},
        missing_value_convention=(
            f"Polars null. {null_summary} in this fetch"
            + (
                " — zero nulls found, see this product's own `lineage.json` for the row count."
                if not any_nulls
                else " — see the gotchas below and this product's own `lineage.json` for the "
                "`note` field's full account."
            )
        ),
        gotchas=[model.known_gotcha] if model.known_gotcha else [],
        external_docs={
            f"{model.label} technical documentation": model.docs_url,
            "Open-Meteo Historical Forecast API": (
                "https://open-meteo.com/en/docs/historical-forecast-api"
            ),
        },
    )


def main() -> int:
    """Download one model over the trial-area grid for one date range, and write it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    arguments = parser.parse_args()
    model = MODELS[arguments.model]

    box = load_trial_area_box()
    points = box.grid_points(spacing_deg=GRID_SPACING_DEG)

    frame = fetch_grid_frame(
        points=points,
        variables=model.hourly_variables,
        models_parameter=model.models_parameter,
        start_date=arguments.start_date,
        end_date=arguments.end_date,
    )

    output_dir = WEATHER_DOWNLOADS_DIR / model.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{model.output_dir}_{arguments.start_date}_{arguments.end_date}.parquet"
    output_path = output_dir / filename
    frame.write_parquet(output_path)
    size_mb = output_path.stat().st_size / 1e6

    print(
        f"{model.label}: wrote {frame.height} rows for {points.height} points "
        f"({arguments.start_date} to {arguments.end_date}) to {output_path}, {size_mb:.3f} MB"
    )

    _write_docs_for_model(
        frame=frame,
        model=model,
        n_points=points.height,
        output_dir=output_dir,
        start_date=arguments.start_date,
        end_date=arguments.end_date,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
