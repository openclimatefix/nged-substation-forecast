"""Download one Open-Meteo forecast model over a grid of points inside the trial-area box.

One-off throwaway script for the downloads in
<https://github.com/openclimatefix/nged-substation-forecast/issues/841>. Covers the three
Open-Meteo-served products the issue asks for first: ECMWF IFS HRES 9 km, DMI and KNMI
HARMONIE-AROME, and Meteo-France ARPEGE Europe. A further Open-Meteo model needs only a new entry
in `MODELS` below.

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
from paths import WEATHER_DOWNLOADS_DIR, load_trial_area_box

HISTORICAL_FORECAST_URL: Final[str] = "https://historical-forecast-api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT_SECONDS: Final[float] = 600.0
MAX_ATTEMPTS: Final[int] = 5
HOURLY_VARIABLES: Final[tuple[str, ...]] = ("shortwave_radiation", "direct_radiation")
GRID_SPACING_DEG: Final[float] = 0.05
"""Spacing used for every product here, chosen to land in the "a few dozen points" the issue
describes across the box, rather than resolving each product at its own native grid spacing. Read
from this one constant everywhere — the request, the lineage note, and the README all derive from
it, so it can never drift out of sync with what was actually fetched."""


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


MODELS: Final[dict[str, OpenMeteoGridModel]] = {
    "ecmwf-ifs-hres": OpenMeteoGridModel(
        output_dir="ECMWF-IFS-HRES",
        models_parameter="ecmwf_ifs04",
        label="ECMWF IFS HRES 9 km",
        docs_url="https://www.ecmwf.int/en/forecasts/documentation-and-support",
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


def fetch_grid_frame(
    *,
    points: pl.DataFrame,
    variables: tuple[str, ...],
    models_parameter: str,
    start_date: str,
    end_date: str,
) -> pl.DataFrame:
    """Fetch one date range at every grid point, in a single request.

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
    null_counts = frame.select(HOURLY_VARIABLES).null_count().row(0, named=True)
    null_summary = "; ".join(
        f"`{column}`: {count} null{'s' if count != 1 else ''}"
        for column, count in null_counts.items()
    )
    any_nulls = any(count > 0 for count in null_counts.values())

    write_lineage_note(
        product_dir=output_dir,
        source_address=HISTORICAL_FORECAST_URL,
        request_description=(
            f"{model.label}, models={model.models_parameter}, hourly={','.join(HOURLY_VARIABLES)}, "
            f"{n_points} grid points at {GRID_SPACING_DEG} degree spacing inside the "
            f"trial-area box (a few grid cells' margin around the NGED generator roster's own "
            f"extent)"
        ),
        variables=list(HOURLY_VARIABLES),
        extra={
            "n_points": n_points,
            "grid_spacing_deg": GRID_SPACING_DEG,
            "date_range_fetched": [start_date, end_date],
            "row_count": frame.height,
            "note": (
                "Each hourly value is a mean over the hour ENDING at its `time` label (confirmed "
                "against clear-sky irradiance: values track best when shifted 20-30 minutes "
                f"earlier), and `time` is timezone-aware UTC. Null counts in this fetch — "
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
            "point_id": "Running index (0-based) into the trial-area box's regular lat/lon grid "
            "at `GRID_SPACING_DEG` spacing, in request order — not a coordinate, and not "
            "guaranteed to be a distinct model grid cell (adjacent points can share a cell where "
            "the model's own resolution is coarser than the grid spacing).",
            "time": "UTC, timezone-aware. Marks the END of the hour each value averages over "
            "(confirmed against clear-sky irradiance).",
            "shortwave_radiation": "Global horizontal irradiance, W/m^2.",
            "direct_radiation": "Direct (beam) horizontal irradiance, W/m^2.",
        },
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
        variables=HOURLY_VARIABLES,
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
