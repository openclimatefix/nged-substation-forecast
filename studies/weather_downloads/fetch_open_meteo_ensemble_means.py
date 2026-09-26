"""Download Open-Meteo's ensemble MEAN (and spread) series at the nine study sites.

One-off throwaway script for comparing high-resolution ensemble means with the ECMWF ENS mean, all
served by the one Open-Meteo API so the processing is identical. The four products, each a
`models=` value on the Ensemble API (<https://open-meteo.com/en/docs/ensemble-api>), are in
`PRODUCTS`. Each `_ensemble_mean` model is a pseudo-model that holds the mean over the members of
one ensemble, with the standard deviation across members under the `_spread` suffix of each
variable. The plain ICON models (`icon_d2_eps`, `icon_eu_eps`) serve members only, and their
`_spread` columns are null, so the mean models are the ones fetched.

**What each series is.** The Ensemble API with `start_date`/`end_date` returns one stitched series:
for every hour, the value from the newest run that covers it, so the leads are short and mixed and
there is no `init_time`. The output is keyed by `site` and `time` only. The first date served is
recorded in `FIRST_DATE`, measured by `--probe-first-date`, which bisects on one-day requests. The
archive looks like a rolling or growing window rather than a full history, so a later re-run may
find an earlier start: `lineage.json` records the first date served on the day of the run.

**Requests.** Each request covers one 14-day window for one `cell_selection` group of sites (the six
PV sites with `nearest`, the three wind sites with `land`, as `fetch_open_meteo_single_runs.py`
does) and all 16 series (8 variables and their spreads). Open-Meteo weights a request by roughly
sites x ceil(series / 10) x ceil(days / 14), so one window costs `sites x 2` calls. The sites come
from the private roster at run time and stay in memory. Rows carry only the anonymised `site` label.
No coordinate, and never the API key, reaches a log line, a file, or the lineage note;
`OPEN_METEO_TOKEN` must be exported into the environment (see `paths.open_meteo_api_key`).

**Resuming.** Each complete window is a parquet under `<product>/_chunks/`, written to `.tmp` and
renamed, never overwritten. A rerun skips the windows on disk. The combined
`<product>/<product>.parquet` is written once, and the script refuses to replace it. The last
window ends on the `--end-date` (default: yesterday, UTC), so its file name carries its end date.

Wind speeds are in km/h, as in the other Open-Meteo files here, and radiation in W/m2.

Run it with `uv run python studies/weather_downloads/fetch_open_meteo_ensemble_means.py`. Pass
`--max-windows 2` to fetch only two windows per product, or `--product ukmo_uk` for one product.
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from math import ceil
from pathlib import Path
from typing import Any, Final

import polars as pl
from fetch_open_meteo_previous_runs import _check_timestamp_convention
from fetch_open_meteo_single_runs import (
    RADIATION_UNITS,
    WIND_SPEED_UNIT,
    _get_json,
    _require_api_key,
    _sites,
    _write_atomically,
)
from lineage import write_lineage_note, write_readme
from paths import WEATHER_DOWNLOADS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("fetch_open_meteo_ensemble_means")

ENSEMBLE_URL: Final[str] = "https://customer-ensemble-api.open-meteo.com/v1/ensemble"
"""The commercial Ensemble API host. It needs `OPEN_METEO_TOKEN`."""

OUTPUT_ROOT: Final[Path] = WEATHER_DOWNLOADS_DIR / "OPEN-METEO-ENSEMBLE-MEANS"

BASE_VARIABLES: Final[tuple[str, ...]] = (
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "temperature_2m",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_100m",
    "cloud_cover",
)
"""Open-Meteo's normalised names. Each is also requested with the `_spread` suffix."""

SERIES: Final[tuple[str, ...]] = tuple(
    name for base in BASE_VARIABLES for name in (base, f"{base}_spread")
)

WINDOW_DAYS: Final[int] = 14
"""One request window. Open-Meteo counts every 14 days of a request as one more call."""

FIRST_DATE: Final[date] = date(2026, 6, 25)
"""The first date every product serves, measured on 2026-09-26 by `--probe-first-date`."""

REQUEST_SLEEP_SECONDS: Final[float] = 1.0
CACHE_DIR_NAME: Final[str] = "_chunks"


@dataclass(frozen=True)
class EnsembleMeanProduct:
    """One ensemble-mean model the Ensemble API serves, and where its output is written."""

    output_dir: str
    models_parameter: str
    label: str
    docs_url: str
    always_null: tuple[str, ...]
    """Series the API serves as all null for this product, measured in the probe."""


PRODUCTS: Final[dict[str, EnsembleMeanProduct]] = {
    "ukmo_uk": EnsembleMeanProduct(
        output_dir="UKMO-UK-ENSEMBLE-MEAN-2KM",
        models_parameter="ukmo_uk_ensemble_mean_2km",
        label="MOGREPS-UK ensemble mean, 2 km",
        docs_url="https://open-meteo.com/en/docs/ukmo-api",
        always_null=(
            "diffuse_radiation_spread",
            "wind_speed_100m",
            "wind_speed_100m_spread",
            "wind_direction_100m",
            "wind_direction_100m_spread",
        ),
    ),
    "icon_d2": EnsembleMeanProduct(
        output_dir="ICON-D2-EPS-ENSEMBLE-MEAN",
        models_parameter="dwd_icon_d2_eps_ensemble_mean",
        label="ICON-D2-EPS ensemble mean, 2 km",
        docs_url="https://open-meteo.com/en/docs/dwd-api",
        always_null=("wind_speed_100m_spread", "wind_direction_100m_spread"),
    ),
    "icon_eu": EnsembleMeanProduct(
        output_dir="ICON-EU-EPS-ENSEMBLE-MEAN",
        models_parameter="dwd_icon_eu_eps_ensemble_mean",
        label="ICON-EU-EPS ensemble mean, 7 km",
        docs_url="https://open-meteo.com/en/docs/dwd-api",
        always_null=(
            "wind_speed_100m",
            "wind_speed_100m_spread",
            "wind_direction_100m",
            "wind_direction_100m_spread",
        ),
    ),
    "ecmwf_ifs": EnsembleMeanProduct(
        output_dir="ECMWF-IFS-ENS-MEAN-025",
        models_parameter="ecmwf_ifs025_ensemble_mean",
        label="ECMWF IFS ENS ensemble mean, 0.25 degrees",
        docs_url="https://open-meteo.com/en/docs/ecmwf-api",
        always_null=("direct_radiation_spread", "diffuse_radiation_spread"),
    ),
}


def _windows(*, first: date, last: date) -> list[tuple[date, date]]:
    """Return consecutive `WINDOW_DAYS` windows covering `first`..`last`, the final one clipped."""
    windows = []
    start = first
    while start <= last:
        windows.append((start, min(start + timedelta(days=WINDOW_DAYS - 1), last)))
        start += timedelta(days=WINDOW_DAYS)
    return windows


def _check_block(*, block: dict[str, Any], position: int, n_blocks: int) -> None:
    """Fail loudly unless one response block is in its position and its units are the known ones.

    A site label comes from the block's position, so a reordered response would swap two
    anonymised labels without any error. Every block after the first carries a `location_id` equal
    to its position. A unit of `undefined` is accepted only for a series that is entirely null.

    Args:
        block: One location's block of the response.
        position: The block's index in the response list.
        n_blocks: How many blocks the response holds.

    Raises:
        RuntimeError: If `location_id` differs from the position, or a unit differs from
            `WIND_SPEED_UNIT` for wind speed or `RADIATION_UNITS` for radiation.
    """
    if n_blocks > 1 and block.get("location_id", 0 if position == 0 else None) != position:
        msg = f"response block {position} is out of order or has no location_id"
        raise RuntimeError(msg)
    units = block["hourly_units"]
    for name in SERIES:
        unit = units[name]
        if unit == "undefined":
            if any(value is not None for value in block["hourly"][name]):
                msg = f"{name} has values but its unit is undefined"
                raise RuntimeError(msg)
        elif name.startswith("wind_speed") and unit != WIND_SPEED_UNIT:
            msg = f"{name} is reported in {unit!r}, expected {WIND_SPEED_UNIT!r}"
            raise RuntimeError(msg)
        elif "radiation" in name and unit not in RADIATION_UNITS:
            msg = f"{name} is reported in {unit!r}, expected W/m2"
            raise RuntimeError(msg)


def fetch_window_group(
    *, product: EnsembleMeanProduct, sites: pl.DataFrame, start: date, end: date
) -> pl.DataFrame:
    """Fetch one window for one `cell_selection` group of sites, in one request.

    Args:
        product: The ensemble-mean model to request.
        sites: One `cell_selection` group of the roster, carrying `site`, `latitude`, `longitude`,
            and `cell_selection`.
        start: The first date of the window.
        end: The last date of the window, inclusive.

    Returns:
        One row per (site, hour), with `site`, `time` (naive UTC), and one `Float64` column per
        entry of `SERIES`. Coordinates never enter the frame.

    Raises:
        RuntimeError: If the response does not carry one block per site, a block is out of
            position or in unexpected units, or the API refuses the request.
    """
    request = (
        f"{ENSEMBLE_URL}"
        f"?latitude={','.join(str(value) for value in sites['latitude'])}"
        f"&longitude={','.join(str(value) for value in sites['longitude'])}"
        f"&start_date={start.isoformat()}&end_date={end.isoformat()}"
        f"&hourly={','.join(SERIES)}&models={product.models_parameter}&timezone=UTC"
        f"&cell_selection={sites['cell_selection'][0]}&apikey={_require_api_key()}"
    )
    payload = _get_json(url=request)
    blocks = payload if isinstance(payload, list) else [payload]
    if len(blocks) != sites.height:
        msg = f"asked for {sites.height} sites and got {len(blocks)} blocks"
        raise RuntimeError(msg)
    for position, block in enumerate(blocks):
        _check_block(block=block, position=position, n_blocks=len(blocks))
    frame = pl.concat(
        pl.DataFrame(
            {"site": site, "time": block["hourly"]["time"]}
            | {name: block["hourly"][name] for name in SERIES},
            schema_overrides=dict.fromkeys(SERIES, pl.Float64),
        )
        for site, block in zip(sites["site"], blocks, strict=True)
    )
    return frame.with_columns(pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M", time_unit="us"))


def _fetch_window(
    *, product: EnsembleMeanProduct, sites: pl.DataFrame, start: date, end: date
) -> pl.DataFrame:
    """Fetch one window for every site, one request per `cell_selection` group.

    Args:
        product: The ensemble-mean model to request.
        sites: The roster with `cell_selection`.
        start: The first date of the window.
        end: The last date of the window, inclusive.

    Returns:
        The window for every site, sorted by `site` and `time`.

    Raises:
        RuntimeError: If any site does not have exactly 24 hourly rows per day of the window.
    """
    groups = []
    for group in sites.partition_by("cell_selection", maintain_order=True):
        groups.append(fetch_window_group(product=product, sites=group, start=start, end=end))
        time.sleep(REQUEST_SLEEP_SECONDS)
    frame = pl.concat(groups).sort("site", "time")
    expected_hours = ((end - start).days + 1) * 24
    short = frame.group_by("site").agg(pl.len().alias("n")).filter(pl.col("n") != expected_hours)
    if not short.is_empty():
        msg = f"window {start}..{end}: sites {sorted(short['site'])} lack {expected_hours} hours"
        raise RuntimeError(msg)
    return frame


def _combined_path(*, product: EnsembleMeanProduct) -> Path:
    return OUTPUT_ROOT / product.output_dir / f"{product.output_dir}.parquet"


def _fetch_product(
    *,
    product: EnsembleMeanProduct,
    sites: pl.DataFrame,
    last: date,
    max_windows: int | None,
) -> bool:
    """Fetch every missing window of one product, then write its combined file once.

    Args:
        product: The ensemble-mean model to fetch.
        sites: The roster with `cell_selection`.
        last: The last date wanted, inclusive.
        max_windows: How many missing windows to fetch, or `None` for all.

    Returns:
        Whether the combined file was written.

    Raises:
        FileExistsError: If the combined file already exists, because it is never overwritten.
    """
    product_dir = OUTPUT_ROOT / product.output_dir
    combined = _combined_path(product=product)
    if combined.exists():
        msg = f"{combined.name} already exists and is never overwritten"
        raise FileExistsError(msg)
    cache = product_dir / CACHE_DIR_NAME
    cache.mkdir(parents=True, exist_ok=True)
    fetched = 0
    paths = []
    for start, end in _windows(first=FIRST_DATE, last=last):
        path = cache / f"{start.isoformat()}_{end.isoformat()}.parquet"
        paths.append(path)
        if path.exists():
            continue
        if max_windows is not None and fetched >= max_windows:
            continue
        frame = _fetch_window(product=product, sites=sites, start=start, end=end)
        _write_atomically(frame=frame, path=path)
        fetched += 1
        _LOG.info("%s: wrote window %s..%s", product.output_dir, start, end)
    if not all(path.exists() for path in paths):
        _LOG.info("%s: windows missing, combined file not written", product.output_dir)
        return False
    frame = pl.concat(pl.read_parquet(path) for path in paths).sort("site", "time")
    _write_atomically(frame=frame, path=combined)
    _write_docs(product=product, frame=frame, sites=sites, first=FIRST_DATE, last=last)
    return True


def _column_description(*, name: str) -> str:
    """Describe one series column, with its unit, from its name."""
    base, is_spread = (name.removesuffix("_spread"), name.endswith("_spread"))
    unit = {
        "shortwave_radiation": "W/m2, mean over the hour ending at `time`",
        "direct_radiation": "W/m2, mean over the hour ending at `time`",
        "diffuse_radiation": "W/m2, mean over the hour ending at `time`",
        "temperature_2m": "degrees C (spread in K)",
        "wind_speed_10m": "km/h",
        "wind_speed_100m": "km/h",
        "wind_direction_100m": "degrees",
        "cloud_cover": "percent",
    }[base]
    kind = "Standard deviation across ensemble members of" if is_spread else "Ensemble mean of"
    return f"{kind} `{base}`, {unit}."


def _write_docs(
    *,
    product: EnsembleMeanProduct,
    frame: pl.DataFrame,
    sites: pl.DataFrame,
    first: date,
    last: date,
) -> None:
    """Write `lineage.json` and `README.md` for one product, quoting only measured numbers."""
    timestamp = _check_timestamp_convention(frame=frame, sites=sites)
    null_fraction = {
        name: frame[name].null_count() / frame.height for name in SERIES if frame.height
    }
    output_dir = OUTPUT_ROOT / product.output_dir
    write_lineage_note(
        product_dir=output_dir,
        source_address=ENSEMBLE_URL,
        request_description=(
            f"{product.label}, models={product.models_parameter}, hourly={','.join(SERIES)}, "
            f"{sites.height} meter sites, {first} to {last}, 14-day windows"
        ),
        variables=list(SERIES),
        extra={
            "n_sites": sites.height,
            "row_count": frame.height,
            "first_date_served_at_run": first.isoformat(),
            "date_range_fetched": [first.isoformat(), last.isoformat()],
            "retrieved_on": datetime.now(UTC).date().isoformat(),
            "null_fraction_per_column": null_fraction,
            "timestamp_convention_check": timestamp,
        },
    )
    write_readme(
        product_dir=output_dir,
        product_name=product.label,
        source_web_page="https://open-meteo.com/en/docs/ensemble-api",
        script_path="studies/weather_downloads/fetch_open_meteo_ensemble_means.py",
        lineage_filenames=["lineage.json"],
        columns={
            "site": "Anonymised meter label (`A`-`F` for solar, `W1`-`W3` for wind).",
            "time": "UTC, hourly, naive. The hour label; radiation is a mean over the hour ending.",
            **{name: _column_description(name=name) for name in SERIES},
        },
        missing_value_convention=(
            "Polars null. The series the API serves as all null for this product are "
            f"{', '.join(f'`{name}`' for name in product.always_null) or 'none'}; every other "
            "column should have no nulls, which `validate_open_meteo_ensemble_means.py` checks."
        ),
        gotchas=[
            (
                "**This is one stitched series, not a forecast at a stated lead.** For each hour "
                "the API returns the value from the newest run that covers it, so leads are short "
                "and mixed, there is no `init_time`, and it cannot be used for lead-resolved "
                "skill."
            ),
            (
                "**The archive starts on 2026-06-25 for all four ensemble-mean products, on the "
                "day this was measured (2026-09-26).** It may be a window that moves or a history "
                "that only grows, so a later re-run can find a different start."
            ),
            "**Wind speeds are in km/h**, as in the other Open-Meteo files, not m/s.",
            (
                "**`_spread` is a standard deviation across members**, in the unit of the "
                "variable, not a range."
            ),
        ],
        external_docs={
            f"{product.label} model documentation": product.docs_url,
            "Open-Meteo Ensemble API": "https://open-meteo.com/en/docs/ensemble-api",
        },
    )


def _probe_first_date() -> None:
    """Print, per product, the first date on which the API serves at least 20 non-null hours.

    Bisects between 2020-01-01 and 8 days ago on one-day, one-variable, one-location requests,
    which is about 13 calls per product. The probe location is a public one (52.0, -1.0), never a
    study site. Assumes the archive has no gaps after its first day.
    """
    import urllib.error
    import urllib.request

    key = _require_api_key()

    def _hours(*, models: str, day: date) -> int:
        url = (
            f"{ENSEMBLE_URL}?latitude=52.0&longitude=-1.0&hourly=temperature_2m&models={models}"
            f"&start_date={day}&end_date={day}&apikey={key}"
        )
        try:
            body = json.load(urllib.request.urlopen(url, timeout=60.0))
        except urllib.error.HTTPError:
            return 0
        return sum(value is not None for value in body["hourly"]["temperature_2m"])

    for product in PRODUCTS.values():
        low, high = date(2020, 1, 1), datetime.now(UTC).date() - timedelta(days=8)
        while (high - low).days > 1:
            middle = low + (high - low) // 2
            if _hours(models=product.models_parameter, day=middle) >= 20:
                high = middle
            else:
                low = middle
        _LOG.info("%s first date served: %s", product.models_parameter, high)


def main() -> int:
    """Fetch every product's ensemble mean and spread at every study site."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", choices=tuple(PRODUCTS), default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    parser.add_argument("--probe-first-date", action="store_true")
    arguments = parser.parse_args()
    if arguments.probe_first_date:
        _probe_first_date()
        return 0
    last = arguments.end_date or datetime.now(UTC).date() - timedelta(days=1)
    products = [PRODUCTS[arguments.product]] if arguments.product else list(PRODUCTS.values())
    sites = _sites()
    n_windows = len(_windows(first=FIRST_DATE, last=last))
    calls = len(products) * n_windows * sites.height * ceil(len(SERIES) / 10)
    _LOG.info(
        "%d products x %d windows, about %d weighted calls in total",
        len(products),
        n_windows,
        calls,
    )
    for product in products:
        _fetch_product(product=product, sites=sites, last=last, max_windows=arguments.max_windows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
