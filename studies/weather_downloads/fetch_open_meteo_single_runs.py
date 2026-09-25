"""Download Open-Meteo's Single Runs API for ECMWF IFS HRES at the nine study sites.

One-off throwaway script for the matched-lead forecast study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/810>. The study needs IFS at
as-issued lead days 0, 1, 2, 3, 5, 7 and 10. The Previous Runs API
(`fetch_open_meteo_previous_runs.py`) serves lead days 0 to 7 only for `ecmwf_ifs025`, and its
`ecmwf_ifs` (HRES) columns are mostly null. The Single Runs API
(<https://open-meteo.com/en/docs/single-runs-api>) returns every lead of one named model run in
one request, so fetching the 00 UTC run of every day gives every lead day. Day 14 is out of scope:
the IFS HRES horizon is 10 days.

**Requests.** Each run costs two requests, one per `cell_selection` group: the six PV sites with
`nearest` and the three wind sites with `land`, the settings the sibling per-site fetchers use. The
sites come from the private roster at run time and stay in memory. Rows carry only the anonymised
`site` label. No coordinate, and never the API key, reaches a log line, a file, or the lineage note;
`OPEN_METEO_API_KEY` must be exported into the environment (or set in `.env`).

**A run is stored only if it is complete.** Every site must have `LEADS_PER_RUN` hourly leads and
every variable non-null, except the two radiation variables at lead 0, which are null by the
averaging convention. A run the API refuses with HTTP 400 "requested model run is not available" is
recorded in `_unavailable_runs.json`. A historical run that comes back incomplete is recorded with
its reason in `_incomplete_runs.json` and the backfill continues. Both ledgers hold dates only. A
trailing run that is not yet published or complete is skipped and retried on the next invocation.

**Refusals and errors.** Every refusal counts against `--max-runs`. After
`MAX_CONSECUTIVE_REFUSALS` in a row the script stops, because a run of refusals means the request is
wrong or the archive has ended, not that many days are missing. HTTP 429 stops the script cleanly
and it resumes from the checkpoints. HTTP 5xx and transport failures are retried with backoff. The
URL, the key and any coordinate are never printed, and any refusal reason is truncated.

**Resuming.** One parquet per month. A month with every run day fetched or documented unavailable is
`YYYY-MM.parquet`; otherwise it is `YYYY-MM.partial.parquet`, rewritten atomically after every run,
so a killed process loses at most one run.

Run it with `uv run python studies/weather_downloads/fetch_open_meteo_single_runs.py`. Pass
`--max-runs 4` to fetch only the first four missing runs.
"""

import argparse
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import polars as pl
from fetch_open_meteo_previous_runs import (
    BASE_VARIABLES,
    _check_timestamp_convention,
    _pv_sites,
    _wind_sites,
)
from lineage import write_lineage_note, write_readme
from paths import WEATHER_DOWNLOADS_DIR, open_meteo_api_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("fetch_open_meteo_single_runs")

SINGLE_RUNS_URL: Final[str] = "https://customer-single-runs-api.open-meteo.com/v1/forecast"
"""The commercial Single Runs host. It needs `OPEN_METEO_API_KEY`."""

MODELS_PARAMETER: Final[str] = "ecmwf_ifs"
"""The `models=` value for IFS HRES. `ecmwf_ifs025` answers "run not available" on this API."""

PRODUCT_DIR: Final[Path] = WEATHER_DOWNLOADS_DIR / "ECMWF-IFS-SINGLE-RUNS"
COMBINED_FILENAME: Final[str] = "ECMWF-IFS-SINGLE-RUNS.parquet"
UNAVAILABLE_FILENAME: Final[str] = "_unavailable_runs.json"
INCOMPLETE_FILENAME: Final[str] = "_incomplete_runs.json"

FIRST_RUN_DATE: Final[date] = date(2024, 3, 14)
"""The first day the Single Runs archive serves for IFS."""

RUN_HOUR_UTC: Final[int] = 0
LEADS_PER_RUN: Final[int] = 241
"""Hourly leads 0 to 240 h inclusive, requested as `forecast_hours=241`."""

RADIATION_VARIABLES: Final[tuple[str, ...]] = ("shortwave_radiation", "direct_radiation")
"""Null at lead 0, because a mean over the hour ending at the run time has no data yet."""

TRAILING_DAYS_MAY_BE_INCOMPLETE: Final[int] = 2
"""The newest run days may not be published in full yet; a shortfall there is skipped, not fatal."""

REQUEST_SLEEP_SECONDS: Final[float] = 0.5
REQUEST_TIMEOUT_SECONDS: Final[float] = 120.0
MAX_ATTEMPTS: Final[int] = 5
"""Attempts per request for a transport failure or an HTTP 5xx."""

MAX_CONSECUTIVE_REFUSALS: Final[int] = 5
NOT_AVAILABLE_PHRASE: Final[str] = "requested model run is not available"
"""Substring of the reason in the API's HTTP 400 refusal for a run it does not hold."""

MAX_REASON_CHARS: Final[int] = 100
DECIMAL_NUMBER: Final[re.Pattern[str]] = re.compile(r"-?\d+\.\d+")
"""A refusal reason can quote a request parameter back, and a coordinate is a decimal number, so
`_safe_reason` masks every decimal number and truncates."""

WIND_SPEED_UNIT: Final[str] = "km/h"
RADIATION_UNITS: Final[frozenset[str]] = frozenset({"W/m\u00b2", "W/m2"})
"""The units the API must report in `hourly_units`; anything else fails loudly."""


class RunNotAvailableError(RuntimeError):
    """The API refused the run with HTTP 400 "requested model run is not available"."""


class IncompleteRunError(RuntimeError):
    """The API returned a run with missing leads or null values."""


class RateLimitedError(RuntimeError):
    """The API answered HTTP 429. The script stops and resumes from its checkpoints."""


class TooManyRefusalsError(RuntimeError):
    """`MAX_CONSECUTIVE_REFUSALS` runs in a row were refused."""


@dataclass
class FetchProgress:
    """The mutable counters shared by every month of one invocation."""

    runs_remaining: int | None
    """Runs (fetched or refused) still allowed by `--max-runs`, or `None` for no limit."""
    consecutive_refusals: int = 0
    n_fetched: int = 0


def _safe_reason(*, text: str) -> str:
    """Return `text` with decimal numbers masked, cut to `MAX_REASON_CHARS`."""
    return DECIMAL_NUMBER.sub("<number>", text)[:MAX_REASON_CHARS]


def _read_reason(*, refusal: urllib.error.HTTPError) -> str:
    """Return the `reason` of an error response, or a note that the body was not JSON."""
    body = refusal.read()
    try:
        parsed = json.loads(body or b"{}")
    except ValueError:
        return "response body was not JSON"
    return str(parsed.get("reason", "no reason given")) if isinstance(parsed, dict) else "no reason"


def _get_json(*, url: str) -> Any:
    """Fetch one URL, retrying 5xx and transport failures, and classifying every refusal.

    Args:
        url: The full request URL, which carries coordinates and the key and is never logged.

    Returns:
        The decoded JSON payload.

    Raises:
        RunNotAvailableError: On HTTP 400 whose reason contains `NOT_AVAILABLE_PHRASE`.
        RateLimitedError: On HTTP 429.
        RuntimeError: On any other refusal, a non-JSON success body, or `MAX_ATTEMPTS` failures.
    """
    for attempt in range(MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as refusal:
            reason = _read_reason(refusal=refusal)
            if refusal.code == 429:
                msg = "Open-Meteo answered HTTP 429 (rate limited)"
                raise RateLimitedError(msg) from None
            if refusal.code == 400 and NOT_AVAILABLE_PHRASE in reason.lower():
                msg = "HTTP 400: requested model run is not available"
                raise RunNotAvailableError(msg) from None
            if refusal.code >= 500:
                _LOG.warning("attempt %d failed with HTTP %d, retrying", attempt + 1, refusal.code)
                time.sleep(5.0 * (attempt + 1))
                continue
            shown = _safe_reason(text=reason)
            msg = f"Open-Meteo refused the request with HTTP {refusal.code}: {shown}"
            raise RuntimeError(msg) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as failure:
            _LOG.warning("attempt %d failed (%s), retrying", attempt + 1, type(failure).__name__)
            time.sleep(5.0 * (attempt + 1))
        else:
            if isinstance(payload, dict) and payload.get("error"):
                shown = _safe_reason(text=str(payload.get("reason")))
                msg = f"Open-Meteo refused the request: {shown}"
                raise RuntimeError(msg)
            return payload
    msg = f"Open-Meteo failed after {MAX_ATTEMPTS} attempts"
    raise RuntimeError(msg)


def _require_api_key() -> str:
    """Return the Open-Meteo API key, never logging it.

    Returns:
        The key.

    Raises:
        RuntimeError: If `OPEN_METEO_API_KEY` is neither exported nor in `.env`.
    """
    key = open_meteo_api_key()
    if not key:
        msg = "export OPEN_METEO_API_KEY into the environment first (its value is never logged)"
        raise RuntimeError(msg)
    return key


def _sites() -> pl.DataFrame:
    """Return the nine study sites with the `cell_selection` each group is queried with.

    Returns:
        One row per site: `site`, `latitude`, `longitude`, `cell_selection`. Held in memory only.
    """
    return pl.concat(
        [
            _pv_sites().with_columns(cell_selection=pl.lit("nearest")),
            _wind_sites().with_columns(cell_selection=pl.lit("land")),
        ]
    )


def _run_time(*, run_date: date) -> datetime:
    """Return the run's `init_time` as a naive UTC datetime."""
    return datetime(run_date.year, run_date.month, run_date.day, RUN_HOUR_UTC, tzinfo=UTC).replace(
        tzinfo=None
    )


def fetch_run_group(*, sites: pl.DataFrame, run_date: date) -> pl.DataFrame:
    """Fetch every lead of one run for one `cell_selection` group of sites, in one request.

    Args:
        sites: One `cell_selection` group of the roster, carrying `site`, `latitude`, `longitude`,
            and `cell_selection`.
        run_date: The day of the 00 UTC run.

    Returns:
        One row per (site, lead), with `site`, `init_time`, `valid_time`, `lead_hours`, and one
        `Float64` column per entry of `BASE_VARIABLES`. Coordinates never enter the frame.

    Raises:
        RunNotAvailableError: If the API refuses the run as not available.
        RuntimeError: If the response does not carry one block per site, a block is out of
            position or in unexpected units, or the API refuses the request for any other reason.
    """
    init_time = _run_time(run_date=run_date)
    cell_selection = sites["cell_selection"][0]
    request = (
        f"{SINGLE_RUNS_URL}"
        f"?latitude={','.join(str(value) for value in sites['latitude'])}"
        f"&longitude={','.join(str(value) for value in sites['longitude'])}"
        f"&run={init_time.strftime('%Y-%m-%dT%H:%M')}&forecast_hours={LEADS_PER_RUN}"
        f"&hourly={','.join(BASE_VARIABLES)}&models={MODELS_PARAMETER}&timezone=UTC"
        f"&cell_selection={cell_selection}&apikey={_require_api_key()}"
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
            {"site": site, "valid_time": block["hourly"]["time"]}
            | {name: block["hourly"][name] for name in BASE_VARIABLES},
            schema_overrides=dict.fromkeys(BASE_VARIABLES, pl.Float64),
        )
        for site, block in zip(sites["site"], blocks, strict=True)
    )
    return frame.with_columns(
        pl.col("valid_time").str.to_datetime("%Y-%m-%dT%H:%M", time_unit="us"),
        init_time=pl.lit(init_time, dtype=pl.Datetime("us")),
    ).with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("init_time")).dt.total_hours().cast(pl.Int32)
    )


def _check_block(*, block: dict[str, Any], position: int, n_blocks: int) -> None:
    """Fail loudly unless one response block is in the position its label assumes, in known units.

    The label of a site comes from the block's position, so a reordered response would swap two
    anonymised labels without any error. Every block of a multi-location response carries a
    `location_id`, which must equal its position.

    Args:
        block: One location's block of the response.
        position: The block's index in the response list.
        n_blocks: How many blocks the response holds.

    Raises:
        RuntimeError: If `location_id` differs from the position, or `hourly_units` differs from
            `WIND_SPEED_UNIT` for wind speed or `RADIATION_UNITS` for radiation.
    """
    if n_blocks > 1 and block.get("location_id") != position:
        msg = (
            f"response block {position} is out of order or has no location_id, so site labels "
            "could be swapped: inspect the response layout (not the key or the coordinates)"
        )
        raise RuntimeError(msg)
    units = block["hourly_units"]
    for name in BASE_VARIABLES:
        if name.startswith("wind_speed") and units[name] != WIND_SPEED_UNIT:
            msg = f"{name} is reported in {units[name]!r}, expected {WIND_SPEED_UNIT!r}"
            raise RuntimeError(msg)
        if name in RADIATION_VARIABLES and units[name] not in RADIATION_UNITS:
            msg = f"{name} is reported in {units[name]!r}, expected W/m2"
            raise RuntimeError(msg)


def _shortfall(*, frame: pl.DataFrame) -> str | None:
    """Describe why a run is incomplete, or return `None` if it is complete.

    Args:
        frame: One run's rows for every site.

    Returns:
        A sentence naming the shortfall, without any coordinate, or `None`.
    """
    per_site = frame.group_by("site").agg(pl.col("lead_hours").sort().alias("leads"))
    expected_leads = list(range(LEADS_PER_RUN))
    wrong = per_site.filter(pl.col("leads").list.len() != LEADS_PER_RUN)
    if not wrong.is_empty():
        return f"sites {sorted(wrong['site'])} do not have {LEADS_PER_RUN} leads"
    if any(leads.to_list() != expected_leads for leads in per_site["leads"]):
        return f"lead hours are not 0..{LEADS_PER_RUN - 1} in one-hour steps"
    for name in BASE_VARIABLES:
        column = (
            frame.filter(pl.col("lead_hours") > 0)[name]
            if name in RADIATION_VARIABLES
            else frame[name]
        )
        if column.is_null().any():
            return f"{name} has nulls"
    return None


def fetch_run(*, sites: pl.DataFrame, run_date: date) -> pl.DataFrame:
    """Fetch one complete run for every site, one request per `cell_selection` group.

    Args:
        sites: The roster with `cell_selection`.
        run_date: The day of the 00 UTC run.

    Returns:
        The run for every site, sorted by `site` and `lead_hours`.

    Raises:
        IncompleteRunError: If the run fails the completeness check in `_shortfall`.
    """
    frame = pl.concat(
        _fetch_and_sleep(sites=group, run_date=run_date)
        for group in sites.partition_by("cell_selection", maintain_order=True)
    ).sort("site", "lead_hours")
    reason = _shortfall(frame=frame)
    if reason is not None:
        msg = f"run {run_date} is incomplete: {reason}"
        raise IncompleteRunError(msg)
    return frame


def _fetch_and_sleep(*, sites: pl.DataFrame, run_date: date) -> pl.DataFrame:
    """Fetch one group, then pause so requests stay sequential and paced."""
    frame = fetch_run_group(sites=sites, run_date=run_date)
    time.sleep(REQUEST_SLEEP_SECONDS)
    return frame


def _month_paths(*, month: str) -> tuple[Path, Path]:
    """Return the complete-month and partial-month parquet paths for `YYYY-MM`."""
    return PRODUCT_DIR / f"{month}.parquet", PRODUCT_DIR / f"{month}.partial.parquet"


def _read_ledger(*, filename: str) -> dict[date, str]:
    """Return a ledger of run days to reasons, or an empty one if the file does not exist."""
    path = PRODUCT_DIR / filename
    if not path.exists():
        return {}
    return {date.fromisoformat(day): reason for day, reason in json.loads(path.read_text()).items()}


def _write_ledger(*, ledger: dict[date, str], filename: str) -> None:
    """Write a ledger of run days to reasons atomically. It holds dates and reasons only."""
    path = PRODUCT_DIR / filename
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps({day.isoformat(): ledger[day] for day in sorted(ledger)}))
    temporary.rename(path)


def _write_atomically(*, frame: pl.DataFrame, path: Path) -> None:
    """Write `frame` to `<path>.tmp` and rename it, so a kill never leaves a truncated file."""
    temporary = path.with_name(path.name + ".tmp")
    frame.write_parquet(temporary)
    temporary.rename(path)


def _month_days(*, month: str, first: date, last: date) -> list[date]:
    """Return the run days of `YYYY-MM` that fall inside `first`..`last`."""
    start = date.fromisoformat(f"{month}-01")
    days = []
    day = start
    while day.month == start.month:
        if first <= day <= last:
            days.append(day)
        day += timedelta(days=1)
    return days


def _months(*, first: date, last: date) -> list[str]:
    """Return every `YYYY-MM` from `first`'s month to `last`'s month."""
    months = []
    year, month = first.year, first.month
    while (year, month) <= (last.year, last.month):
        months.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def _fetch_month(
    *,
    month: str,
    sites: pl.DataFrame,
    first: date,
    last: date,
    unavailable: dict[date, str],
    incomplete: dict[date, str],
    progress: FetchProgress,
) -> None:
    """Fetch every missing run of one month, checkpointing after each run.

    Args:
        month: The month as `YYYY-MM`.
        sites: The roster with `cell_selection`.
        first: The first run day wanted.
        last: The last run day wanted.
        unavailable: The run days refused by the API, mapped to a reason. Updated in place.
        incomplete: The historical run days that came back incomplete, mapped to the shortfall.
            Updated in place.
        progress: The counters shared by every month. Updated in place.

    Raises:
        TooManyRefusalsError: If `MAX_CONSECUTIVE_REFUSALS` runs in a row are refused.
    """
    complete_path, partial_path = _month_paths(month=month)
    days = _month_days(month=month, first=first, last=last)
    if complete_path.exists():
        return
    cache = pl.read_parquet(partial_path) if partial_path.exists() else None
    fetched_days = (
        set(cache["init_time"].dt.date().unique().to_list()) if cache is not None else set()
    )
    trailing_from = datetime.now(UTC).date() - timedelta(days=TRAILING_DAYS_MAY_BE_INCOMPLETE)
    for day in days:
        if day in fetched_days or day in unavailable or day in incomplete:
            continue
        if progress.runs_remaining is not None and progress.runs_remaining <= 0:
            break
        started = time.monotonic()
        try:
            frame = fetch_run(sites=sites, run_date=day)
        except RunNotAvailableError:
            if day > trailing_from:
                _LOG.info("run %s not published yet, skipping", day)
                continue
            _count_run(progress=progress)
            progress.consecutive_refusals += 1
            _LOG.warning("run %s refused as not available: recorded as a gap", day)
            unavailable[day] = "HTTP 400: requested model run is not available"
            _write_ledger(ledger=unavailable, filename=UNAVAILABLE_FILENAME)
            if progress.consecutive_refusals >= MAX_CONSECUTIVE_REFUSALS:
                msg = (
                    f"{MAX_CONSECUTIVE_REFUSALS} runs in a row were refused, the last one {day}: "
                    "stopping. Check the request settings and the archive's start before resuming."
                )
                raise TooManyRefusalsError(msg) from None
            continue
        except IncompleteRunError as shortfall:
            if day > trailing_from:
                _LOG.info("run %s not complete yet, skipping", day)
                continue
            _count_run(progress=progress)
            _LOG.warning("run %s is incomplete: recorded as a gap and skipped", day)
            incomplete[day] = _safe_reason(text=str(shortfall))
            _write_ledger(ledger=incomplete, filename=INCOMPLETE_FILENAME)
            continue
        _count_run(progress=progress)
        progress.consecutive_refusals = 0
        progress.n_fetched += 1
        cache = frame if cache is None else pl.concat([cache, frame]).sort("init_time", "site")
        fetched_days.add(day)
        _write_atomically(frame=cache, path=partial_path)
        _LOG.info(
            "run %s: 2 requests, %.1f s, %d rows", day, time.monotonic() - started, frame.height
        )
    every_day = _month_days(month=month, first=FIRST_RUN_DATE, last=date.max)
    if (
        cache is not None
        and set(every_day) <= fetched_days | unavailable.keys() | incomplete.keys()
    ):
        partial_path.rename(complete_path)
        _LOG.info("%s: month complete", month)


def _count_run(*, progress: FetchProgress) -> None:
    """Count one attempted run (fetched, refused, or incomplete) against `--max-runs`."""
    if progress.runs_remaining is not None:
        progress.runs_remaining -= 1


def _combine() -> pl.DataFrame:
    """Read every monthly file (complete or partial) into one sorted frame.

    Returns:
        All fetched runs, sorted by `init_time`, `site`, `lead_hours`.

    Raises:
        RuntimeError: If no run has been fetched.
    """
    paths = sorted(PRODUCT_DIR.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9].parquet")) + sorted(
        PRODUCT_DIR.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9].partial.parquet")
    )
    if not paths:
        msg = "no run has been fetched"
        raise RuntimeError(msg)
    return pl.concat(pl.read_parquet(path) for path in paths).sort(
        "init_time", "site", "lead_hours"
    )


def _column_descriptions() -> dict[str, str]:
    """Return the README's one-line description of every column."""
    return {
        "site": "Anonymised meter label (`A`-`F` for solar, `W1`-`W3` for wind), not a coordinate.",
        "init_time": "The model run's 00 UTC start, UTC, timezone-naive.",
        "valid_time": "The hour the values describe, UTC, timezone-naive.",
        "lead_hours": "`valid_time - init_time` in hours, 0 to 240, `Int32`.",
        "shortwave_radiation": "Global horizontal irradiance, W/m^2, mean over the hour ending at "
        "`valid_time` (null at lead 0).",
        "direct_radiation": "Direct (beam) horizontal irradiance, W/m^2, averaged as above "
        "(null at lead 0).",
        "temperature_2m": "Air temperature at 2 m, degC.",
        "wind_speed_10m": "Wind speed at 10 m, km/h (the API's default unit).",
        "wind_direction_10m": "Wind direction at 10 m, degrees.",
        "wind_speed_100m": "Wind speed at 100 m, km/h (the API's default unit).",
        "wind_direction_100m": "Wind direction at 100 m, degrees.",
    }


def _write_docs(
    *,
    frame: pl.DataFrame,
    sites: pl.DataFrame,
    unavailable: dict[date, str],
    incomplete: dict[date, str],
) -> None:
    """Write `lineage.json` and `README.md` from the combined frame."""
    runs = frame["init_time"].unique().sort()
    lead_one_hour = frame.filter(pl.col("lead_hours") > 0).select(
        "site", pl.col("valid_time").alias("time"), "shortwave_radiation"
    )
    timestamp = _check_timestamp_convention(frame=lead_one_hour, sites=sites)
    write_lineage_note(
        product_dir=PRODUCT_DIR,
        source_address=SINGLE_RUNS_URL,
        request_description=(
            f"ECMWF IFS HRES, models={MODELS_PARAMETER}, hourly={','.join(BASE_VARIABLES)}, "
            f"forecast_hours={LEADS_PER_RUN}, the {RUN_HOUR_UTC:02d} UTC run of each day, "
            f"{sites.height} meter sites, two requests per run (cell_selection nearest for PV, "
            "land for wind)"
        ),
        variables=list(BASE_VARIABLES),
        extra={
            "n_sites": sites.height,
            "row_count": frame.height,
            "n_runs": runs.len(),
            "first_init_time": str(runs.min()),
            "last_init_time": str(runs.max()),
            "unavailable_runs": sorted(day.isoformat() for day in unavailable),
            "incomplete_runs": {day.isoformat(): incomplete[day] for day in sorted(incomplete)},
            "note": (
                "Clear-sky correlation of shortwave_radiation at leads 1-240 against the label "
                "itself, the label shifted 30 minutes earlier, and the label shifted 30 minutes "
                f"later: {timestamp}. The higher of the two shifted correlations names the "
                "averaging convention."
            ),
        },
    )
    write_readme(
        product_dir=PRODUCT_DIR,
        product_name="ECMWF IFS HRES 9 km, Single Runs, 00 UTC daily",
        source_web_page="https://open-meteo.com/en/docs/single-runs-api",
        script_path="studies/weather_downloads/fetch_open_meteo_single_runs.py",
        lineage_filenames=["lineage.json"],
        columns=_column_descriptions(),
        missing_value_convention=(
            "Polars null. Only `shortwave_radiation` and `direct_radiation` at `lead_hours` 0 are "
            "null, because a mean over the hour ending at the run time has no data yet. A run day "
            "the API does not hold, or one that was not complete, has no rows; the run days the "
            "API refused are in `lineage.json`'s `unavailable_runs` field."
        ),
        gotchas=[
            (
                "**Lead days.** Derive the as-issued lead day as `lead_hours // 24`, or select "
                "exact `lead_hours` in 0, 24, 48, 72, 120, 168, 240. Every run holds every lead, "
                "so there is one row per (site, run, lead)."
            ),
            (
                "**Model version changes.** The IFS cycle changed before and inside the span. "
                "ECMWF's IFS cycle 48r1 went live on 2023-06-27 06 UTC. Cycle 49r1 went live on "
                "2024-11-12, at 06 UTC on one project page and 12 UTC on another, so the hour is "
                "to be confirmed. Cycle 50r1 went live on 2026-05-12 06 UTC, together with AIFS "
                "v2. These dates are read from ECMWF pages, not verified in this data. Assign a "
                "model era from `init_time` when comparing across the span."
            ),
            (
                "**Archive start.** The archive starts on 2024-03-14, and Open-Meteo labels the "
                "runs from that date as cycle 49r1 hindcasts, which predates the operational 49r1 "
                "date above. Whether the start reflects a grid or source change is unverified."
            ),
            (
                "**Hourly values at long leads are not native.** IFS HRES output steps coarsen at "
                "longer leads, and Open-Meteo interpolates or disaggregates them to hourly. "
                "Hourly values at lead days 5, 7, and 10 are therefore not native model output."
            ),
            (
                "**Horizon.** The IFS HRES horizon is 10 days, so there is no lead day 14. AIFS "
                "is not substituted."
            ),
            (
                "**Cell selection.** The six PV sites use `cell_selection=nearest` and the three "
                "wind sites use `cell_selection=land`. Two sites whose selected 9 km cell is the "
                "same carry identical series, so identical series for a pair of site labels are "
                "expected and are not a defect."
            ),
            (
                "**Units.** The API's defaults: wind speeds in km/h, temperature in degC. "
                "The fetch fails if `hourly_units` reports another unit for wind speed or "
                "radiation. "
                "Radiation is a mean over the hour ending at `valid_time`; `lineage.json`'s "
                "`note` records the convention measured from the data."
            ),
        ],
        external_docs={
            "Open-Meteo Single Runs API": "https://open-meteo.com/en/docs/single-runs-api",
            "ECMWF IFS documentation": (
                "https://www.ecmwf.int/en/forecasts/documentation-and-support"
            ),
        },
    )


def main() -> int:
    """Fetch every missing 00 UTC run, then write the combined file, lineage, and README."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=FIRST_RUN_DATE)
    parser.add_argument("--end", type=date.fromisoformat, default=datetime.now(UTC).date())
    parser.add_argument("--max-runs", type=int, default=None)
    arguments = parser.parse_args()
    _require_api_key()

    sites = _sites()
    PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
    progress = FetchProgress(runs_remaining=arguments.max_runs)
    unavailable = _read_ledger(filename=UNAVAILABLE_FILENAME)
    incomplete = _read_ledger(filename=INCOMPLETE_FILENAME)
    started = time.monotonic()
    try:
        for month in _months(first=arguments.start, last=arguments.end):
            _fetch_month(
                month=month,
                sites=sites,
                first=arguments.start,
                last=arguments.end,
                unavailable=unavailable,
                incomplete=incomplete,
                progress=progress,
            )
            if progress.runs_remaining is not None and progress.runs_remaining <= 0:
                break
    except (RateLimitedError, TooManyRefusalsError) as stop:
        _LOG.error(
            "stopped after %d runs: %s. Re-run the same command to resume.",
            progress.n_fetched,
            stop,
        )
        return 1
    _LOG.info("fetched %d runs in %.1f s", progress.n_fetched, time.monotonic() - started)
    frame = _combine()
    _write_atomically(frame=frame, path=PRODUCT_DIR / COMBINED_FILENAME)
    _write_docs(frame=frame, sites=sites, unavailable=unavailable, incomplete=incomplete)
    return 0


if __name__ == "__main__":
    sys.exit(main())
