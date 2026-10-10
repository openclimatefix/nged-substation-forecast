"""Download the solar-relevant variables of Open-Meteo's Single Runs archive of ECMWF IFS HRES.

One-off throwaway script for the follow-up to the ERA5 variable-ladder study planned in
<https://github.com/openclimatefix/nged-substation-forecast/pull/1097>. The ladder asks which ERA5
variables help predict solar farm output. ERA5 is a reanalysis, so the follow-up repeats a shortened
ladder on past IFS forecasts at matched lead times. `fetch_open_meteo_single_runs.py` already holds
the radiation, temperature, and wind of the same runs for nine sites. This script fetches 21
variables for the six solar farms only, from the same 00 UTC run of every day, in one request per
run. Sixteen are new (three cloud layers, total cloud, dew point, pressure,
boundary-layer height, total column water vapour, CAPE, convective inhibition, visibility, skin
temperature, snow depth, snowfall, precipitation, and wind gust). Four repeat the sibling's
(`shortwave_radiation`, `direct_radiation`, `temperature_2m`, and `wind_speed_10m`), so that the
validation can compare direct with shortwave radiation, and so that this product stands alone.

**Reused machinery.** The request, retry, and refusal handling and the small helpers are imported
from `fetch_open_meteo_single_runs.py`. The month-file, ledger, and combine functions are copies of
that script's with this product's directory, because the original reads module constants. The
variable list, the completeness rule, the response checks, and the validation are this script's
own.

**Requests.** One request per run: the six solar sites with `cell_selection=nearest`, the setting
the sibling per-site fetchers use. The sites come from the private site list at run time and stay in
memory. Rows carry only the anonymised `site` label. No coordinate, and never the API key, reaches a
log line, a file, or the lineage note; `OPEN_METEO_TOKEN` must be exported into the environment.

**A run is stored only if it is complete.** Every site must have `LEADS_PER_RUN` hourly leads and
every variable must be non-null, except the variables whose lead 0 is null by the averaging
convention (`LEAD_ZERO_NULL_VARIABLES`) and convective inhibition, which the API leaves null where
it is undefined (`MAY_BE_NULL_VARIABLES`). A refused or incomplete run goes to a ledger of dates,
and the backfill continues. A trailing run that is not yet published or complete is skipped and
retried on the next invocation. HTTP 429 stops the script cleanly, and it resumes from the
checkpoints.

**Validation.** After the last run the script checks the combined frame against plausible ranges,
the row count, duplicate keys, and the null pattern, and it writes the result into `lineage.json`. A
value outside its range stops the script before it writes the combined file.

**Resuming.** One parquet per month, `YYYY-MM.parquet` once every run day of the month is fetched or
documented, and `YYYY-MM.partial.parquet` otherwise, rewritten atomically after every run.

Run it with `uv run python studies/weather_downloads/fetch_open_meteo_ifs_solar_variables.py`. Pass
`--max-runs 4` to fetch only the first four missing runs, which is the pilot.
"""

import argparse
import json
import logging
import sys
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import polars as pl
from fetch_open_meteo_previous_runs import _pv_sites
from fetch_open_meteo_single_runs import (
    FIRST_RUN_DATE,
    LEADS_PER_RUN,
    MAX_CONSECUTIVE_REFUSALS,
    MODELS_PARAMETER,
    REQUEST_SLEEP_SECONDS,
    SINGLE_RUNS_URL,
    TRAILING_DAYS_MAY_BE_INCOMPLETE,
    FetchProgress,
    IncompleteRunError,
    RateLimitedError,
    RunNotAvailableError,
    TooManyRefusalsError,
    _count_run,
    _get_json,
    _month_days,
    _months,
    _require_api_key,
    _run_time,
    _safe_reason,
    _write_atomically,
)
from lineage import write_lineage_note, write_readme
from studies.sources import ECMWF_IFS_SINGLE_RUNS_SOLAR_PRODUCT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("fetch_open_meteo_ifs_solar_variables")

PRODUCT_DIR: Final[Path] = ECMWF_IFS_SINGLE_RUNS_SOLAR_PRODUCT_DIR
COMBINED_FILENAME: Final[str] = "ECMWF-IFS-SINGLE-RUNS-SOLAR.parquet"
UNAVAILABLE_FILENAME: Final[str] = "_unavailable_runs.json"
INCOMPLETE_FILENAME: Final[str] = "_incomplete_runs.json"

# name: (unit the API must report, smallest plausible value, largest plausible value)
VARIABLES: Final[dict[str, tuple[str, float, float]]] = {
    "cloud_cover": ("%", 0.0, 100.0),
    "cloud_cover_low": ("%", 0.0, 100.0),
    "cloud_cover_mid": ("%", 0.0, 100.0),
    "cloud_cover_high": ("%", 0.0, 100.0),
    "shortwave_radiation": ("W/m²", -1.0, 1200.0),
    "direct_radiation": ("W/m²", -1.0, 1100.0),
    "temperature_2m": ("°C", -25.0, 45.0),
    "dew_point_2m": ("°C", -30.0, 30.0),
    "surface_pressure": ("hPa", 900.0, 1100.0),
    "boundary_layer_height": ("m", 0.0, 6000.0),
    "total_column_integrated_water_vapour": ("kg/m²", 0.0, 70.0),
    "cape": ("J/kg", 0.0, 6000.0),
    "convective_inhibition": ("J/kg", -1500.0, 1500.0),
    "visibility": ("m", 0.0, 200_000.0),
    "surface_temperature": ("°C", -30.0, 70.0),
    "snow_depth": ("m", 0.0, 3.0),
    "snowfall": ("cm", 0.0, 10.0),
    "precipitation": ("mm", 0.0, 60.0),
    "wind_speed_10m": ("km/h", 0.0, 200.0),
    "wind_gusts_10m": ("km/h", 0.0, 300.0),
}
"""The 20 variables requested, each with the unit the API must report and a plausible range.

The names are Open-Meteo's. Each name was probed on 2026-10-10 against the Single Runs API for the
runs of 2024-04-10 and 2025-06-15 and returned 241 hourly values. `cloud_base`, `albedo`,
`freezing_level_height`, and `lifted_index` returned only nulls and are not requested.
`diffuse_radiation` is not requested: the API derives it as shortwave minus direct radiation, so
it adds no information and goes negative where direct exceeds shortwave."""

LEAD_ZERO_NULL_VARIABLES: Final[frozenset[str]] = frozenset(
    {
        "shortwave_radiation",
        "direct_radiation",
        "snowfall",
        "precipitation",
        "wind_gusts_10m",
    }
)
"""Null at lead 0, because an hourly mean or accumulation ending at the run time has no data yet."""

MAY_BE_NULL_VARIABLES: Final[frozenset[str]] = frozenset({"convective_inhibition"})
"""Null wherever convective inhibition is undefined, which is most hours."""

EXPECTED_SITES: Final[int] = 6

RADIATION_TOLERANCE: Final[float] = 5.0
"""How far, in W/m², direct radiation may exceed shortwave radiation before the row is counted."""

DEW_POINT_TOLERANCE: Final[float] = 0.5
"""How far, in °C, the dew point may exceed the air temperature before the row is counted."""

MAX_CROSS_CHECK_SHARE: Final[float] = 0.02
"""The share of rows that may break a relation between variables before validation fails."""


def _sites() -> pl.DataFrame:
    """Return the six solar sites as `site`, `latitude`, `longitude`, held in memory only."""
    return _pv_sites().select("site", "latitude", "longitude")


def _check_block(*, block: dict[str, Any], position: int, n_blocks: int) -> None:
    """Fail loudly unless one response block is in its expected position and in the expected units.

    The label of a site comes from the block's position, so a reordered response would swap two
    anonymised labels without any error. Every block after the first carries a `location_id` equal
    to its position, and Open-Meteo omits it from the first block.

    Args:
        block: One location's block of the response.
        position: The block's index in the response list.
        n_blocks: How many blocks the response holds.

    Raises:
        RuntimeError: If `location_id` differs from the position, or a unit differs from
            `VARIABLES`.
    """
    if n_blocks > 1 and block.get("location_id", 0 if position == 0 else None) != position:
        msg = (
            f"response block {position} is out of order or has no location_id, so site labels "
            "could be swapped: inspect the response layout (not the key or the coordinates)"
        )
        raise RuntimeError(msg)
    units = block["hourly_units"]
    for name, (unit, _, _) in VARIABLES.items():
        if units[name] != unit:
            msg = f"{name} is reported in {units[name]!r}, expected {unit!r}"
            raise RuntimeError(msg)


def fetch_run(*, sites: pl.DataFrame, run_date: date) -> pl.DataFrame:
    """Fetch every lead of one run for every site, in one request, and check it is complete.

    Args:
        sites: The site list with `site`, `latitude`, and `longitude`.
        run_date: The day of the 00 UTC run.

    Returns:
        One row per (site, lead), with `site`, `init_time`, `valid_time`, `lead_hours`, and one
        `Float64` column per entry of `VARIABLES`. Coordinates never enter the frame.

    Raises:
        RunNotAvailableError: If the API refuses the run as not available.
        IncompleteRunError: If a lead or a required value is missing.
        RuntimeError: If the response is out of position, in unexpected units, or refused.
    """
    init_time = _run_time(run_date=run_date)
    request = (
        f"{SINGLE_RUNS_URL}"
        f"?latitude={','.join(str(value) for value in sites['latitude'])}"
        f"&longitude={','.join(str(value) for value in sites['longitude'])}"
        f"&run={init_time.strftime('%Y-%m-%dT%H:%M')}&forecast_hours={LEADS_PER_RUN}"
        f"&hourly={','.join(VARIABLES)}&models={MODELS_PARAMETER}&timezone=UTC"
        f"&cell_selection=nearest&apikey={_require_api_key()}"
    )
    payload = _get_json(url=request)
    time.sleep(REQUEST_SLEEP_SECONDS)
    blocks = payload if isinstance(payload, list) else [payload]
    if len(blocks) != sites.height:
        msg = f"asked for {sites.height} sites and got {len(blocks)} blocks"
        raise RuntimeError(msg)
    for position, block in enumerate(blocks):
        _check_block(block=block, position=position, n_blocks=len(blocks))
    frame = (
        pl.concat(
            pl.DataFrame(
                {"site": site, "valid_time": block["hourly"]["time"]}
                | {name: block["hourly"][name] for name in VARIABLES},
                schema_overrides=dict.fromkeys(VARIABLES, pl.Float64),
            )
            for site, block in zip(sites["site"], blocks, strict=True)
        )
        .with_columns(
            pl.col("valid_time").str.to_datetime("%Y-%m-%dT%H:%M", time_unit="us"),
            init_time=pl.lit(init_time, dtype=pl.Datetime("us")),
        )
        .with_columns(
            lead_hours=(pl.col("valid_time") - pl.col("init_time")).dt.total_hours().cast(pl.Int32)
        )
        .sort("site", "lead_hours")
    )
    reason = shortfall(frame=frame)
    if reason is not None:
        msg = f"run {run_date} is incomplete: {reason}"
        raise IncompleteRunError(msg)
    return frame


def shortfall(*, frame: pl.DataFrame) -> str | None:
    """Describe why a run is incomplete, or return `None` if it is complete.

    Args:
        frame: One run's rows for every site.

    Returns:
        A sentence naming the shortfall, without any coordinate, or `None`.
    """
    per_site = frame.group_by("site").agg(pl.col("lead_hours").sort().alias("leads"))
    expected_leads = list(range(LEADS_PER_RUN))
    if per_site.height != EXPECTED_SITES:
        return f"{per_site.height} sites, expected {EXPECTED_SITES}"
    if any(leads.to_list() != expected_leads for leads in per_site["leads"]):
        return f"leads are not 0..{LEADS_PER_RUN - 1} in one-hour steps at every site"
    for name in VARIABLES:
        if name in MAY_BE_NULL_VARIABLES:
            continue
        column = (
            frame.filter(pl.col("lead_hours") > 0)[name]
            if name in LEAD_ZERO_NULL_VARIABLES
            else frame[name]
        )
        if column.is_null().any():
            return f"{name} has nulls"
    return None


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


def _record_gap(
    *, day: date, reason: str, ledger: dict[date, str], filename: str, progress: FetchProgress
) -> None:
    """Record a run day the API refused or returned incomplete, and stop after a streak of them.

    Args:
        day: The run day.
        reason: Why the run has no rows. It holds no coordinate.
        ledger: The ledger to add the day to. Updated in place.
        filename: The ledger's file.
        progress: The counters shared by every month. Updated in place.

    Raises:
        TooManyRefusalsError: If `MAX_CONSECUTIVE_REFUSALS` runs in a row had no rows. A streak
            means the request is wrong, the archive has ended, or a variable is null on a newer
            model cycle, and not that many separate days are missing.
    """
    _count_run(progress=progress)
    progress.consecutive_refusals += 1
    _LOG.warning("run %s has no rows (%s): recorded as a gap", day, reason)
    ledger[day] = reason
    _write_ledger(ledger=ledger, filename=filename)
    if progress.consecutive_refusals >= MAX_CONSECUTIVE_REFUSALS:
        msg = (
            f"{MAX_CONSECUTIVE_REFUSALS} runs in a row had no rows, the last one {day}: stopping. "
            "Check the request, the archive's start, and the newest model cycle before resuming."
        )
        raise TooManyRefusalsError(msg)


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
        sites: The site list.
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
    if complete_path.exists():
        return
    cache = pl.read_parquet(partial_path) if partial_path.exists() else None
    fetched_days = (
        set(cache["init_time"].dt.date().unique().to_list()) if cache is not None else set()
    )
    trailing_from = datetime.now(UTC).date() - timedelta(days=TRAILING_DAYS_MAY_BE_INCOMPLETE)
    for day in _month_days(month=month, first=first, last=last):
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
            _record_gap(
                day=day,
                reason="HTTP 400: requested model run is not available",
                ledger=unavailable,
                filename=UNAVAILABLE_FILENAME,
                progress=progress,
            )
            continue
        except IncompleteRunError as problem:
            if day > trailing_from:
                _LOG.info("run %s not complete yet, skipping", day)
                continue
            # The ledger entry is permanent: a later invocation skips this run day. A transient
            # glitch is retried by deleting the run's entry from `_incomplete_runs.json`.
            _record_gap(
                day=day,
                reason=_safe_reason(text=str(problem)),
                ledger=incomplete,
                filename=INCOMPLETE_FILENAME,
                progress=progress,
            )
            continue
        _count_run(progress=progress)
        progress.consecutive_refusals = 0
        progress.n_fetched += 1
        cache = frame if cache is None else pl.concat([cache, frame]).sort("init_time", "site")
        fetched_days.add(day)
        _write_atomically(frame=cache, path=partial_path)
        _LOG.info("run %s: %.1f s, %d rows", day, time.monotonic() - started, frame.height)
    every_day = _month_days(month=month, first=FIRST_RUN_DATE, last=date.max)
    if (
        cache is not None
        and set(every_day) <= fetched_days | unavailable.keys() | incomplete.keys()
    ):
        partial_path.rename(complete_path)
        _LOG.info("%s: month complete", month)


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


def validate(*, frame: pl.DataFrame) -> dict[str, Any]:
    """Check the combined frame against the plausibility and structure checks.

    Args:
        frame: Every fetched run.

    Returns:
        What was measured: the row, run, and site counts and the null share of each variable.

    Raises:
        RuntimeError: If a key is duplicated, a run is short, a value lies outside its range, or a
            required variable has nulls outside its allowed positions.
    """
    keys = frame.select("site", "init_time", "lead_hours")
    if keys.is_duplicated().any():
        msg = "duplicate (site, init_time, lead_hours) keys"
        raise RuntimeError(msg)
    runs = frame["init_time"].n_unique()
    expected_rows = runs * EXPECTED_SITES * LEADS_PER_RUN
    if frame.height != expected_rows:
        msg = f"{frame.height} rows, expected {expected_rows} for {runs} runs"
        raise RuntimeError(msg)
    problems = []
    for name, (_, low, high) in VARIABLES.items():
        values = frame[name].drop_nulls()
        if values.is_empty():
            if name not in MAY_BE_NULL_VARIABLES:
                problems.append(f"{name} is all null")
            continue
        smallest, largest = float(values.min()), float(values.max())  # ty: ignore[invalid-argument-type]
        if smallest < low or largest > high:
            problems.append(f"{name} spans {smallest} to {largest}, outside {low} to {high}")
    if problems:
        msg = "implausible values: " + "; ".join(problems)
        raise RuntimeError(msg)
    null_share = {name: frame[name].null_count() / frame.height for name in VARIABLES}
    observed = {
        name: [frame[name].min(), frame[name].max()]
        for name in VARIABLES
        if not frame[name].drop_nulls().is_empty()
    }
    return {
        "rows": frame.height,
        "runs": runs,
        "null_share": null_share,
        "observed_range": observed,
        "radiation_check_failures": cross_check_failures(frame=frame),
    }


def cross_check_failures(*, frame: pl.DataFrame) -> dict[str, int]:
    """Count the rows that break a relation between variables, which per-variable ranges miss.

    A label swapped between two variables stays inside both variables' ranges, so the relations
    between them are checked as well.

    Args:
        frame: Every fetched run.

    Returns:
        The number of rows breaking each relation. The relations are direct radiation no larger
        than shortwave radiation and dew point no higher than air temperature, each with a
        tolerance. At leads beyond about 100 h, where the output steps coarsen, a few rows have
        direct radiation above shortwave radiation, so the check counts them and fails only if
        they exceed `MAX_CROSS_CHECK_SHARE`.

    Raises:
        RuntimeError: If more than `MAX_CROSS_CHECK_SHARE` of the rows break a relation.
    """
    radiation = frame.filter(pl.col("lead_hours") > 0)
    failures = {
        "direct_above_shortwave": radiation.filter(
            pl.col("direct_radiation") > pl.col("shortwave_radiation") + RADIATION_TOLERANCE
        ).height,
        "dew_point_above_temperature": frame.filter(
            pl.col("dew_point_2m") > pl.col("temperature_2m") + DEW_POINT_TOLERANCE
        ).height,
    }
    for name, count in failures.items():
        if count > MAX_CROSS_CHECK_SHARE * frame.height:
            msg = f"{count} of {frame.height} rows break {name}"
            raise RuntimeError(msg)
    return failures


def _write_docs(
    *,
    frame: pl.DataFrame,
    checks: dict[str, Any],
    unavailable: dict[date, str],
    incomplete: dict[date, str],
) -> None:
    """Write `lineage.json` and `README.md` from the combined frame."""
    runs = frame["init_time"].unique().sort()
    write_lineage_note(
        product_dir=PRODUCT_DIR,
        source_address=SINGLE_RUNS_URL,
        request_description=(
            f"ECMWF IFS HRES, models={MODELS_PARAMETER}, hourly={','.join(VARIABLES)}, "
            f"forecast_hours={LEADS_PER_RUN}, the 00 UTC run of each day, {EXPECTED_SITES} solar "
            "sites, one request per run (cell_selection nearest)"
        ),
        variables=list(VARIABLES),
        extra={
            "n_sites": EXPECTED_SITES,
            "row_count": frame.height,
            "n_runs": runs.len(),
            "first_init_time": str(runs.min()),
            "last_init_time": str(runs.max()),
            "unavailable_runs": sorted(day.isoformat() for day in unavailable),
            "incomplete_runs": {day.isoformat(): incomplete[day] for day in sorted(incomplete)},
            "null_share": checks["null_share"],
        },
    )
    columns = {
        "site": "Anonymised meter label (`A` to `F`), not a coordinate.",
        "init_time": "The model run's 00 UTC start, UTC, timezone-naive.",
        "valid_time": "The hour the values describe, UTC, timezone-naive.",
        "lead_hours": "`valid_time - init_time` in hours, 0 to 240, `Int32`.",
        **{name: f"Open-Meteo `{name}`, in {unit}." for name, (unit, _, _) in VARIABLES.items()},
    }
    write_readme(
        product_dir=PRODUCT_DIR,
        product_name="ECMWF IFS HRES 9 km, Single Runs, solar variables, 00 UTC daily",
        source_web_page="https://open-meteo.com/en/docs/single-runs-api",
        script_path="studies/weather_downloads/fetch_open_meteo_ifs_solar_variables.py",
        lineage_filenames=["lineage.json"],
        columns=columns,
        missing_value_convention=(
            "Polars null. `shortwave_radiation`, `direct_radiation`, "
            "`snowfall`, `precipitation`, and `wind_gusts_10m` are null at `lead_hours` 0 because "
            "a mean or accumulation over the hour ending at the run time has no data yet. "
            "`convective_inhibition` is null wherever it is undefined, which is most hours. A run "
            "day the API does not hold, or one that was not complete, has no rows; the run days "
            "are listed in `lineage.json`."
        ),
        gotchas=[
            (
                "**The same runs as `ECMWF-IFS-SINGLE-RUNS`.** That product holds radiation, "
                "temperature, and wind for nine sites. This product holds 20 variables for the six "
                "solar sites, and four of them (`shortwave_radiation`, `direct_radiation`, "
                "`temperature_2m`, `wind_speed_10m`) repeat that product's. To join the two on "
                "(`site`, `init_time`, `valid_time`), drop the four from one side first."
            ),
            (
                "**Model version changes.** ECMWF's IFS cycle 49r1 went live on 2024-11-12 (the "
                "hour is unconfirmed) and cycle 50r1 on 2026-05-12. Open-Meteo labels the runs "
                "from 2024-03-14 as 49r1 hindcasts, which predates the operational 49r1 date. "
                "Assign a model era from `init_time`, and cut folds within each era."
            ),
            (
                "**Hour conventions.** Radiation, precipitation, snowfall, and gusts are means or "
                "totals over the hour ending at `valid_time`. The other variables are values at "
                "`valid_time`. `ECMWF-IFS-SINGLE-RUNS/lineage.json` records the convention "
                "measured for radiation from the same runs (hour-ending)."
            ),
            (
                "**Hourly values at long leads are not native.** IFS HRES output steps coarsen at "
                "longer leads, and Open-Meteo interpolates or disaggregates them to hourly."
            ),
            (
                "**Radiation of exactly -1.0 W/m^2** occurs at leads 73 to 90 h in the sibling "
                "product, and may occur here. Clip radiation to 0 before use."
            ),
            (
                "**Not requested.** `cloud_base`, `albedo`, `freezing_level_height`, and "
                "`lifted_index` returned only nulls on 2026-10-10, and `diffuse_radiation` is "
                "derived as shortwave minus direct radiation and goes negative where direct "
                "exceeds shortwave. `relative_humidity_2m` follows "
                "from temperature and dew point."
            ),
        ],
        external_docs={
            "Open-Meteo Single Runs API": "https://open-meteo.com/en/docs/single-runs-api",
            "Open-Meteo ECMWF API": "https://open-meteo.com/en/docs/ecmwf-api",
        },
    )


def main() -> int:
    """Fetch every missing 00 UTC run, then validate and write the combined file and notes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=FIRST_RUN_DATE)
    parser.add_argument("--end", type=date.fromisoformat, default=datetime.now(UTC).date())
    parser.add_argument("--max-runs", type=int, default=None)
    arguments = parser.parse_args()
    _require_api_key()

    sites = _sites()
    if sites.height != EXPECTED_SITES:
        msg = f"the site list holds {sites.height} solar sites, expected {EXPECTED_SITES}"
        raise RuntimeError(msg)
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
    checks = validate(frame=frame)
    _LOG.info("validation passed: %s", json.dumps(checks, default=str))
    if arguments.max_runs is not None:
        _LOG.info("--max-runs was set, so the combined file and the notes are not written")
        return 0
    _write_atomically(frame=frame, path=PRODUCT_DIR / COMBINED_FILENAME)
    _write_docs(frame=frame, checks=checks, unavailable=unavailable, incomplete=incomplete)
    return 0


if __name__ == "__main__":
    sys.exit(main())
