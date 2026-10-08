"""Shared machinery for the GB electricity price and dispatch download scripts.

Written for the battery-versus-solar-PV study (PR 1094). `fetch_gb_prices.py` and
`fetch_bmu_dispatch.py` both import it. It holds the HTTP client with retry and backoff, the
resumable per-chunk cache, the Elexon settlement-period arithmetic, the expected-row-count and gap
helpers, and the `README.md` and `lineage.json` writers.

**Elexon settlement days follow UK clock time.** Settlement period 1 starts at 00:00 UK local time,
so a day has 48 periods, except the day the clocks go forward (46 periods) and the day they go back
(50 periods). Every timestamp this module returns is UTC.
"""

import json
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final
from zoneinfo import ZoneInfo

import httpx
import polars as pl

WINDOW_START: Final[date] = date(2025, 9, 1)
"""First day of the study window (a UTC day for streams, a settlement date for settlement data)."""

WINDOW_END: Final[date] = date(2026, 9, 30)
"""Last day of the study window, inclusive."""

ELEXON_API: Final[str] = "https://data.elexon.co.uk/bmrs/api/v1"
ELEXON_ATTRIBUTION: Final[str] = (
    "Contains BMRS data © Elexon Limited copyright and database right 2026"
)
ELEXON_LICENCE: Final[str] = (
    "Elexon's BMRS terms require the attribution line above. The wording is taken from the Elexon "
    "Insights Solution documentation and has not been independently verified."
)

FETCH_THREADS: Final[int] = 4
HTTP_TIMEOUT_S: Final[float] = 300.0
HTTP_RETRIES: Final[int] = 6
BACKOFF_BASE_S: Final[float] = 5.0
TOO_MANY_REQUESTS: Final[int] = 429
NOT_FOUND: Final[int] = 404
SETTLEMENT_PERIOD: Final[timedelta] = timedelta(minutes=30)
LONDON: Final[ZoneInfo] = ZoneInfo("Europe/London")
MAX_LISTED_GAPS: Final[int] = 50
"""The longest list of missing timestamps written into `lineage.json`; the count is always exact."""

QueryParams = list[tuple[str, str | float | None]]
"""Query parameters as a list of pairs, because Elexon repeats `bmUnit=` once per unit."""


def days_between(*, start: date, end: date) -> list[date]:
    """Return every day from `start` to `end`, both included."""
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def utc_midnight(*, day: date) -> datetime:
    """Return 00:00 UTC on `day`."""
    return datetime(year=day.year, month=day.month, day=day.day, tzinfo=UTC)


def settlement_day_start_utc(*, settlement_date: date) -> datetime:
    """Return the UTC instant at which settlement period 1 of `settlement_date` starts."""
    local_midnight = datetime(
        year=settlement_date.year,
        month=settlement_date.month,
        day=settlement_date.day,
        tzinfo=LONDON,
    )
    return local_midnight.astimezone(UTC)


def periods_in_settlement_day(*, settlement_date: date) -> int:
    """Return how many settlement periods `settlement_date` has: 46, 48, or 50."""
    start = settlement_day_start_utc(settlement_date=settlement_date)
    end = settlement_day_start_utc(settlement_date=settlement_date + timedelta(days=1))
    return int((end - start) / SETTLEMENT_PERIOD)


def period_start_utc(*, settlement_date: date, settlement_period: int) -> datetime:
    """Return the UTC instant at which a settlement period starts.

    Args:
        settlement_date: The settlement date.
        settlement_period: The period number, from 1 to the day's period count.

    Returns:
        The start of the period, in UTC.

    Raises:
        ValueError: If the period number does not exist on that settlement date.
    """
    if not 1 <= settlement_period <= periods_in_settlement_day(settlement_date=settlement_date):
        raise ValueError(
            f"Settlement period {settlement_period} does not exist on {settlement_date}"
        )
    start = settlement_day_start_utc(settlement_date=settlement_date)
    return start + (settlement_period - 1) * SETTLEMENT_PERIOD


def expected_settlement_starts(*, start: date, end: date) -> list[datetime]:
    """Return the UTC start of every settlement period of every settlement date in `start..end`."""
    return [
        period_start_utc(settlement_date=day, settlement_period=period)
        for day in days_between(start=start, end=end)
        for period in range(1, periods_in_settlement_day(settlement_date=day) + 1)
    ]


def expected_half_hour_starts(*, start: date, end: date) -> list[datetime]:
    """Return every half-hour start from 00:00 UTC on `start` to 23:30 UTC on `end`."""
    first = utc_midnight(day=start)
    count = (end - start).days * 48 + 48
    return [first + index * SETTLEMENT_PERIOD for index in range(count)]


def expected_hour_starts(*, start: date, end: date) -> list[datetime]:
    """Return every hour start from 00:00 UTC on `start` to 23:00 UTC on `end`."""
    first = utc_midnight(day=start)
    count = (end - start).days * 24 + 24
    return [first + timedelta(hours=index) for index in range(count)]


def summarise_gaps(*, expected: Sequence[datetime], actual: pl.Series) -> dict[str, Any]:
    """Compare the timestamps a table holds against the ones it should hold.

    Args:
        expected: Every timestamp the table should hold, in UTC.
        actual: The table's UTC timestamp column. Duplicates count once.

    Returns:
        The expected count, the distinct actual count, the number of missing and unexpected
        timestamps, and the first `MAX_LISTED_GAPS` missing timestamps as ISO strings.
    """
    present = set(actual.drop_nulls().to_list())
    wanted = set(expected)
    missing = sorted(wanted - present)
    return {
        "expected_rows": len(wanted),
        "distinct_rows": len(present),
        "missing_count": len(missing),
        "unexpected_count": len(present - wanted),
        "missing_first": [stamp.isoformat() for stamp in missing[:MAX_LISTED_GAPS]],
    }


class IncompleteChunkError(Exception):
    """Raised by a fetcher when the source returned fewer rows than the chunk must hold.

    Elexon answers HTTP 200 with an empty `data` list for a date it has not published yet, so a
    short chunk means the source is not ready, and it must not be cached as done.
    """


def require_complete(*, frame: pl.DataFrame, expected_rows: int, key: str) -> pl.DataFrame:
    """Return `frame`, or raise `IncompleteChunkError` if it has fewer than `expected_rows` rows."""
    if frame.height < expected_rows:
        raise IncompleteChunkError(f"chunk {key} has {frame.height} rows, expected {expected_rows}")
    return frame


def period_time_mismatches(*, frame: pl.DataFrame) -> int:
    """Count settlement-date and period pairs whose `time` differs from the computed UTC start."""
    pairs = frame.select("settlement_date", "settlement_period", "time").unique()
    return sum(
        1
        for settlement_date, period, stamp in pairs.iter_rows()
        if stamp != period_start_utc(settlement_date=settlement_date, settlement_period=period)
    )


def get_response(*, url: str, params: QueryParams | None = None) -> httpx.Response:
    """GET `url`, retrying a transient failure with exponential backoff, and raise on the last.

    HTTP 429, every 5xx status, and network errors are retried. Any other 4xx status is final
    because a repeat request cannot succeed.
    """
    for attempt in range(HTTP_RETRIES):
        try:
            response = httpx.get(url, params=params, timeout=HTTP_TIMEOUT_S, follow_redirects=True)
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            final = 400 <= status < 500 and status != TOO_MANY_REQUESTS
            if final or attempt == HTTP_RETRIES - 1:
                raise
            time.sleep(BACKOFF_BASE_S * 2**attempt)
        except httpx.TransportError:
            if attempt == HTTP_RETRIES - 1:
                raise
            time.sleep(BACKOFF_BASE_S * 2**attempt)
        else:
            return response
    raise AssertionError("unreachable")


def get_json(*, url: str, params: QueryParams | None = None) -> Any:
    """GET `url` and return the decoded JSON body.

    Args:
        url: The URL, without a query string.
        params: The query parameters, as pairs so that a name can repeat.

    Returns:
        The decoded JSON body.

    Raises:
        RuntimeError: If the body is not JSON. The message names the status and the first bytes of
            the body, never the query parameters.
    """
    response = get_response(url=url, params=params)
    try:
        return response.json()
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"{url} returned a non-JSON body (HTTP {response.status_code}): {response.text[:200]!r}"
        ) from error


def write_parquet_atomic(*, frame: pl.DataFrame, path: Path) -> None:
    """Write `frame` to `path` through a `.partial` file, so a crash leaves no partial chunk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".parquet.partial")
    frame.write_parquet(partial, compression="zstd")
    partial.rename(path)


def fetch_missing_chunks(
    *,
    cache_dir: Path,
    keys: Sequence[str],
    fetch: Callable[[str], pl.DataFrame],
    label: str,
    threads: int = FETCH_THREADS,
) -> dict[str, list[str]]:
    """Fetch every chunk that is not already cached, writing each to disk as soon as it arrives.

    A chunk is one file, `cache_dir/<key>.parquet`. A file already on disk is skipped, so a re-run
    after a crash fetches only what is missing. A chunk that answers HTTP 404, or that its fetcher
    rejects with `IncompleteChunkError`, is recorded as not published and left uncached, but only
    if every later key is also not published: a gap in the middle of the window raises. Any other
    failure aborts the run.

    Args:
        cache_dir: The folder of cached chunks.
        keys: One key per chunk, used as the file stem.
        fetch: Returns the chunk for a key. An empty frame with the right schema is a valid chunk.
        label: Names the source in progress messages.
        threads: How many chunks are in flight at once.

    Returns:
        The keys that were `fetched`, were already `cached`, and were `not_published`.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [key for key in keys if not (cache_dir / f"{key}.parquet").exists()]
    outcome: dict[str, list[str]] = {
        "fetched": [],
        "cached": [key for key in keys if key not in todo],
        "not_published": [],
    }
    print(f"{label}: {len(outcome['cached'])} chunks cached, {len(todo)} to fetch")

    def one(key: str) -> tuple[str, bool]:
        try:
            frame = fetch(key)
        except httpx.HTTPStatusError as error:
            if error.response.status_code == NOT_FOUND:
                return key, False
            raise
        except IncompleteChunkError as error:
            print(f"{label}: {error}")
            return key, False
        write_parquet_atomic(frame=frame, path=cache_dir / f"{key}.parquet")
        return key, True

    started = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=threads)
    try:
        for done, (key, published) in enumerate(pool.map(one, todo), start=1):
            outcome["fetched" if published else "not_published"].append(key)
            if not published:
                print(f"{label} {key}: not published yet (404), skipping")
            if done % 25 == 0 or done == len(todo):
                print(f"{label}: {done}/{len(todo)} chunks, {time.monotonic() - started:.0f} s")
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()
    skipped = set(outcome["not_published"])
    if skipped:
        first = min(keys.index(key) for key in skipped)
        later = [key for key in keys[first:] if key not in skipped]
        if later:
            raise RuntimeError(
                f"{label}: chunks {sorted(skipped)} are not published but later chunks {later[:3]} "
                "exist, so the gap is not at the end of the window"
            )
    return outcome


def read_chunks(*, cache_dir: Path, keys: Sequence[str], schema: dict[str, Any]) -> pl.DataFrame:
    """Concatenate the cached chunks for `keys`, in key order, skipping keys with no file.

    Args:
        cache_dir: The folder of cached chunks.
        keys: The chunks to read.
        schema: The schema of an empty result, for when no chunk exists.

    Returns:
        The concatenated rows.
    """
    paths = [cache_dir / f"{key}.parquet" for key in sorted(keys)]
    existing = [path for path in paths if path.exists()]
    if not existing:
        return pl.DataFrame(schema=schema)
    return pl.concat([pl.read_parquet(path) for path in existing])


def write_lineage(*, product_dir: Path, note: dict[str, Any]) -> None:
    """Write `<product_dir>/lineage.json` with the retrieval time added."""
    product_dir.mkdir(parents=True, exist_ok=True)
    record = {"retrieved_at_utc": datetime.now(UTC).isoformat()} | note
    (product_dir / "lineage.json").write_text(json.dumps(record, indent=2, default=str))


def write_readme(
    *,
    product_dir: Path,
    title: str,
    source_page: str,
    script_path: str,
    attribution: str | None,
    cache_hint: str,
    licence: str,
    timestamp_convention: str,
    columns: dict[str, str],
    row_summary: str,
    gotchas: list[str],
) -> None:
    """Write `<product_dir>/README.md`, a companion to `lineage.json` for humans.

    Args:
        product_dir: The source's folder.
        title: The source's name.
        source_page: A web page that describes the source.
        script_path: Repo-relative path of the script that wrote the folder.
        attribution: The attribution line the licence requires, or None if it requires none.
        cache_hint: Where the cached chunks are, relative to `product_dir`, for the re-fetch note.
        licence: The licence, and where it came from.
        timestamp_convention: What each time column means, in plain words.
        columns: Every column of the written parquet mapped to a one-line description with units.
        row_summary: The measured row counts against the expected counts, and the gaps.
        gotchas: Known traps, one bullet each.
    """
    product_dir.mkdir(parents=True, exist_ok=True)
    column_lines = "\n".join(f"- `{name}`: {description}" for name, description in columns.items())
    gotcha_lines = "\n".join(f"- {gotcha}" for gotcha in gotchas) or "- None known."
    attribution_lines = f"\n**Attribution:** {attribution}\n" if attribution else ""
    readme = f"""# {title}

Public data for the battery-versus-solar-PV study
(<https://github.com/openclimatefix/nged-substation-forecast/pull/1094>). Nothing under
`data/` is committed to the repository.

- **Source:** <{source_page}>
- **Re-download with:** `{script_path}`. The script's docstring gives the exact command. Delete a
  file in `{cache_hint}` to fetch that chunk again.
- **Full request details and the list of gaps:** `lineage.json` next to this file.

## Licence
{attribution_lines}
{licence}

## Timestamp convention

{timestamp_convention}

## Columns

{column_lines}

## Row counts and gaps

{row_summary}

## Gotchas

{gotcha_lines}
"""
    (product_dir / "README.md").write_text(readme)
