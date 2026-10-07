"""Download the sources of the solar-BMU census into `data/studies/per_study/solar_bmu_census/`.

Run: `uv run python studies/solar_bmu_census/fetch_sources.py`. Every request is cached under
`downloads/` with its retrieval time, so a rerun on the same window makes no network call, and a
crash costs one request. Half-hourly output (Elexon dataset B1610) is saved one parquet file per
BMU, written atomically. The study's window is the 12 complete calendar months that end at least
two weeks before the run, because B1610 lags real time by about a week. The window and the run date
are written to `lineage.json`, and every later script reads them from there. The Maximum Export
Limit is fetched by `collate.py` (through `fetch_mels`), for the solar BMUs only.
"""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import httpx
import polars as pl
from studies.sources import PER_STUDY_DIR

ELEXON_API: Final[str] = "https://data.elexon.co.uk/bmrs/api/v1"
NESO_TEC_PAGE: Final[str] = (
    "https://www.neso.energy/data-portal/transmission-entry-capacity-tec-register"
)
REPD_PAGE: Final[str] = (
    "https://www.gov.uk/government/publications/renewable-energy-planning-database-monthly-extract"
)
STUDY_DIR: Final[Path] = PER_STUDY_DIR / "solar_bmu_census"
DOWNLOADS_DIR: Final[Path] = STUDY_DIR / "downloads"
RAW_DIR: Final[Path] = DOWNLOADS_DIR / "raw"
OUTPUT_DIR: Final[Path] = DOWNLOADS_DIR / "b1610"
"""One parquet file per BMU: its half-hourly settled output over the window."""
IGCPU_MAX_WINDOW_DAYS: Final[int] = 700
"""The IGCPU endpoint rejects a publish-time window longer than 731 days."""
IGCPU_WINDOWS: Final[int] = 4
"""Four 700-day windows reach back to 2019, and the register holds nothing before 2023."""
MEL_WINDOW_DAYS: Final[int] = 30
LAG_MARGIN_DAYS: Final[int] = 14
FETCH_THREADS: Final[int] = 4
HTTP_TIMEOUT_S: Final[float] = 300.0
HTTP_RETRIES: Final[int] = 5
BYTE_ORDER_MARK: Final[str] = chr(0xFEFF)
TOO_MANY_REQUESTS: Final[int] = 429


@dataclass(frozen=True)
class Window:
    """A half-open UTC window `(start, end]` of half-hour end times."""

    start: datetime
    end: datetime

    @property
    def label(self) -> str:
        """The window as `YYYYMMDD_YYYYMMDD`, for file names."""
        return f"{self.start:%Y%m%d}_{self.end:%Y%m%d}"

    @property
    def expected_half_hours(self) -> int:
        """The number of half-hours in the window."""
        return int((self.end - self.start) / timedelta(minutes=30))


def study_window(*, today: date) -> Window:
    """Return the 12 complete calendar months ending at least `LAG_MARGIN_DAYS` before `today`.

    Args:
        today: The date of the run.

    Returns:
        The window, from midnight UTC on the first day of its first month to midnight UTC on the
        first day after its last month.
    """
    cutoff = today - timedelta(days=LAG_MARGIN_DAYS)
    end = datetime(year=cutoff.year, month=cutoff.month, day=1, tzinfo=UTC)
    start = end.replace(year=end.year - 1)
    return Window(start=start, end=end)


def _get(*, url: str, params: dict[str, str] | None = None) -> httpx.Response:
    """GET `url`, retrying a transient failure, and raise if the last attempt fails.

    A client error other than 429 is final, because a repeat request cannot succeed.
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
            time.sleep(5 * 2**attempt)
        except httpx.TransportError:
            if attempt == HTTP_RETRIES - 1:
                raise
            time.sleep(5 * 2**attempt)
        else:
            return response
    raise AssertionError("unreachable")


def cached_text(
    *, name: str, url: str, params: dict[str, str] | None = None, encoding: str | None = None
) -> str:
    """Return the body of `url`, from `RAW_DIR/<name>.json` if it exists, else by fetching it.

    The cache file holds the retrieval time, the URL, and the raw body, so a reader can tell when
    the data was fetched.

    Args:
        name: The cache file's stem. It must change whenever the requested window changes.
        url: The URL to fetch.
        params: Query parameters.
        encoding: The text encoding of the body, for a file whose server does not state one.

    Returns:
        The response body as text.
    """
    path = RAW_DIR / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["body"]
    response = _get(url=url, params=params)
    if encoding is not None:
        response.encoding = encoding
    record = {
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "requested_url": url,
        "final_url": str(response.request.url),
        "body": response.text,
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".json.partial")
    partial.write_text(json.dumps(record), encoding="utf-8")
    partial.rename(path)
    return record["body"]


def fetch_bmu_reference() -> list[dict[str, Any]]:
    """Fetch the Elexon BMU reference data, which lists every registered BMU."""
    body = cached_text(name="bmunits_all", url=f"{ELEXON_API}/reference/bmunits/all")
    return json.loads(body)


def fetch_igcpu(*, today: date) -> list[dict[str, Any]]:
    """Fetch every IGCPU (report B1420) publication, in windows the endpoint accepts."""
    rows: list[dict[str, Any]] = []
    window_end = datetime(year=today.year, month=today.month, day=today.day, tzinfo=UTC)
    window_end += timedelta(days=1)
    for _ in range(IGCPU_WINDOWS):
        window_start = window_end - timedelta(days=IGCPU_MAX_WINDOW_DAYS)
        body = cached_text(
            name=f"igcpu_{today:%Y%m%d}_{window_start:%Y%m%d}",
            url=f"{ELEXON_API}/datasets/IGCPU",
            params={
                "publishDateTimeFrom": f"{window_start:%Y-%m-%dT%H:%MZ}",
                "publishDateTimeTo": f"{window_end:%Y-%m-%dT%H:%MZ}",
                "format": "json",
            },
        )
        rows += json.loads(body)["data"]
        window_end = window_start
    return rows


def _linked_csv(*, page_name: str, page_url: str, pattern: str) -> str:
    """Return the CSV link the publisher's page carries, which changes with each release."""
    page = cached_text(name=page_name, url=page_url)
    links = re.findall(pattern=pattern, string=page)
    if not links:
        raise RuntimeError(f"No CSV link matching {pattern!r} found on {page_url}")
    return links[0]


def fetch_tec() -> pl.DataFrame:
    """Download the current NESO Transmission Entry Capacity register as all-string columns."""
    link = _linked_csv(
        page_name="tec_page",
        page_url=NESO_TEC_PAGE,
        pattern=r"https://api\.neso\.energy/[^\"' <]*/tec-register[^\"' <]*\.csv",
    )
    body = cached_text(name="tec_register", url=link)
    return pl.read_csv(body.lstrip("﻿").encode("utf-8"), infer_schema_length=0)


def fetch_repd() -> pl.DataFrame:
    """Download the Renewable Energy Planning Database CSV as all-string columns."""
    link = _linked_csv(
        page_name="repd_page",
        page_url=REPD_PAGE,
        pattern=(
            r"https://assets\.publishing\.service\.gov\.uk/[^\"' <]*/REPD_Publication"
            r"[^\"' <]*\.csv"
        ),
    )
    body = cached_text(name="repd_register", url=link, encoding="cp1252")
    return pl.read_csv(body.encode("utf-8"), infer_schema_length=0)


def parse_b1610(*, rows: list[dict[str, Any]], window: Window) -> pl.DataFrame:
    """Turn B1610 stream rows into one row per half-hour, in the window.

    `halfHourEndTime` carries no zone marker, and Elexon states it in UTC, so it is parsed as UTC. A
    request's window includes both ends, so rows are cut to `(start, end]`. The stream returns one
    row per half-hour, already the latest settlement run, and rows are deduplicated on the half-hour
    end time only as a guard.

    Args:
        rows: The decoded JSON rows of one BMU.
        window: The study window.

    Returns:
        Columns `half_hour_end_time` (UTC) and `output_mwh` (energy in the half-hour), sorted by
        time.
    """
    frame = pl.DataFrame(
        {
            "half_hour_end_time": [row["halfHourEndTime"] for row in rows],
            "output_mwh": [row["quantity"] for row in rows],
        },
        schema={"half_hour_end_time": pl.String, "output_mwh": pl.Float64},
    )
    return (
        frame.with_columns(
            half_hour_end_time=pl.col("half_hour_end_time").str.to_datetime(
                format=None, time_zone="UTC", time_unit="us"
            )
        )
        .filter(
            pl.col("half_hour_end_time") > window.start, pl.col("half_hour_end_time") <= window.end
        )
        .unique(subset="half_hour_end_time", keep="last", maintain_order=True)
        .sort("half_hour_end_time")
    )


def _fetch_one_bmu(*, bmu_id: str, window: Window) -> int:
    """Fetch one BMU's B1610 output into `OUTPUT_DIR`, unless cached, and return its row count.

    Returns 0 for a file already on disk, so a resumed run counts only the rows it fetched.
    """
    path = OUTPUT_DIR / f"{bmu_id}_{window.label}.parquet"
    if path.exists():
        return 0
    response = _get(
        url=f"{ELEXON_API}/datasets/B1610/stream",
        params={
            "from": f"{window.start:%Y-%m-%dT%H:%MZ}",
            "to": f"{window.end:%Y-%m-%dT%H:%MZ}",
            "bmUnit": bmu_id,
        },
    )
    frame = parse_b1610(rows=response.json(), window=window)
    partial = path.with_suffix(".parquet.partial")
    frame.write_parquet(partial)
    partial.rename(path)
    return frame.height


def b1610_bmu_ids(*, reference: list[dict[str, Any]]) -> list[str]:
    """Return each BMU identifier once, leaving out interconnectors, for the B1610 fetch.

    The reference data lists a few BMUs twice, and two threads must not write one file.
    """
    return sorted(
        {
            str(row["elexonBmUnit"])
            for row in reference
            if row["elexonBmUnit"] and row["interconnectorId"] is None
        }
    )


def fetch_b1610(*, bmu_ids: list[str], window: Window) -> None:
    """Fetch the half-hourly output of every BMU in `bmu_ids`, resuming from the files on disk."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=FETCH_THREADS)
    try:
        results = pool.map(lambda bmu_id: _fetch_one_bmu(bmu_id=bmu_id, window=window), bmu_ids)
        for done, _ in enumerate(results, start=1):
            if done % 200 == 0:
                print(f"B1610: {done}/{len(bmu_ids)} BMUs, {time.monotonic() - started:.0f} s")
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()


def fetch_mels(*, bmu_ids: list[str], today: date) -> dict[str, float | None]:
    """Return the largest Maximum Export Limit in the last 30 days for each BMU, in MW.

    Uses `datasets/MELS/stream`, because `balancing/physical/all` takes one settlement period per
    request and `datasets/MELS` is limited to one hour.
    """
    end = datetime(year=today.year, month=today.month, day=today.day, tzinfo=UTC)
    start = end - timedelta(days=MEL_WINDOW_DAYS)
    largest: dict[str, float | None] = {}
    for bmu_id in bmu_ids:
        body = cached_text(
            name=f"mels_{bmu_id}_{today:%Y%m%d}",
            url=f"{ELEXON_API}/datasets/MELS/stream",
            params={
                "from": f"{start:%Y-%m-%dT%H:%MZ}",
                "to": f"{end:%Y-%m-%dT%H:%MZ}",
                "bmUnit": bmu_id,
            },
        )
        levels = [
            max(row["levelFrom"], row["levelTo"])
            for row in json.loads(body)
            if row["levelFrom"] is not None and row["levelTo"] is not None
        ]
        largest[bmu_id] = max(levels) if levels else None
    return largest


def recorded_run() -> tuple[date, Window]:
    """Return the run date and window that `fetch_sources.py` recorded in `lineage.json`.

    Every later script reads them here, so a script run on a later day or in a later month still
    reads the files the fetch wrote.

    Raises:
        FileNotFoundError: If the fetch has not been run.
    """
    lineage = json.loads((DOWNLOADS_DIR / "lineage.json").read_text(encoding="utf-8"))
    start, end = (datetime.fromisoformat(stamp) for stamp in lineage["window_utc"])
    return date.fromisoformat(lineage["run_date"]), Window(start=start, end=end)


def write_provenance(*, window: Window, bmu_count: int, today: date) -> None:
    """Write `lineage.json` and a `README.md` for the downloads, computed from what is on disk."""
    files = sorted(OUTPUT_DIR.glob(f"*_{window.label}.parquet"))
    sample = pl.read_parquet(files[0]) if files else pl.DataFrame()
    lineage = {
        "sources": {
            "bmu_reference": f"{ELEXON_API}/reference/bmunits/all",
            "igcpu": f"{ELEXON_API}/datasets/IGCPU",
            "b1610": f"{ELEXON_API}/datasets/B1610/stream",
            "mels": f"{ELEXON_API}/datasets/MELS/stream",
            "tec_register": NESO_TEC_PAGE,
            "repd": REPD_PAGE,
        },
        "run_date": today.isoformat(),
        "window_utc": [window.start.isoformat(), window.end.isoformat()],
        "bmus_requested": bmu_count,
        "b1610_files": len(files),
        "written_at_utc": datetime.now(UTC).isoformat(),
        "per_request_retrieval_times": "each file in raw/ carries `retrieved_at_utc`",
    }
    (DOWNLOADS_DIR / "lineage.json").write_text(json.dumps(lineage, indent=2), encoding="utf-8")
    columns = "\n".join(f"- `{name}`: `{dtype}`" for name, dtype in sample.schema.items())
    (DOWNLOADS_DIR / "README.md").write_text(
        f"""# Solar-BMU census downloads

Written by `studies/solar_bmu_census/fetch_sources.py`. Window: {window.start:%Y-%m-%d} to
{window.end:%Y-%m-%d} (UTC, half-open).

- `raw/`: each small source (BMU reference data, IGCPU, TEC register, REPD, and the Maximum Export
  Limit of the solar BMUs, which `collate.py` fetches) as JSON with `retrieved_at_utc`,
  `requested_url`, `final_url`, and `body`.
- `b1610/<BMU>_<window>.parquet`: one file per BMU, half-hourly settled output (Elexon dataset
  B1610), {len(files)} files. Columns:

{columns}

`output_mwh` is energy in the half-hour ending at `half_hour_end_time`, so MW is twice the value.
A small negative value at night is the unit's own import. A BMU has no row for a half-hour in
which B1610 published nothing, including every half-hour before the unit began generating. The
latest settlement run that Elexon holds is the value kept for each half-hour.
""",
        encoding="utf-8",
    )


def main() -> None:
    """Fetch every source for the census and write the provenance files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--today", type=date.fromisoformat, default=datetime.now(UTC).date())
    args = parser.parse_args()
    window = study_window(today=args.today)
    reference = fetch_bmu_reference()
    fetch_igcpu(today=args.today)
    fetch_tec()
    fetch_repd()
    bmu_ids = b1610_bmu_ids(reference=reference)
    fetch_b1610(bmu_ids=bmu_ids, window=window)
    write_provenance(window=window, bmu_count=len(bmu_ids), today=args.today)
    print(f"Fetched {len(bmu_ids)} BMUs for {window.start:%Y-%m-%d} to {window.end:%Y-%m-%d}")


if __name__ == "__main__":
    main()
