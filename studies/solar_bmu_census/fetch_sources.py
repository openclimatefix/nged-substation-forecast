"""Download the sources of the solar-BMU census into `data/studies/per_study/solar_bmu_census/`.

A BMU is a Balancing Mechanism Unit, IGCPU is the Installed Generation Capacity per Unit report, and
NESO is the National Energy System Operator. Run:
`uv run python studies/solar_bmu_census/fetch_sources.py`.

Every request is cached under `inputs/` with its retrieval time, so a rerun on the same window sends
no HTTP request. A crash loses at most the one request in flight. Half-hourly output (Elexon dataset
B1610) is saved as one parquet file per BMU, written atomically. The study's window is the 12
complete calendar months that end at least `LAG_MARGIN_DAYS` (14 days) before the run, because
B1610 lags real time by about 7 days. The window and the run date are written to `lineage.json`, and
every later script reads them from there. The Maximum Export Limit is fetched by `collate.py`
(through `fetch_mels`), for the solar BMUs only.
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
from studies.sources import SOLAR_BMU_CENSUS_DIR, SOLAR_BMU_CENSUS_INPUTS_DIR

ELEXON_API: Final[str] = "https://data.elexon.co.uk/bmrs/api/v1"
NESO_TEC_PAGE: Final[str] = (
    "https://www.neso.energy/data-portal/transmission-entry-capacity-tec-register"
)
REPD_PAGE: Final[str] = (
    "https://www.gov.uk/government/publications/renewable-energy-planning-database-monthly-extract"
)
NESO_DNO_AREAS_PAGE: Final[str] = (
    "https://neso.energy/data-portal/gis-boundaries-gb-dno-license-areas"
)
NESO_DNO_AREAS_GEOJSON: Final[str] = (
    "https://api.neso.energy/dataset/0e377f16-95e9-4c15-a1fc-49e06a39cfa0/resource/"
    "1c6a7dc0-1b6c-443a-bc67-5f7125649434/download/gb-dno-license-areas-20240503-as-geojson.geojson"
)
"""NESO's map of the 14 distribution network operator licence areas. Each feature carries the GSP
group identifier (`Name`), the operator (`DNO`), and the area name (`Area`), and the coordinates are
Ordnance Survey National Grid metres (EPSG:27700)."""
LCCC_API: Final[str] = "https://dp.lowcarboncontracts.uk/api/3/action/datastore_search"
LCCC_MAPPING_PAGE: Final[str] = "https://dp.lowcarboncontracts.uk/dataset/cfd-to-bm-unit-mapping"
LCCC_PORTFOLIO_PAGE: Final[str] = (
    "https://dp.lowcarboncontracts.uk/dataset/cfd-contract-portfolio-status"
)
LCCC_MAPPING_RESOURCE: Final[str] = "c16f141d-2db9-4160-ade1-d0d19d224dc9"
"""The Low Carbon Contracts Company's datastore resource for the CfD-to-BM-Unit mapping, which has
one row for each Contract for Difference (CfD) identifier and the BM Unit that carries it."""
LCCC_PORTFOLIO_RESOURCE: Final[str] = "fdaf09d2-8cff-4799-a5b0-1c59444e492b"
"""The datastore resource for the CfD Contract Portfolio Status, which has one row for each CfD
unit with its name, technology, connection type, status, and maximum contract capacity."""
LCCC_PAGE_LIMIT: Final[int] = 5000
"""The rows requested from a datastore resource. A response with fewer rows than the resource's
total raises, so a resource that outgrows the limit fails loudly."""
STUDY_DIR: Final[Path] = SOLAR_BMU_CENSUS_DIR
INPUTS_DIR: Final[Path] = SOLAR_BMU_CENSUS_INPUTS_DIR
RAW_DIR: Final[Path] = INPUTS_DIR / "raw"
OUTPUT_DIR: Final[Path] = INPUTS_DIR / "b1610"
"""One parquet file per BMU: its half-hourly settled output over the window."""
IGCPU_MAX_WINDOW_DAYS: Final[int] = 700
"""The IGCPU endpoint rejects a publish-time window longer than 731 days."""
IGCPU_WINDOWS: Final[int] = 4
"""Four 700-day windows reach back to 2019. IGCPU holds no publication before 2023, so four windows
fetch every publication."""
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

    The cache file holds the retrieval time, the requested URL, the final URL after redirects, and
    the raw body, so a reader can tell when the data was fetched.

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
    return pl.read_csv(body.lstrip(BYTE_ORDER_MARK).encode("utf-8"), infer_schema_length=0)


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


def fetch_dno_areas() -> dict[str, Any]:
    """Return NESO's licence-area map of the 14 distribution network operators, as GeoJSON."""
    return json.loads(cached_text(name="dno_licence_areas", url=NESO_DNO_AREAS_GEOJSON))


def _lccc_records(*, name: str, resource_id: str) -> list[dict[str, Any]]:
    """Fetch every row of one Low Carbon Contracts Company datastore resource."""
    body = cached_text(
        name=name,
        url=LCCC_API,
        params={"resource_id": resource_id, "limit": str(LCCC_PAGE_LIMIT)},
    )
    result = json.loads(body)["result"]
    records: list[dict[str, Any]] = result["records"]
    if len(records) != result["total"]:
        raise RuntimeError(f"{name}: got {len(records)} of {result['total']} rows")
    return records


def fetch_lccc() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch the Low Carbon Contracts Company's CfD-to-BM-Unit mapping and CfD portfolio.

    Returns:
        The mapping rows (`CFD_Id`, `BMU_Id`, `Effective_From`, `Effective_date_to`) and the
        portfolio rows (`CFD_ID`, `Name_of_CFD_Unit`, `Technology_Type`,
        `Transmission_or_Distribution_connection`, `Status`, `Maximum_Contract_Capacity_MW`, and
        others).
    """
    return (
        _lccc_records(name="lccc_cfd_bmu_mapping", resource_id=LCCC_MAPPING_RESOURCE),
        _lccc_records(name="lccc_cfd_portfolio", resource_id=LCCC_PORTFOLIO_RESOURCE),
    )


def single_site_cfd_bmus(
    *, mapping: list[dict[str, Any]], portfolio: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Find the `C__` BM Units that carry exactly one named CfD unit.

    A `C__` BM Unit is an Additional Supplier BM Unit that Elexon registers solely to allocate
    Contract for Difference assets. A BM Unit with one current CfD identifier, whose portfolio row
    names a unit, is one named generating site, so the census counts it as single-site. A BM Unit
    that carries several current CfD identifiers pools several sites.

    Args:
        mapping: The rows of `fetch_lccc`'s mapping. A row with a non-empty `Effective_date_to`
            has ended and is ignored.
        portfolio: The rows of `fetch_lccc`'s portfolio.

    Returns:
        Maps each such BM Unit identifier to its `cfd_id`, `name`, `technology`, `connection`,
        `status`, and `capacity_mw` (the maximum contract capacity, None when the portfolio leaves
        it blank).
    """
    by_cfd = {str(row["CFD_ID"]): row for row in portfolio}
    current: dict[str, list[str]] = {}
    for row in mapping:
        bmu_id = str(row["BMU_Id"])
        if bmu_id.startswith("C__") and not row["Effective_date_to"]:
            current.setdefault(bmu_id, []).append(str(row["CFD_Id"]))
    units: dict[str, dict[str, Any]] = {}
    for bmu_id, cfd_ids in current.items():
        unit = by_cfd.get(cfd_ids[0]) if len(cfd_ids) == 1 else None
        if unit is None or not str(unit["Name_of_CFD_Unit"]).strip():
            continue
        capacity = str(unit["Maximum_Contract_Capacity_MW"]).strip()
        units[bmu_id] = {
            "cfd_id": cfd_ids[0],
            "name": str(unit["Name_of_CFD_Unit"]).strip(),
            "technology": str(unit["Technology_Type"]),
            "connection": str(unit["Transmission_or_Distribution_connection"]),
            "status": str(unit["Status"]),
            "capacity_mw": float(capacity) if capacity else None,
        }
    return units


def parse_b1610(*, rows: list[dict[str, Any]], window: Window) -> pl.DataFrame:
    """Turn B1610 stream rows into one row per half-hour, in the window.

    `halfHourEndTime` carries no zone marker, and Elexon states it in UTC, so it is parsed as UTC. A
    request's window includes both ends, so rows are cut to `(start, end]`. The stream returns one
    row per half-hour, and each row already holds the latest settlement run. Rows are deduplicated
    on the half-hour end time as a guard only.

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

    Returns 0 for a file already on disk.
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
    request and `datasets/MELS` is limited to 1 hour per request.
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

    Every later script reads the run date and window from `lineage.json`, so a script run on a later
    day or in a later month still reads the files the fetch wrote.

    Returns:
        The run date, and the half-open UTC window of the B1610 download.

    Raises:
        FileNotFoundError: If the fetch has not been run.
    """
    lineage = json.loads((INPUTS_DIR / "lineage.json").read_text(encoding="utf-8"))
    start, end = (datetime.fromisoformat(stamp) for stamp in lineage["window_utc"])
    return date.fromisoformat(lineage["run_date"]), Window(start=start, end=end)


def write_provenance(*, window: Window, bmu_count: int, today: date) -> None:
    """Write `lineage.json` and a `README.md` for the inputs, computed from what is on disk."""
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
            "dno_licence_areas": NESO_DNO_AREAS_GEOJSON,
            "lccc_cfd_bmu_mapping": LCCC_MAPPING_PAGE,
            "lccc_cfd_portfolio": LCCC_PORTFOLIO_PAGE,
            "cams_irradiance_for_the_solar_estimate": (
                "solar_estimate.CAMS_PUBLIC_POINTS_PATH, read by report.py and not fetched here: "
                "hourly global horizontal irradiance of the CAMS radiation service at the "
                "single-site solar BMUs' positions and at 18 grid points across Great Britain"
            ),
        },
        "run_date": today.isoformat(),
        "window_utc": [window.start.isoformat(), window.end.isoformat()],
        "bmus_requested": bmu_count,
        "b1610_files": len(files),
        "written_at_utc": datetime.now(UTC).isoformat(),
        "per_request_retrieval_times": "each file in raw/ carries `retrieved_at_utc`",
    }
    (INPUTS_DIR / "lineage.json").write_text(json.dumps(lineage, indent=2), encoding="utf-8")
    columns = "\n".join(f"- `{name}`: `{dtype}`" for name, dtype in sample.schema.items())
    (INPUTS_DIR / "README.md").write_text(
        f"""# Solar-BMU census inputs

Written by `studies/solar_bmu_census/fetch_sources.py`. Window: {window.start:%Y-%m-%d} to
{window.end:%Y-%m-%d} (UTC, half-open).

- `raw/`: each small source (BMU reference data, IGCPU, TEC register, REPD, NESO's licence-area map
  of the distribution network operators, the Low Carbon Contracts Company's CfD-to-BM-Unit mapping
  and CfD portfolio, and the Maximum Export Limit of the solar BMUs, which `collate.py` fetches) as
  JSON with `retrieved_at_utc`, `requested_url`, `final_url`, and `body`.
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
    fetch_dno_areas()
    fetch_lccc()
    bmu_ids = b1610_bmu_ids(reference=reference)
    fetch_b1610(bmu_ids=bmu_ids, window=window)
    write_provenance(window=window, bmu_count=len(bmu_ids), today=args.today)
    print(f"Fetched {len(bmu_ids)} BMUs for {window.start:%Y-%m-%d} to {window.end:%Y-%m-%d}")


if __name__ == "__main__":
    main()
