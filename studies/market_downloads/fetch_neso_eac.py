"""Download NESO's Enduring Auction Capability (EAC) results for 2025-09-01 to 2026-09-30.

Written for the battery-versus-solar-PV study. The EAC is the National Energy System
Operator's (NESO) day-ahead auction for frequency response and reserve. Run all three sources, or
name some:

    uv run python studies/market_downloads/fetch_neso_eac.py
    uv run python studies/market_downloads/fetch_neso_eac.py --sources neso_eac_response_reserve

- `neso_eac_response_reserve`: the cleared volume and price of every auction unit, for Dynamic
  Containment (`DCH`, `DCL`), Dynamic Moderation (`DMH`, `DML`), Dynamic Regulation (`DRH`, `DRL`),
  Quick Reserve (`PQR`, `NQR`), and Slow Reserve (`PSR`, `NSR`). One request for each UTC delivery
  day and resource.
- `neso_eac_balancing_reserve`: the same columns for Balancing Reserve (`PBR`, `NBR`). NESO
  published Balancing Reserve in its own resource until 2025-10-29 and in the response-and-reserve
  resources afterwards, so this source reads from both.
- `neso_eac_unit_to_bmu`: a table that links each auction unit in the two sources above to
  candidate Balancing Mechanism Units (BMUs) in Elexon's register, and a table that says which of
  the study's BMUs have an auction unit. The mapping reads the two parquet files above, so the
  script runs it last.

Each result source gets a folder under `data/studies/downloads/market/` holding one tidy parquet; a
`README.md` and a `lineage.json`, both written from the values the script measured; and a
`_day_cache/` of per-chunk parquet files. Each chunk is written the moment it arrives, so a crash
loses at most one chunk and a re-run fetches only the chunks that are missing. Requests are
keyless, use 4 threads, and back off exponentially on HTTP 429 and 5xx.

Pass `--start`, `--end` (both inclusive), and `--output-root` for a small test run. The mapping
reads the BMU register from `--bmu-reference` and the study's BMUs from `--bmu-list`.
"""

import argparse
import difflib
import re
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, date, datetime, time, timedelta
from functools import partial
from pathlib import Path
from typing import Any, Final, NamedTuple

import polars as pl
from fetch_bmu_dispatch import read_bmu_list
from market_common import (
    FETCH_THREADS,
    LONDON,
    WINDOW_END,
    WINDOW_START,
    IncompleteChunkError,
    days_between,
    expected_half_hour_starts,
    fetch_missing_chunks,
    get_json,
    read_chunks,
    summarise_gaps,
    utc_midnight,
    write_lineage,
    write_parquet_atomic,
    write_readme,
)
from studies.sources import MARKET_DOWNLOADS_DIR

SCRIPT_PATH: Final[str] = "studies/market_downloads/fetch_neso_eac.py"
SOURCE_NAMES: Final[tuple[str, ...]] = (
    "neso_eac_response_reserve",
    "neso_eac_balancing_reserve",
    "neso_eac_unit_to_bmu",
)
"""The sources in run order. The mapping runs last because it reads the other two."""

SQL_URL: Final[str] = "https://api.neso.energy/api/3/action/datastore_search_sql"
EAC_PAGE: Final[str] = "https://www.neso.energy/data-portal/eac-auction-results"
EAC_BR_PAGE: Final[str] = "https://www.neso.energy/data-portal/eac-br-auction-results"
NESO_LICENCE: Final[str] = (
    "NESO Open Data Licence, as shown on the dataset's page on NESO's CKAN data portal. The "
    "licence text has not been independently checked."
)
NESO_ATTRIBUTION: Final[str] = "Supplied by the National Energy System Operator (NESO) Open Data"
"""A plain-words credit. The licence page was not read for required wording."""

PAGE_ROWS: Final[int] = 30_000
"""Rows per SQL request. The CKAN SQL endpoint truncates a reply at 32,000 rows."""
BALANCING_RESERVE: Final[str] = "Balancing Reserve"
SPARSE_PRODUCTS: Final[frozenset[str]] = frozenset({"NBR", "NSR"})
"""Negative Balancing Reserve and negative Slow Reserve clear only in a few periods, so a period
with no row is normal and the gap check skips them."""
EFA_PRODUCTS: Final[frozenset[str]] = frozenset({"DCH", "DCL", "DMH", "DML", "DRH", "DRL"})
"""The products delivered in the six 4-hour Electricity Forward Agreement (EFA) blocks."""
EFA_LOCAL_START_HOURS: Final[tuple[int, ...]] = (3, 7, 11, 15, 19, 23)
"""UK clock hours at which an EFA block starts: block 1 runs 23:00 to 03:00."""
UTC_TIME: Final[pl.Datetime] = pl.Datetime(time_unit="us", time_zone="UTC")
NESO_TIME_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%S"


class EacResource(NamedTuple):
    """One CKAN resource of EAC results by unit, and the delivery starts it holds."""

    resource_id: str
    name: str
    first_start: datetime
    """The first `deliveryStart` the resource held when the script was written (UTC)."""
    last_start: datetime | None
    """The last `deliveryStart` the resource held when the script was written, or None for a
    resource NESO keeps appending to."""


BALANCING_RESERVE_RESOURCE: Final[EacResource] = EacResource(
    resource_id="5d8e47be-e262-4398-89b0-6f93f636faf6",
    name="NESO Balancing-Reserve Results By Unit 2023-2024",
    first_start=datetime(2024, 3, 12, 23, 0, tzinfo=UTC),
    last_start=datetime(2025, 10, 29, 22, 30, tzinfo=UTC),
)
RESPONSE_RESERVE_RESOURCES: Final[tuple[EacResource, ...]] = (
    EacResource(
        resource_id="c312358e-87da-480e-88cc-d45c1dce1b41",
        name="NESO Response-Reserve Results By Unit FY2025 (Archive)",
        first_start=datetime(2025, 3, 31, 22, 0, tzinfo=UTC),
        last_start=datetime(2026, 3, 31, 21, 30, tzinfo=UTC),
    ),
    EacResource(
        resource_id="a63ab354-7e68-44c2-ad96-c6f920c30e85",
        name="NESO Response-Reserve Results By Unit",
        first_start=datetime(2026, 3, 31, 22, 0, tzinfo=UTC),
        last_start=None,
    ),
)
"""The two resources that hold the window's response and reserve results, including Balancing
Reserve from 2025-10-29 23:00 UTC. NESO's "Daily Results By Unit" resource (`07f0dd8b-...`) holds
only the last two delivery days, so it cannot supply a historical window."""

SOURCE_RESOURCES: Final[dict[str, tuple[tuple[EacResource, bool], ...]]] = {
    "neso_eac_response_reserve": tuple((r, False) for r in RESPONSE_RESERVE_RESOURCES),
    "neso_eac_balancing_reserve": (
        (BALANCING_RESERVE_RESOURCE, True),
        *((r, True) for r in RESPONSE_RESERVE_RESOURCES),
    ),
}
"""For each result source, its resources, each with whether the query keeps only the Balancing
Reserve rows (True) or drops them (False). The dedicated Balancing Reserve resource holds nothing
else, so the filter is harmless there."""

RESULT_SCHEMA: Final[dict[str, Any]] = {
    "time": UTC_TIME,
    "time_end": UTC_TIME,
    "auction_product": pl.String,
    "service_type": pl.String,
    "auction_unit": pl.String,
    "participant": pl.String,
    "technology_type": pl.String,
    "post_code": pl.String,
    "executed_quantity_mw": pl.Float64,
    "clearing_price_gbp_per_mw_per_h": pl.Float64,
    "unit_result_id": pl.String,
}
RENAMES: Final[dict[str, str]] = {
    "deliveryStart": "time",
    "deliveryEnd": "time_end",
    "auctionProduct": "auction_product",
    "serviceType": "service_type",
    "auctionUnit": "auction_unit",
    "registeredAuctionParticipant": "participant",
    "technologyType": "technology_type",
    "postCode": "post_code",
    "executedQuantity": "executed_quantity_mw",
    "clearingPrice": "clearing_price_gbp_per_mw_per_h",
    "unitResultID": "unit_result_id",
}
SORT_COLUMNS: Final[tuple[str, ...]] = ("time", "auction_product", "auction_unit", "unit_result_id")

MIN_FUZZY_SCORE: Final[float] = 0.85
"""Lowest identifier similarity at which a fuzzy match is reported."""
FUZZY_CANDIDATES: Final[int] = 3
SIBLING_SCORE: Final[float] = 0.7
PREFIX_SCORE: Final[float] = 0.95
BATTERY_STEM_SCORE: Final[float] = 0.8
BMU_PREFIX_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Z0-9]{1,2}_{1,2}")
COMPANY_WORDS: Final[frozenset[str]] = frozenset(
    {"limited", "ltd", "plc", "llp", "gmbh", "uk", "energy", "power", "customers", "trading"}
)
PLAUSIBLE_METHODS: Final[frozenset[str]] = frozenset(
    {"exact_national_grid_id", "exact_elexon_id", "elexon_id_without_prefix"}
)
"""The match methods that identify a single BMU rather than a possibly related one."""

MAPPING_COLUMNS: Final[dict[str, Any]] = {
    "auction_unit": pl.String,
    "candidate_bmu_id": pl.String,
    "match_method": pl.String,
    "match_score": pl.Float64,
    "candidate_national_grid_bmu_id": pl.String,
    "candidate_lead_party": pl.String,
    "auction_participant": pl.String,
    "party_similarity": pl.Float64,
    "technology_type": pl.String,
    "services": pl.String,
    "result_rows": pl.Int64,
    "candidate_is_study_bmu": pl.Boolean,
}


def efa_block_starts(*, start: date, end: date) -> list[datetime]:
    """Return every EFA block start in UTC whose UTC date is in `start..end`.

    Blocks start at 03:00, 07:00, 11:00, 15:00, 19:00, and 23:00 UK clock time.
    """
    first = utc_midnight(day=start)
    after = utc_midnight(day=end + timedelta(days=1))
    starts = [
        datetime.combine(day, time(hour=hour), tzinfo=LONDON).astimezone(UTC)
        for day in days_between(start=start - timedelta(days=1), end=end + timedelta(days=1))
        for hour in EFA_LOCAL_START_HOURS
    ]
    return sorted(stamp for stamp in starts if first <= stamp < after)


def parse_results(*, rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Turn datastore records into the tidy result table.

    The SQL endpoint returns every number as a string, so the two number columns are cast.

    Args:
        rows: The records, as the CKAN SQL endpoint returns them.

    Returns:
        A table with the columns of `RESULT_SCHEMA`, sorted by `SORT_COLUMNS`.
    """
    if not rows:
        return pl.DataFrame(schema=RESULT_SCHEMA)
    frame = (
        pl.DataFrame(rows, infer_schema_length=None)
        .select(*RENAMES)
        .rename(RENAMES)
        .with_columns(
            pl.col("time").str.to_datetime(
                format=NESO_TIME_FORMAT, time_zone="UTC", time_unit="us"
            ),
            pl.col("time_end").str.to_datetime(
                format=NESO_TIME_FORMAT, time_zone="UTC", time_unit="us"
            ),
            pl.col("executed_quantity_mw", "clearing_price_gbp_per_mw_per_h").cast(pl.Float64),
        )
    )
    return frame.select(*RESULT_SCHEMA).sort(*SORT_COLUMNS)


def day_query(*, resource: EacResource, day: date, balancing_reserve_only: bool) -> str:
    """Return the SQL that selects one UTC delivery day from one resource.

    Args:
        resource: The CKAN resource.
        day: The UTC day of `deliveryStart`.
        balancing_reserve_only: Keep only Balancing Reserve rows (True) or drop them (False).

    Returns:
        The SQL, ending before the `LIMIT` clause.
    """
    first = utc_midnight(day=day)
    after = first + timedelta(days=1)
    operator = "=" if balancing_reserve_only else "<>"
    return (
        f'SELECT * FROM "{resource.resource_id}" '
        f"WHERE \"deliveryStart\" >= '{first:%Y-%m-%dT%H:%M:%S}' "
        f"AND \"deliveryStart\" < '{after:%Y-%m-%dT%H:%M:%S}' "
        f"AND \"serviceType\" {operator} '{BALANCING_RESERVE}' "
        'ORDER BY "_id"'
    )


def resource_overlaps_day(*, resource: EacResource, day: date) -> bool:
    """Return whether the resource held any `deliveryStart` on the UTC `day`."""
    first = utc_midnight(day=day)
    after = first + timedelta(days=1)
    ends_before = resource.last_start is not None and resource.last_start < first
    return not (ends_before or resource.first_start >= after)


def fetch_resource_day(
    *, resource: EacResource, day: date, balancing_reserve_only: bool
) -> list[dict[str, Any]]:
    """Fetch every record of one UTC delivery day from one resource, paging through the reply.

    Args:
        resource: The CKAN resource to query.
        day: The UTC day of `deliveryStart`.
        balancing_reserve_only: Keep only Balancing Reserve rows (True) or drop them (False).

    Returns:
        The records, in `_id` order.

    Raises:
        RuntimeError: If the reply reports failure.
    """
    query = day_query(resource=resource, day=day, balancing_reserve_only=balancing_reserve_only)
    records: list[dict[str, Any]] = []
    while True:
        body = get_json(
            url=SQL_URL, params=[("sql", f"{query} LIMIT {PAGE_ROWS} OFFSET {len(records)}")]
        )
        if not body.get("success"):
            raise RuntimeError(f"NESO SQL failed for {resource.resource_id} {day}: {body}")
        page = body["result"]["records"]
        records.extend(page)
        if len(page) < PAGE_ROWS:
            return records


def fetch_result_day(*, source: str, key: str) -> pl.DataFrame:
    """Fetch one UTC delivery day of a result source, from every resource that overlaps it."""
    day = date.fromisoformat(key)
    rows: list[dict[str, Any]] = []
    for resource, balancing_reserve_only in SOURCE_RESOURCES[source]:
        if resource_overlaps_day(resource=resource, day=day):
            rows.extend(
                fetch_resource_day(
                    resource=resource, day=day, balancing_reserve_only=balancing_reserve_only
                )
            )
    if not rows:
        raise IncompleteChunkError(f"{source} {key}: no resource holds rows for this day")
    return parse_results(rows=rows)


def product_gaps(*, frame: pl.DataFrame, start: date, end: date) -> dict[str, dict[str, Any]]:
    """Compare each product's delivery starts with the time grid the product should fill.

    A product is expected on every point of its time grid between its first and last delivery start
    in the table. The time grid is the EFA block starts for the six response products and every
    half-hour for the rest. Sparse products (`SPARSE_PRODUCTS`) are skipped.

    Args:
        frame: The result table.
        start: First day of the window.
        end: Last day of the window, inclusive.

    Returns:
        For each product, `summarise_gaps` over its grid, plus the product's first and last start.
    """
    efa = efa_block_starts(start=start, end=end)
    half_hours = expected_half_hour_starts(start=start, end=end)
    summary: dict[str, dict[str, Any]] = {}
    for product in sorted(frame["auction_product"].unique().to_list()):
        if product in SPARSE_PRODUCTS:
            continue
        times = frame.filter(pl.col("auction_product") == product)["time"]
        starts: list[datetime] = times.to_list()
        low, high = min(starts), max(starts)
        grid = efa if product in EFA_PRODUCTS else half_hours
        expected = [stamp for stamp in grid if low <= stamp <= high]
        summary[product] = summarise_gaps(expected=expected, actual=times) | {
            "first_start": low.isoformat(),
            "last_start": high.isoformat(),
        }
    return summary


def block_lengths(*, frame: pl.DataFrame) -> dict[str, list[str]]:
    """Return the distinct delivery lengths, in hours, seen for each product."""
    lengths = (
        frame.select(
            "auction_product",
            ((pl.col("time_end") - pl.col("time")).dt.total_minutes() / 60).alias("hours"),
        )
        .unique()
        .sort("auction_product", "hours")
    )
    result: dict[str, list[str]] = defaultdict(list)
    for product, hours in lengths.iter_rows():
        result[product].append(f"{hours:g}")
    return dict(result)


def _gap_summary(*, gaps: dict[str, dict[str, Any]]) -> str:
    lines = [
        f"  - `{product}`: {value['distinct_rows']} of {value['expected_rows']} expected delivery "
        f"starts, {value['missing_count']} missing, {value['unexpected_count']} off the grid; "
        f"first start {value['first_start']}, last start {value['last_start']}"
        for product, value in gaps.items()
    ]
    return "\n".join(lines) or "  - none"


def fetch_results(*, source: str, root: Path, start: date, end: date, threads: int) -> None:
    """Download one result source one UTC delivery day at a time and write its folder."""
    output_dir = root / source
    cache_dir = output_dir / "_day_cache"
    keys = [day.isoformat() for day in days_between(start=start, end=end)]
    chunks = fetch_missing_chunks(
        cache_dir=cache_dir,
        keys=keys,
        fetch=partial(_fetch_for, source),
        label=source,
        threads=threads,
    )
    frame = read_chunks(cache_dir=cache_dir, keys=keys, schema=RESULT_SCHEMA)
    duplicates = frame.height - frame["unit_result_id"].n_unique()
    if duplicates:
        raise RuntimeError(f"{source}: {duplicates} duplicate unitResultID values across chunks")
    gaps = product_gaps(frame=frame, start=start, end=end)
    lengths = block_lengths(frame=frame)
    write_parquet_atomic(frame=frame, path=output_dir / f"{source}.parquet")
    resources = [
        {
            "resource_id": resource.resource_id,
            "name": resource.name,
            "first_start": resource.first_start.isoformat(),
            "last_start": resource.last_start.isoformat() if resource.last_start else None,
            "rows_kept": "Balancing Reserve only"
            if only
            else "every service except Balancing Reserve",
        }
        for resource, only in SOURCE_RESOURCES[source]
    ]
    write_lineage(
        product_dir=output_dir,
        note={
            "source_address": SQL_URL,
            "request": (
                f"One SQL query for each UTC delivery day from {start} to {end} and each resource "
                "whose delivery starts overlap that day, paged by 30,000 rows"
            ),
            "window": [start.isoformat(), end.isoformat()],
            "script": SCRIPT_PATH,
            "rows": frame.height,
            "distinct_auction_units": frame["auction_unit"].n_unique(),
            "resources": resources,
            "gaps_by_product": gaps,
            "delivery_hours_by_product": lengths,
            "chunks_not_published": chunks["not_published"],
        },
    )
    write_readme(
        product_dir=output_dir,
        script_path=SCRIPT_PATH,
        cache_hint="_day_cache/",
        **_result_readme(source=source, frame=frame, gaps=gaps, lengths=lengths),
    )
    print(f"{source}: wrote {frame.height} rows for {frame['auction_unit'].n_unique()} units")


def _fetch_for(source: str, key: str) -> pl.DataFrame:
    return fetch_result_day(source=source, key=key)


def _result_readme(
    *,
    source: str,
    frame: pl.DataFrame,
    gaps: dict[str, dict[str, Any]],
    lengths: dict[str, list[str]],
) -> dict[str, Any]:
    """Return the `write_readme` arguments that differ between the two result sources."""
    balancing = source == "neso_eac_balancing_reserve"
    length_text = "; ".join(
        f"`{product}` {', '.join(hours)} h" for product, hours in lengths.items()
    )
    return {
        "title": (
            "NESO Enduring Auction Capability (EAC) Balancing Reserve results by unit"
            if balancing
            else "NESO Enduring Auction Capability (EAC) response and reserve results by unit"
        ),
        "source_page": EAC_BR_PAGE if balancing else EAC_PAGE,
        "attribution": NESO_ATTRIBUTION,
        "licence": NESO_LICENCE,
        "timestamp_convention": (
            "`time` is the UTC start of the delivery period (NESO's `deliveryStart`) and "
            "`time_end` is its end. NESO states that these values are in UTC. The Dynamic "
            "Containment, Moderation, and Regulation products are delivered in 4-hour Electricity "
            "Forward Agreement (EFA) blocks that start at 23:00, 03:00, 07:00, 11:00, 15:00, and "
            "19:00 UK clock time, which is 22:00 UTC and so on in summer. The gap counts below "
            "say how many block starts fall off that time grid. Quick Reserve, Slow Reserve, and "
            f"Balancing Reserve are half-hourly. Delivery lengths seen: {length_text}. The window "
            "is by UTC day of `time`, so a UTC day can hold part of two EFA days."
        ),
        "columns": {
            "time": "Start of the delivery period, UTC",
            "time_end": "End of the delivery period, UTC",
            "auction_product": (
                "NESO product code: DCH/DCL (Dynamic Containment high/low frequency), DMH/DML "
                "(Moderation), DRH/DRL (Regulation), PQR/NQR (positive/negative Quick Reserve), "
                "PSR/NSR (Slow Reserve), PBR/NBR (Balancing Reserve)"
            ),
            "service_type": (
                "NESO service: Response, Quick Reserve, Slow Reserve, or Balancing Reserve"
            ),
            "auction_unit": (
                "NESO's auction unit identifier. It is often a BMU's National Grid identifier but "
                "not always; see `neso_eac_unit_to_bmu/`"
            ),
            "participant": "The registered auction participant (company) that owns the bid",
            "technology_type": "NESO's technology label for the unit, for example Batteries",
            "post_code": "Outward part of the unit's postcode",
            "executed_quantity_mw": "Volume accepted at the clearing price, MW",
            "clearing_price_gbp_per_mw_per_h": "Cleared price, GBP per MW of capacity per hour",
            "unit_result_id": "NESO's unique identifier of the unit result row",
        },
        "row_summary": (
            f"- Rows written: {frame.height}, for {frame['auction_unit'].n_unique()} auction "
            "units.\n- Delivery starts compared with the time grid each product should fill, "
            "between its first and last start in the table (`NBR` and `NSR` are skipped because "
            "they clear in few periods):\n"
            + _gap_summary(gaps=gaps)
            + "\n- `lineage.json` holds the "
            "same counts with the first missing starts."
        ),
        "gotchas": [
            (
                "A row is one unit's accepted volume in one delivery period. A unit that won "
                "nothing has no row, so the table is not a panel with zeros."
            ),
            (
                "The price is per MW of capacity per hour, not per MWh of energy. A row's revenue "
                "is `executed_quantity_mw * clearing_price * hours`, where `hours` is "
                "`time_end - time` in hours. Do not assume 4 hours: the blocks on the eve of a "
                "clock change last 3 or 5 hours (see the delivery lengths above)."
            ),
            (
                "NESO's 'Daily Results By Unit' resource (`07f0dd8b-...`) "
                "holds only the most recent two delivery days. The history comes from the "
                "'Results By Unit' resources listed in `lineage.json`."
            ),
            (
                "Balancing Reserve left its own dataset on 2025-10-29 22:30 UTC and appears in "
                "the response-and-reserve resources from 23:00 UTC that day. "
                + (
                    "This source joins both parts."
                    if balancing
                    else "This source drops those rows; the Balancing Reserve rows are in "
                    "`neso_eac_balancing_reserve/`."
                )
            ),
            (
                "NESO may restate results after publication. The table holds what the datastore "
                "returned on the retrieval date in `lineage.json`."
            ),
        ],
    }


def party_tokens(*, name: str | None) -> set[str]:
    """Return the lower-case words of a company name, without legal-form and generic words."""
    if name is None:
        return set()
    words = re.findall(r"[a-z0-9]+", name.lower())
    return {word for word in words if word not in COMPANY_WORDS}


def party_similarity(*, left: str | None, right: str | None) -> float | None:
    """Return the word overlap of two company names, or None if either has no distinctive word.

    The overlap is the Jaccard similarity of the names' distinctive words (see `party_tokens`).
    """
    first, second = party_tokens(name=left), party_tokens(name=right)
    if not first or not second:
        return None
    return len(first & second) / len(first | second)


def stem(*, identifier: str) -> str:
    """Return the part of a unit or BMU identifier before its last hyphen."""
    return identifier.rsplit("-", maxsplit=1)[0] if "-" in identifier else identifier


class BmuIndex(NamedTuple):
    """Lookups over Elexon's BMU register, keyed by upper-case identifier."""

    by_national_grid_id: dict[str, list[str]]
    by_elexon_id: dict[str, list[str]]
    by_stripped_elexon_id: dict[str, list[str]]
    by_stem: dict[str, list[str]]
    rows: dict[str, dict[str, Any]]
    """The register row for each Elexon BMU identifier."""
    fuzzy_ids: dict[str, list[str]]
    """Every identifier, national grid or stripped Elexon, mapped to the Elexon BMU identifiers."""


def build_bmu_index(*, reference: pl.DataFrame) -> BmuIndex:
    """Index the BMU register for the matching passes.

    Elexon's register has the Elexon identifier (`elexon_bmu_id`, null for transmission BMUs that
    only have a National Grid identifier) and `national_grid_bmu_id`. A BMU with no Elexon
    identifier is keyed by its National Grid identifier.

    Args:
        reference: The `bmu_reference.parquet` table.

    Returns:
        The lookups.
    """
    by_national: dict[str, list[str]] = defaultdict(list)
    by_elexon: dict[str, list[str]] = defaultdict(list)
    by_stripped: dict[str, list[str]] = defaultdict(list)
    by_stem: dict[str, list[str]] = defaultdict(list)
    fuzzy: dict[str, list[str]] = defaultdict(list)
    rows: dict[str, dict[str, Any]] = {}
    for row in reference.iter_rows(named=True):
        elexon_id = row["elexon_bmu_id"] or row["national_grid_bmu_id"]
        if elexon_id is None:
            continue
        rows[elexon_id] = row
        keys: set[str] = set()
        if row["national_grid_bmu_id"]:
            national = row["national_grid_bmu_id"].upper()
            by_national[national].append(elexon_id)
            keys.add(national)
        if row["elexon_bmu_id"]:
            full = row["elexon_bmu_id"].upper()
            by_elexon[full].append(elexon_id)
            stripped = BMU_PREFIX_PATTERN.sub("", full)
            by_stripped[stripped].append(elexon_id)
            keys.add(stripped)
        for key in keys:
            by_stem[stem(identifier=key)].append(elexon_id)
            fuzzy[key].append(elexon_id)
    return BmuIndex(by_national, by_elexon, by_stripped, by_stem, rows, fuzzy)


def candidates_for_unit(*, unit: str, index: BmuIndex) -> list[tuple[str, str, float]]:
    """Return `(bmu_id, method, score)` candidates for one auction unit, best first.

    Exact passes run first and stop the search: if any exact pass matches, only those matches are
    returned. Otherwise the unit's stem (the text before its last hyphen) is compared with the
    stem of every BMU, then the unit's stem plus the letter `B` (many battery BMUs add it to
    their site code, so `ERSK-01` may be `ERSKB-1`), then fuzzy identifier similarity.

    Args:
        unit: The auction unit identifier.
        index: The BMU register lookups.

    Returns:
        The candidates, empty if nothing matches.
    """
    key = unit.upper()
    exact_passes = (
        ("exact_national_grid_id", 1.0, index.by_national_grid_id),
        ("exact_elexon_id", 1.0, index.by_elexon_id),
        ("elexon_id_without_prefix", PREFIX_SCORE, index.by_stripped_elexon_id),
    )
    exact = [
        (bmu, method, score)
        for method, score, lookup in exact_passes
        for bmu in lookup.get(key, [])
    ]
    if exact:
        best: dict[str, tuple[str, str, float]] = {}
        for bmu, method, score in exact:
            if bmu not in best or score > best[bmu][2]:
                best[bmu] = (bmu, method, score)
        return sorted(best.values(), key=lambda item: -item[2])
    found: dict[str, tuple[str, str, float]] = {}
    for bmu in index.by_stem.get(stem(identifier=key), []):
        found[bmu] = (bmu, "same_stem_sibling", SIBLING_SCORE)
    for bmu in index.by_stem.get(f"{stem(identifier=key)}B", []):
        found[bmu] = (bmu, "stem_plus_battery_b", BATTERY_STEM_SCORE)
    for close in difflib.get_close_matches(
        key, list(index.fuzzy_ids), n=FUZZY_CANDIDATES, cutoff=MIN_FUZZY_SCORE
    ):
        score = difflib.SequenceMatcher(a=key, b=close).ratio()
        for bmu in index.fuzzy_ids[close]:
            if bmu not in found:
                found[bmu] = (bmu, "fuzzy_id", round(score, 3))
    return sorted(found.values(), key=lambda item: -item[2])


def describe_units(*, results: list[pl.DataFrame]) -> pl.DataFrame:
    """Return one row for each auction unit with its participant, technology, services, and rows.

    A unit with more than one participant or technology label keeps the one on most rows.
    """
    stacked = pl.concat(results)
    most_common = (
        stacked.group_by("auction_unit", "participant", "technology_type")
        .len()
        .sort("len", descending=True)
        .group_by("auction_unit", maintain_order=True)
        .first()
        .select("auction_unit", "participant", "technology_type")
    )
    totals = stacked.group_by("auction_unit").agg(
        pl.col("service_type").unique().sort().str.join(", ").alias("services"),
        pl.len().alias("result_rows"),
    )
    return most_common.join(totals, on="auction_unit").sort("auction_unit")


def build_mapping(
    *, units: pl.DataFrame, reference: pl.DataFrame, study_bmus: set[str]
) -> pl.DataFrame:
    """Link each auction unit to candidate BMUs.

    Args:
        units: The output of `describe_units`.
        reference: The `bmu_reference.parquet` table.
        study_bmus: The Elexon identifiers of the study's BMUs.

    Returns:
        One row for each unit and candidate BMU, or one row with a null candidate and
        `match_method` `none` for a unit with no candidate. The columns are `MAPPING_COLUMNS`.
    """
    index = build_bmu_index(reference=reference)
    rows: list[dict[str, Any]] = []
    for unit in units.iter_rows(named=True):
        found = candidates_for_unit(unit=unit["auction_unit"], index=index)
        if not found:
            found = [("", "none", 0.0)]
        for bmu, method, score in found:
            register = index.rows.get(bmu)
            lead_party = register["lead_party_name"] if register else None
            rows.append(
                {
                    "auction_unit": unit["auction_unit"],
                    "candidate_bmu_id": bmu or None,
                    "match_method": method,
                    "match_score": score,
                    "candidate_national_grid_bmu_id": register["national_grid_bmu_id"]
                    if register
                    else None,
                    "candidate_lead_party": lead_party,
                    "auction_participant": unit["participant"],
                    "party_similarity": party_similarity(
                        left=unit["participant"], right=lead_party
                    ),
                    "technology_type": unit["technology_type"],
                    "services": unit["services"],
                    "result_rows": unit["result_rows"],
                    "candidate_is_study_bmu": bmu in study_bmus,
                }
            )
    return pl.DataFrame(rows, schema=MAPPING_COLUMNS).sort(
        "auction_unit", "match_score", descending=[False, True]
    )


def study_coverage(*, mapping: pl.DataFrame, study_bmus: list[str]) -> pl.DataFrame:
    """Say, for each study BMU, which auction units may belong to it and how sure the match is.

    `plausible` means an exact identifier match (`PLAUSIBLE_METHODS`). `possible` means only a
    sibling or fuzzy match. `no_identifier_match` means no auction unit matched.
    """
    rows: list[dict[str, Any]] = []
    for bmu in study_bmus:
        hits = mapping.filter(pl.col("candidate_bmu_id") == bmu)
        plausible = hits.filter(pl.col("match_method").is_in(PLAUSIBLE_METHODS))
        if plausible.height:
            verdict, chosen = "plausible", plausible
        elif hits.height:
            verdict, chosen = "possible", hits
        else:
            verdict, chosen = "no_identifier_match", hits
        rows.append(
            {
                "elexon_bmu_id": bmu,
                "eac_match": verdict,
                "auction_units": "; ".join(chosen["auction_unit"].to_list()),
                "match_methods": "; ".join(chosen["match_method"].to_list()),
                "services": "; ".join(sorted(set(chosen["services"].to_list()))),
                "result_rows": int(chosen["result_rows"].sum()) if chosen.height else 0,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "elexon_bmu_id": pl.String,
            "eac_match": pl.String,
            "auction_units": pl.String,
            "match_methods": pl.String,
            "services": pl.String,
            "result_rows": pl.Int64,
        },
    )


def fetch_mapping(*, root: Path, bmu_reference: Path, bmu_list: Path) -> None:
    """Build the auction-unit-to-BMU table from the two result parquets and write the folder.

    Args:
        root: The folder that holds one folder for each source.
        bmu_reference: Elexon's BMU register, as `bmu_reference.parquet`.
        bmu_list: The CSV of the study's BMUs.

    Raises:
        FileNotFoundError: If a result parquet has not been written by an earlier run.
    """
    output_dir = root / "neso_eac_unit_to_bmu"
    results = []
    for source in SOURCE_NAMES[:2]:
        path = root / source / f"{source}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: run the {source} source first")
        results.append(pl.read_parquet(path))
    reference = pl.read_parquet(bmu_reference)
    study_bmus = read_bmu_list(path=bmu_list)
    units = describe_units(results=results)
    mapping = build_mapping(units=units, reference=reference, study_bmus=set(study_bmus))
    coverage = study_coverage(mapping=mapping, study_bmus=study_bmus)
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping.write_csv(output_dir / "eac_unit_to_bmu.csv")
    coverage.write_csv(output_dir / "study_bmu_eac_coverage.csv")
    method_counts = _units_by_best_match_method(mapping=mapping)
    verdicts = coverage["eac_match"].value_counts().sort("eac_match")
    verdict_counts = dict(verdicts.iter_rows())
    unmatched = coverage.filter(pl.col("eac_match") == "no_identifier_match")[
        "elexon_bmu_id"
    ].to_list()
    possible = coverage.filter(pl.col("eac_match") == "possible")["elexon_bmu_id"].to_list()
    write_lineage(
        product_dir=output_dir,
        note={
            "script": SCRIPT_PATH,
            "inputs": {
                "result_parquets": [f"{source}/{source}.parquet" for source in SOURCE_NAMES[:2]],
                "bmu_reference": str(bmu_reference),
                "bmu_list": str(bmu_list),
            },
            "auction_units": units.height,
            "mapping_rows": mapping.height,
            "units_by_best_match_method": method_counts,
            "study_bmus": len(study_bmus),
            "study_bmus_by_verdict": verdict_counts,
            "study_bmus_with_no_eac_unit": unmatched,
            "study_bmus_with_only_a_possible_match": possible,
        },
    )
    write_readme(
        product_dir=output_dir,
        script_path=SCRIPT_PATH,
        cache_hint="../neso_eac_response_reserve/_day_cache/",
        **_mapping_readme(
            units=units,
            method_counts=method_counts,
            verdict_counts=verdict_counts,
            unmatched=unmatched,
            possible=possible,
            no_match_units=_describe_no_match_units(mapping=mapping),
        ),
    )
    print(
        f"neso_eac_unit_to_bmu: {units.height} units, {mapping.height} rows; "
        f"study BMUs {verdict_counts}"
    )


def _describe_no_match_units(*, mapping: pl.DataFrame) -> str:
    """Return the technology counts and the five biggest examples of units with no candidate."""
    none = mapping.filter(pl.col("match_method") == "none")
    if none.is_empty():
        return "none in this run"
    technologies = ", ".join(
        f"{count} {technology}"
        for technology, count in none["technology_type"].value_counts(sort=True).iter_rows()
    )
    examples = ", ".join(
        f"`{unit}`" for unit in none.sort("result_rows", descending=True)["auction_unit"][:5]
    )
    return f"{none.height} units ({technologies}); the five with most result rows are {examples}"


def _units_by_best_match_method(*, mapping: pl.DataFrame) -> dict[str, int]:
    order = {
        "exact_national_grid_id": 0,
        "exact_elexon_id": 1,
        "elexon_id_without_prefix": 2,
        "same_stem_sibling": 3,
        "stem_plus_battery_b": 4,
        "fuzzy_id": 5,
        "none": 6,
    }
    best = (
        mapping.with_columns(
            rank=pl.col("match_method").replace_strict(order, return_dtype=pl.Int8)
        )
        .sort("auction_unit", "rank")
        .group_by("auction_unit", maintain_order=True)
        .first()
    )
    counts = best["match_method"].value_counts()
    return dict(sorted(counts.iter_rows(), key=lambda item: order[item[0]]))


def _mapping_readme(
    *,
    units: pl.DataFrame,
    method_counts: dict[str, int],
    verdict_counts: dict[str, int],
    unmatched: list[str],
    possible: list[str],
    no_match_units: str,
) -> dict[str, Any]:
    methods = "; ".join(f"`{method}`: {count}" for method, count in method_counts.items())
    verdicts = ", ".join(f"{count} {verdict}" for verdict, count in verdict_counts.items())
    return {
        "title": (
            "NESO Enduring Auction Capability (EAC) auction unit to balancing mechanism unit "
            "(BMU) mapping"
        ),
        "source_page": EAC_PAGE,
        "attribution": None,
        "licence": (
            "Derived from the NESO EAC results and Elexon's BMU register. This table is a "
            "heuristic match computed by this script, not a published reference."
        ),
        "timestamp_convention": "No time columns. The table is not dated.",
        "columns": {
            "auction_unit": "NESO auction unit identifier, from the EAC results",
            "candidate_bmu_id": (
                "Elexon BMU identifier (or the National Grid identifier for a BMU that has no "
                "Elexon identifier) of a BMU that might be the auction unit; empty if none"
            ),
            "match_method": (
                "How the pair was found: `exact_national_grid_id` (the unit equals the BMU's "
                "National Grid identifier), `exact_elexon_id`, `elexon_id_without_prefix` (equals "
                "the Elexon identifier after removing a prefix such as `E_` or `2__`), "
                "`same_stem_sibling` (same text before the last hyphen, so possibly another unit "
                "at the same site), `stem_plus_battery_b` (the BMU's text before its last hyphen "
                "is the unit's plus the letter B), `fuzzy_id` (similar identifier, ratio at least "
                f"{MIN_FUZZY_SCORE}), or `none`"
            ),
            "match_score": (
                "1.0 for an exact identifier, 0.95 for a prefix-stripped one, 0.8 for a "
                "battery-B stem, 0.7 for a sibling, "
                "the identifier similarity ratio for a fuzzy match, 0 for none. It is a rule's "
                "label, not a probability"
            ),
            "candidate_national_grid_bmu_id": "The candidate's National Grid identifier",
            "candidate_lead_party": "The candidate's lead party in Elexon's register",
            "auction_participant": "The auction unit's registered participant in the EAC results",
            "party_similarity": (
                "Overlap of the distinctive words of the two company names, 0 to 1, after "
                "dropping words such as Limited and Energy; empty if either name has none"
            ),
            "technology_type": "NESO's technology label for the auction unit",
            "services": "The services the unit cleared in the window",
            "result_rows": "Result rows the unit has in the two EAC tables",
            "candidate_is_study_bmu": "Whether the candidate is one of the study's BMUs",
        },
        "row_summary": (
            f"- Auction units: {units.height}. Best match method for each unit: {methods}.\n"
            f"- Study BMUs: {verdicts}.\n- Study BMUs with no auction unit ({len(unmatched)}): "
            f"{', '.join(f'`{bmu}`' for bmu in unmatched) or 'none'}.\n- Study BMUs with only a "
            f"possible (sibling, battery-B, or fuzzy) match ({len(possible)}): "
            f"{', '.join(f'`{bmu}`' for bmu in possible) or 'none'}.\n"
            "- `study_bmu_eac_coverage.csv` has one row for each study BMU."
        ),
        "gotchas": [
            (
                "The EAC results carry no unit name, so no match is by name. Identifier matches "
                "are strong evidence; sibling, battery-B, and fuzzy matches are only leads. "
                "`party_similarity` compares the auction participant (often a trading company "
                "that bids on the asset owner's behalf) with the register's lead party, so a low "
                "value does not refute an exact identifier match, and a high value is only "
                "supporting evidence."
            ),
            (
                "An auction unit can bundle several assets, and one site can have several BMUs. "
                "The table links identifiers, not asset-to-asset ownership, so the volumes of a "
                "matched unit may cover more or less than the BMU's output."
            ),
            (
                f"Units with `match_method` `none` have no identifier match in Elexon's register: "
                f"{no_match_units}. They may be non-BM assets, which this script could not "
                "confirm."
            ),
            (
                "A study BMU with the verdict `no_identifier_match` may still bid inside an "
                "aggregated unit (an identifier starting `AG-`) or under an identifier this "
                "script cannot link, so the verdict does not show that the BMU sold nothing "
                "into the EAC."
            ),
            (
                "The register is the snapshot in `elexon_bmu_reference/`, taken on its own "
                "retrieval date."
            ),
        ],
    }


def main() -> None:
    """Download the named sources, or all three."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--sources", nargs="+", choices=SOURCE_NAMES, default=list(SOURCE_NAMES))
    parser.add_argument("--start", type=date.fromisoformat, default=WINDOW_START)
    parser.add_argument("--end", type=date.fromisoformat, default=WINDOW_END)
    parser.add_argument("--output-root", type=Path, default=MARKET_DOWNLOADS_DIR)
    parser.add_argument("--threads", type=int, default=FETCH_THREADS)
    parser.add_argument(
        "--bmu-reference",
        type=Path,
        default=MARKET_DOWNLOADS_DIR / "elexon_bmu_reference" / "bmu_reference.parquet",
    )
    parser.add_argument(
        "--bmu-list",
        type=Path,
        default=Path(
            "/home/jack/dev/nged-substation-forecast/data/studies/per_study/"
            "battery_pv_separation/bmu_list.csv"
        ),
    )
    args = parser.parse_args()
    runners: dict[str, Callable[[], None]] = {
        "neso_eac_response_reserve": partial(
            fetch_results,
            source="neso_eac_response_reserve",
            root=args.output_root,
            start=args.start,
            end=args.end,
            threads=args.threads,
        ),
        "neso_eac_balancing_reserve": partial(
            fetch_results,
            source="neso_eac_balancing_reserve",
            root=args.output_root,
            start=args.start,
            end=args.end,
            threads=args.threads,
        ),
        "neso_eac_unit_to_bmu": partial(
            fetch_mapping,
            root=args.output_root,
            bmu_reference=args.bmu_reference,
            bmu_list=args.bmu_list,
        ),
    }
    for name in sorted(args.sources, key=SOURCE_NAMES.index):
        runners[name]()
    print(f"Finished at {datetime.now(UTC):%Y-%m-%d %H:%M} UTC")


if __name__ == "__main__":
    main()
