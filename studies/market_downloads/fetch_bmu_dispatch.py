"""Download the balancing-mechanism dispatch of storage Balancing Mechanism Units (BMUs).

The window is 2025-09-01 to 2026-09-30, and the data comes from Elexon. Written for the
battery-versus-solar-PV study. Run in two phases, so that a person can review the BMU list between
them:

    uv run python studies/market_downloads/fetch_bmu_dispatch.py --phase reference
    uv run python studies/market_downloads/fetch_bmu_dispatch.py --phase dispatch

**Phase `reference`** downloads Elexon's register of every BMU into
`elexon_bmu_reference/bmu_reference.parquet` and writes `elexon_bmu_reference/bmu_candidates.csv`: a
list of candidate storage BMUs for a person to hand-check. The candidates are every pumped-storage
BMU, every BMU with fuel type `OTHER`, every BMU with no fuel type whose name, lead party, or
identifier looks like a battery's (such as `T_LKSDB-1`), and the two BMUs the study names. An
`OTHER` BMU with such a hint is classed `battery_hint`. Every other `OTHER` BMU is classed
`other_fuel_unhinted`. The list leans towards including too many. The script
never overwrites an existing `bmu_candidates.csv`, because a person may have edited it; pass
`--overwrite-candidates` to replace it.

**Phase `dispatch`** reads the candidate list (`--bmu-list` points at an edited copy) and downloads,
for each BMU in it, three tables:

- `elexon_boav`: accepted bid and offer volumes (BOAV) for each settlement period, one request for
  each settlement day, side, and batch of BMUs.
- `elexon_ebocf`: indicative cashflows from accepted bids and offers (EBOCF), requested the same
  way. The table holds no price. A user of the table derives the price as `total_cashflow_gbp`
  divided by `total_volume_accepted_mwh` in `elexon_boav`.
- `elexon_boalf`: the acceptance levels (BOALF), one request for each calendar month and batch of
  BMUs.

Each table is cached one chunk at a time in `_day_cache/batch_<hash>/`. BMUs are grouped into
batches by a hash of each identifier, and each batch's folder is named by a hash of that batch's
BMUs. A re-run therefore resumes where it stopped, and removing one BMU from the list refetches only
that BMU's batch. Pass `--start`, `--end`, and `--output-root` for a small test run. Requests are
keyless, use four threads, and back off exponentially on HTTP 429 and 5xx.
"""

import argparse
import hashlib
import re
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, Final

import polars as pl
from market_common import (
    ELEXON_API,
    ELEXON_ATTRIBUTION,
    ELEXON_LICENCE,
    FETCH_THREADS,
    WINDOW_END,
    WINDOW_START,
    days_between,
    expected_settlement_starts,
    fetch_missing_chunks,
    get_json,
    period_time_mismatches,
    read_chunks,
    utc_midnight,
    write_lineage,
    write_parquet_atomic,
    write_readme,
)
from studies.sources import MARKET_DOWNLOADS_DIR

SCRIPT_PATH: Final[str] = "studies/market_downloads/fetch_bmu_dispatch.py"
BMU_REFERENCE_URL: Final[str] = f"{ELEXON_API}/reference/bmunits/all"
BOALF_URL: Final[str] = f"{ELEXON_API}/datasets/BOALF/stream"
DATA_STATUS_URL: Final[str] = f"{ELEXON_API}/data-status"
SIDES: Final[tuple[str, ...]] = ("bid", "offer")
BATCH_COUNT: Final[int] = 16
"""BMUs are split into this many batches by a hash of each identifier.

Adding or deleting a BMU therefore changes the cache key of one batch only. Each batch holds about
one sixteenth of the list. A batch of 16 identifiers of 12 characters each makes a URL of about 500
bytes.
"""
MAX_URL_LENGTH: Final[int] = 4000
"""A batch whose URL would be longer than `MAX_URL_LENGTH` bytes raises `ValueError` instead of
being sent."""
DATA_STATUS_DAYS: Final[int] = 30
"""The data-status endpoint rejects a settlement-date range of more than 31 days."""
PAIR_COUNT: Final[int] = 6
ALWAYS_INCLUDE: Final[tuple[str, ...]] = ("T_LKSDB-1", "E_DOLLB-1")
"""Two storage BMUs the study names, included in the candidate list whatever the reference says."""
BATTERY_KEYWORDS: Final[tuple[str, ...]] = ("STORAGE", "BATTERY", "BESS")
BATTERY_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[TE]_[A-Z0-9]+B-\d+$")
"""A BMU identifier ending in `B-<n>`, such as `T_LKSDB-1`, a naming pattern many battery sites
follow."""
KNOWN_SOURCE_GAPS: Final[dict[str, list[tuple[str, int]]]] = {
    "BOAV": [("2025-10-14", 20), ("2025-10-14", 21), ("2026-01-18", 26), ("2026-06-23", 8)],
    "EBOCF": [("2025-10-14", 20), ("2025-10-14", 21), ("2026-01-18", 26), ("2026-06-23", 8)],
    "BOALF": [
        ("2025-10-14", 20),
        ("2025-10-14", 21),
        ("2026-01-18", 26),
        ("2026-06-23", 8),
        ("2025-11-23", 29),
    ],
}
"""Periods found missing at the source before this script was written, as `(settlement date,
period)`. The script reports which of them it still finds missing; it never fails on them."""

UTC_TIME: Final[pl.Datetime] = pl.Datetime(time_unit="us", time_zone="UTC")
ISO_SECONDS: Final[str] = "%Y-%m-%dT%H:%M:%SZ"

REFERENCE_COLUMNS: Final[dict[str, str]] = {
    "elexonBmUnit": "elexon_bmu_id",
    "nationalGridBmUnit": "national_grid_bmu_id",
    "eic": "eic",
    "fuelType": "fuel_type",
    "leadPartyName": "lead_party_name",
    "leadPartyId": "lead_party_id",
    "bmUnitType": "bmu_type",
    "bmUnitName": "bmu_name",
    "fpnFlag": "fpn_flag",
    "productionOrConsumptionFlag": "production_or_consumption_flag",
    "demandCapacity": "demand_capacity_mw",
    "generationCapacity": "generation_capacity_mw",
    "transmissionLossFactor": "transmission_loss_factor",
    "creditQualifyingStatus": "credit_qualifying_status",
    "gspGroupId": "gsp_group_id",
    "gspGroupName": "gsp_group_name",
    "interconnectorId": "interconnector_id",
}
"""Elexon's field names mapped to this table's column names."""
NUMERIC_REFERENCE_COLUMNS: Final[tuple[str, ...]] = (
    "demand_capacity_mw",
    "generation_capacity_mw",
    "transmission_loss_factor",
)


def _pair_columns(*, prefix: str) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for index in range(1, PAIR_COUNT + 1):
        columns[f"{prefix}_neg{index}"] = pl.Float64
        columns[f"{prefix}_pos{index}"] = pl.Float64
    return columns


BOAV_SCHEMA: Final[dict[str, Any]] = {
    "time": UTC_TIME,
    "settlement_date": pl.Date,
    "settlement_period": pl.Int16,
    "side": pl.String,
    "bmu_id": pl.String,
    "acceptance_id": pl.Int64,
    "acceptance_duration": pl.String,
    "total_volume_accepted_mwh": pl.Float64,
    "created_time": UTC_TIME,
    **_pair_columns(prefix="pair_volume"),
}
EBOCF_SCHEMA: Final[dict[str, Any]] = {
    "time": UTC_TIME,
    "settlement_date": pl.Date,
    "settlement_period": pl.Int16,
    "side": pl.String,
    "bmu_id": pl.String,
    "total_cashflow_gbp": pl.Float64,
    "created_time": UTC_TIME,
    **_pair_columns(prefix="pair_cashflow"),
}
BOALF_SCHEMA: Final[dict[str, Any]] = {
    "bmu_id": pl.String,
    "acceptance_number": pl.Int64,
    "acceptance_time": UTC_TIME,
    "settlement_date": pl.Date,
    "settlement_period_from": pl.Int16,
    "settlement_period_to": pl.Int16,
    "time_from": UTC_TIME,
    "time_to": UTC_TIME,
    "level_from_mw": pl.Float64,
    "level_to_mw": pl.Float64,
    "deemed_bo_flag": pl.Boolean,
    "so_flag": pl.Boolean,
    "amendment_flag": pl.String,
    "stor_flag": pl.Boolean,
    "rr_flag": pl.Boolean,
}
STATUS_SCHEMA: Final[dict[str, Any]] = {
    "settlement_date": pl.Date,
    "settlement_period": pl.Int16,
    "data_point_count": pl.Int64,
}


def _utc(*, column: str) -> pl.Expr:
    return pl.col(column).str.to_datetime(format=ISO_SECONDS, time_zone="UTC", time_unit="us")


# Phase 1: the BMU reference and the candidate list


def parse_reference(*, rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Turn Elexon's BMU register into a table with snake-case columns and numeric capacities."""
    frame = pl.DataFrame(
        [{new: row.get(old) for old, new in REFERENCE_COLUMNS.items()} for row in rows],
        schema=dict.fromkeys(REFERENCE_COLUMNS.values(), pl.String)
        | {
            "fpn_flag": pl.Boolean,
            "credit_qualifying_status": pl.Boolean,
        },
    )
    return frame.with_columns(
        pl.col(*NUMERIC_REFERENCE_COLUMNS).cast(pl.Float64, strict=False)
    ).sort("elexon_bmu_id")


def battery_hint(*, bmu_id: str | None, bmu_name: str | None, lead_party: str | None) -> str:
    """Say why a BMU looks like a battery, or return an empty string if it does not.

    Args:
        bmu_id: The Elexon BMU identifier.
        bmu_name: The BMU's registered name.
        lead_party: The name of the BMU's lead party.

    Returns:
        A semicolon-separated reason for each rule that matched: a keyword (`storage`, `battery`,
        or `BESS`) in the name or lead party, or an identifier ending in `B-<n>`.
    """
    reasons = [
        f"{label} contains {keyword.lower()}"
        for label, text in (("name", bmu_name), ("lead party", lead_party))
        for keyword in BATTERY_KEYWORDS
        if re.search(rf"\b{keyword}\b", (text or "").upper())
        or (keyword != "BESS" and keyword in (text or "").upper())
    ]
    if bmu_id and BATTERY_ID_PATTERN.match(bmu_id):
        reasons.append("identifier ends in B-<n>")
    return "; ".join(reasons)


def build_candidates(
    *, reference: pl.DataFrame, always_include: Sequence[str] = ALWAYS_INCLUDE
) -> pl.DataFrame:
    """List the BMUs that might be storage, for a person to review.

    A BMU is a candidate if its fuel type is `PS` (pumped storage), if its fuel type is `OTHER` or
    missing and `battery_hint` matches, if its fuel type is `OTHER` with no hint, or if it is in
    `always_include`. The rule errs towards including too many.

    Args:
        reference: The output of `parse_reference`.
        always_include: BMU identifiers included even when the reference lacks them or no rule
            matches.

    Returns:
        One row for each candidate, sorted by BMU identifier, with `candidate_class`
        (`pumped_storage`, `battery_hint`, `other_fuel_unhinted`, or `named_by_study`),
        `battery_hint` (whether any hint rule matched, or the BMU is named by the study),
        `hint_reason`, and seven columns copied from the register: fuel type, name, lead party,
        BMU type, generation and demand capacity, and Grid Supply Point (GSP) group name.
    """
    rows = []
    seen = set()
    for row in reference.iter_rows(named=True):
        bmu_id = row["elexon_bmu_id"]
        if bmu_id is None or bmu_id in seen:
            continue
        fuel = row["fuel_type"]
        hint = battery_hint(
            bmu_id=bmu_id, bmu_name=row["bmu_name"], lead_party=row["lead_party_name"]
        )
        if fuel == "PS":
            candidate_class = "pumped_storage"
        elif fuel in ("OTHER", None) and hint:
            candidate_class = "battery_hint"
        elif fuel == "OTHER":
            candidate_class = "other_fuel_unhinted"
        elif bmu_id in always_include:
            candidate_class = "named_by_study"
        else:
            continue
        seen.add(bmu_id)
        rows.append(
            {
                "elexon_bmu_id": bmu_id,
                "candidate_class": candidate_class,
                "battery_hint": bool(hint),
                "hint_reason": hint,
                "fuel_type": fuel,
                "bmu_name": row["bmu_name"],
                "lead_party_name": row["lead_party_name"],
                "bmu_type": row["bmu_type"],
                "generation_capacity_mw": row["generation_capacity_mw"],
                "demand_capacity_mw": row["demand_capacity_mw"],
                "gsp_group_name": row["gsp_group_name"],
            }
        )
    rows.extend(
        {
            "elexon_bmu_id": bmu_id,
            "candidate_class": "named_by_study",
            "battery_hint": True,
            "hint_reason": "named by the study; absent from the BMU reference",
        }
        for bmu_id in always_include
        if bmu_id not in seen
    )
    schema = {
        "elexon_bmu_id": pl.String,
        "candidate_class": pl.String,
        "battery_hint": pl.Boolean,
        "hint_reason": pl.String,
        "fuel_type": pl.String,
        "bmu_name": pl.String,
        "lead_party_name": pl.String,
        "bmu_type": pl.String,
        "generation_capacity_mw": pl.Float64,
        "demand_capacity_mw": pl.Float64,
        "gsp_group_name": pl.String,
    }
    return pl.DataFrame(rows, schema=schema).sort("elexon_bmu_id")


def run_reference_phase(*, root: Path, overwrite_candidates: bool) -> None:
    """Download the BMU register, write the candidate list, and write the folder's notes."""
    output_dir = root / "elexon_bmu_reference"
    rows = get_json(url=BMU_REFERENCE_URL)
    reference = parse_reference(rows=rows)
    write_parquet_atomic(frame=reference, path=output_dir / "bmu_reference.parquet")
    candidates = build_candidates(reference=reference)
    candidates_path = output_dir / "bmu_candidates.csv"
    if candidates_path.exists() and not overwrite_candidates:
        print(f"{candidates_path.name} exists, so it was left alone (it may hold hand edits)")
    else:
        candidates.write_csv(candidates_path)
    by_class = candidates["candidate_class"].value_counts().sort("candidate_class")
    class_lines = ", ".join(f"{row[0]}: {row[1]}" for row in by_class.iter_rows())
    write_lineage(
        product_dir=output_dir,
        note={
            "source_address": BMU_REFERENCE_URL,
            "request": "The whole BMU register",
            "script": SCRIPT_PATH,
            "rows": reference.height,
            "candidates": candidates.height,
            "candidates_by_class": dict(by_class.iter_rows()),
            "fuel_type_counts": dict(reference["fuel_type"].value_counts().iter_rows()),
        },
    )
    write_readme(
        product_dir=output_dir,
        title="Elexon BMU register and the storage-candidate list",
        source_page="https://bmrs.elexon.co.uk/bm-units",
        script_path=SCRIPT_PATH,
        attribution=ELEXON_ATTRIBUTION,
        cache_hint="(none: the register is one request)",
        licence=ELEXON_LICENCE,
        timestamp_convention=(
            "The register has no time column. It is the register on the retrieval date."
        ),
        columns={
            "elexon_bmu_id": (
                "Elexon Balancing Mechanism Unit (BMU) identifier, the key used by the accepted "
                "volumes (BOAV), indicative cashflows (EBOCF), and acceptance levels (BOALF)"
            ),
            "national_grid_bmu_id": "National Grid's BMU identifier",
            "fuel_type": "Elexon fuel type, for example PS, OTHER, WIND; null for most BMUs",
            "lead_party_name": "Name of the BMU's lead party",
            "bmu_name": "Registered BMU name",
            "bmu_type": "Elexon BMU type letter (T transmission, E embedded, and others)",
            "demand_capacity_mw": "Registered demand capacity, MW",
            "generation_capacity_mw": "Registered generation capacity, MW",
            "other_columns": (
                "The remaining nine columns, `eic`, `lead_party_id`, `fpn_flag`, "
                "`production_or_consumption_flag`, `transmission_loss_factor`, "
                "`credit_qualifying_status`, `gsp_group_id`, `gsp_group_name`, and "
                "`interconnector_id`, follow Elexon's register field names"
            ),
        },
        row_summary=(
            f"- `bmu_reference.parquet`: {reference.height} BMUs.\n"
            f"- `bmu_candidates.csv`: {candidates.height} candidate storage BMUs ({class_lines}).\n"
            "- `bmu_candidates.csv` columns: `candidate_class` says which rule included the BMU "
            "(`pumped_storage`, `battery_hint`, `other_fuel_unhinted`, or `named_by_study`), "
            "`battery_hint` is true if a name or identifier rule matched or the BMU is named by "
            "the study, and `hint_reason` says which rule matched. A person reviews this file "
            "and deletes the rows that are not storage."
        ),
        gotchas=[
            (
                "The candidate list errs towards including too many BMUs. Hand-check it before "
                "running the dispatch phase."
            ),
            (
                "Batteries are found by the name and identifier rules, and by listing every BMU "
                "with fuel type `OTHER` as `other_fuel_unhinted`. The register has no battery "
                "fuel type."
            ),
        ],
    )
    print(f"Wrote {reference.height} BMUs and {candidates.height} candidates to {output_dir}")


# Phase 2: dispatch


def read_bmu_list(*, path: Path) -> list[str]:
    """Return the sorted, de-duplicated `elexon_bmu_id` values of a candidate CSV."""
    ids = pl.read_csv(path, infer_schema_length=0)["elexon_bmu_id"].drop_nulls().to_list()
    return sorted({bmu_id.strip() for bmu_id in ids if bmu_id.strip()})


def list_hash(*, bmu_ids: Sequence[str]) -> str:
    """Return a short hash of a BMU list, used to name the cache folder."""
    return hashlib.sha1("\n".join(sorted(bmu_ids)).encode()).hexdigest()[:8]


def bmu_batches(*, bmu_ids: Sequence[str], count: int = BATCH_COUNT) -> list[list[str]]:
    """Split BMU identifiers into up to `count` batches by a hash of each identifier.

    A BMU's batch depends on its own identifier alone, so editing the list changes only the batches
    that gain or lose a BMU.

    Args:
        bmu_ids: The BMU identifiers.
        count: The number of hash buckets.

    Returns:
        The non-empty batches, each sorted, in bucket order.

    Raises:
        ValueError: If a batch would make a URL longer than `MAX_URL_LENGTH`.
    """
    buckets: list[list[str]] = [[] for _ in range(count)]
    for bmu_id in sorted(bmu_ids):
        bucket = int(hashlib.sha1(bmu_id.encode()).hexdigest()[:8], 16) % count
        buckets[bucket].append(bmu_id)
    batches = [batch for batch in buckets if batch]
    for batch in batches:
        length = 200 + sum(len(bmu_id) + len("&bmUnit=") for bmu_id in batch)
        if length > MAX_URL_LENGTH:
            raise ValueError(f"A batch of {len(batch)} BMUs would make a {length}-byte URL")
    return batches


def bmu_params(*, batch: Sequence[str]) -> list[tuple[str, str | float | None]]:
    """Return the repeated `bmUnit=` query pairs for a batch."""
    return [("bmUnit", bmu_id) for bmu_id in batch]


def _pair_values(*, rows: list[dict[str, Any]], field: str, prefix: str) -> dict[str, list[Any]]:
    values: dict[str, list[Any]] = {}
    for index in range(1, PAIR_COUNT + 1):
        for sign, word in (("negative", "neg"), ("positive", "pos")):
            values[f"{prefix}_{word}{index}"] = [row[field][f"{sign}{index}"] for row in rows]
    return values


def parse_boav(*, rows: list[dict[str, Any]], side: str) -> pl.DataFrame:
    """Turn BOAV rows of one side into a table."""
    data: dict[str, list[Any]] = {
        "time": [row["startTime"] for row in rows],
        "settlement_date": [row["settlementDate"] for row in rows],
        "settlement_period": [row["settlementPeriod"] for row in rows],
        "side": [side] * len(rows),
        "bmu_id": [row["bmUnit"] for row in rows],
        "acceptance_id": [row["acceptanceId"] for row in rows],
        "acceptance_duration": [row["acceptanceDuration"] for row in rows],
        "total_volume_accepted_mwh": [row["totalVolumeAccepted"] for row in rows],
        "created_time": [row["createdDateTime"] for row in rows],
        **_pair_values(rows=rows, field="pairVolumes", prefix="pair_volume"),
    }
    return _typed(data=data, schema=BOAV_SCHEMA).unique(
        subset=["side", "bmu_id", "time", "acceptance_id"], keep="last", maintain_order=True
    )


def parse_ebocf(*, rows: list[dict[str, Any]], side: str) -> pl.DataFrame:
    """Turn EBOCF rows of one side into a table."""
    data: dict[str, list[Any]] = {
        "time": [row["startTime"] for row in rows],
        "settlement_date": [row["settlementDate"] for row in rows],
        "settlement_period": [row["settlementPeriod"] for row in rows],
        "side": [side] * len(rows),
        "bmu_id": [row["bmUnit"] for row in rows],
        "total_cashflow_gbp": [row["totalCashflow"] for row in rows],
        "created_time": [row["createdDateTime"] for row in rows],
        **_pair_values(rows=rows, field="bidOfferPairCashflows", prefix="pair_cashflow"),
    }
    return _typed(data=data, schema=EBOCF_SCHEMA).unique(
        subset=["side", "bmu_id", "time"], keep="last", maintain_order=True
    )


def parse_boalf(*, rows: list[dict[str, Any]], start: datetime, end: datetime) -> pl.DataFrame:
    """Turn BOALF rows into a table, keeping only `start <= time_from < end`."""
    data: dict[str, list[Any]] = {
        "bmu_id": [row["bmUnit"] for row in rows],
        "acceptance_number": [row["acceptanceNumber"] for row in rows],
        "acceptance_time": [row["acceptanceTime"] for row in rows],
        "settlement_date": [row["settlementDate"] for row in rows],
        "settlement_period_from": [row["settlementPeriodFrom"] for row in rows],
        "settlement_period_to": [row["settlementPeriodTo"] for row in rows],
        "time_from": [row["timeFrom"] for row in rows],
        "time_to": [row["timeTo"] for row in rows],
        "level_from_mw": [row["levelFrom"] for row in rows],
        "level_to_mw": [row["levelTo"] for row in rows],
        "deemed_bo_flag": [row["deemedBoFlag"] for row in rows],
        "so_flag": [row["soFlag"] for row in rows],
        "amendment_flag": [row["amendmentFlag"] for row in rows],
        "stor_flag": [row["storFlag"] for row in rows],
        "rr_flag": [row["rrFlag"] for row in rows],
    }
    return (
        _typed(data=data, schema=BOALF_SCHEMA)
        .filter(pl.col("time_from") >= start, pl.col("time_from") < end)
        .unique(maintain_order=True)
        .sort("bmu_id", "time_from", "acceptance_number")
    )


def _typed(*, data: dict[str, list[Any]], schema: dict[str, Any]) -> pl.DataFrame:
    """Build a frame from raw lists, parsing the timestamp strings and the settlement date."""
    raw_schema = {
        name: (pl.String if dtype in (UTC_TIME, pl.Date) else dtype)
        for name, dtype in schema.items()
    }
    frame = pl.DataFrame(data, schema=raw_schema)
    for name, dtype in schema.items():
        if dtype == UTC_TIME:
            frame = frame.with_columns(_utc(column=name))
        elif dtype == pl.Date:
            frame = frame.with_columns(pl.col(name).str.to_date(format="%Y-%m-%d"))
    return frame


def fetch_settlement_day(
    key: str, *, url_fragment: str, parse: Callable[..., pl.DataFrame], batches: list[list[str]]
) -> pl.DataFrame:
    """Fetch one settlement day of BOAV or EBOCF for every side and every batch of BMUs."""
    frames = []
    for side in SIDES:
        for batch in batches:
            body = get_json(
                url=f"{ELEXON_API}/balancing/settlement/{url_fragment}/all/{side}/{key}",
                params=bmu_params(batch=batch),
            )
            frames.append(parse(rows=body["data"], side=side))
    return pl.concat(frames)


def fetch_boalf_month(key: str, *, batches: list[list[str]]) -> pl.DataFrame:
    """Fetch BOALF for every batch of BMUs over the chunk `<first day>_<day after last>`."""
    first_text, after_text = key.split("_")
    first = utc_midnight(day=date.fromisoformat(first_text))
    after = utc_midnight(day=date.fromisoformat(after_text))
    frames = []
    for batch in batches:
        rows = get_json(
            url=BOALF_URL,
            params=[
                ("from", f"{first:%Y-%m-%dT%H:%MZ}"),
                ("to", f"{after:%Y-%m-%dT%H:%MZ}"),
                *bmu_params(batch=batch),
            ],
        )
        frames.append(parse_boalf(rows=rows, start=first, end=after))
    return pl.concat(frames)


def month_chunks(*, start: date, end: date) -> list[str]:
    """Split `start..end` (inclusive) at month boundaries into `<first>_<day after last>` keys."""
    keys = []
    first = start
    while first <= end:
        next_month = (first.replace(day=1) + timedelta(days=32)).replace(day=1)
        after = min(next_month, end + timedelta(days=1))
        keys.append(f"{first.isoformat()}_{after.isoformat()}")
        first = after
    return keys


def fetch_status_chunk(key: str, *, dataset: str) -> pl.DataFrame:
    """Fetch Elexon's data-status counts for a chunk `<first>_<last>` of settlement dates.

    The endpoint reads its dates as UTC days, so a query for the settlement dates `first` to `last`
    leaves out the first two periods of `first` while the clocks are on summer time. The request
    therefore starts one day early. The extra rows are harmless, because gaps are only looked up
    for settlement dates in the window.
    """
    first_text, last = key.split("_")
    first = (date.fromisoformat(first_text) - timedelta(days=1)).isoformat()
    body = get_json(
        url=f"{DATA_STATUS_URL}/{dataset}",
        params=[("settlementDateFrom", first), ("settlementDateTo", last), ("format", "json")],
    )
    rows = body["data"]
    return pl.DataFrame(
        {
            "settlement_date": [row["settlementDate"] for row in rows],
            "settlement_period": [row["settlementPeriod"] for row in rows],
            "data_point_count": [row["dataPointCount"] for row in rows],
        },
        schema={
            "settlement_date": pl.String,
            "settlement_period": pl.Int16,
            "data_point_count": pl.Int64,
        },
    ).with_columns(pl.col("settlement_date").str.to_date(format="%Y-%m-%d"))


def status_chunks(*, start: date, end: date) -> list[str]:
    """Split `start..end` into `<first>_<last>` keys of at most 30 days."""
    keys = []
    first = start
    while first <= end:
        last = min(first + timedelta(days=DATA_STATUS_DAYS - 1), end)
        keys.append(f"{first.isoformat()}_{last.isoformat()}")
        first = last + timedelta(days=1)
    return keys


def source_gaps(*, status: pl.DataFrame, dataset: str, start: date, end: date) -> dict[str, Any]:
    """Compare Elexon's data-status counts with every settlement period in the window.

    Args:
        status: Rows of `settlement_date`, `settlement_period`, and `data_point_count`.
        dataset: The Elexon dataset name, a key of `KNOWN_SOURCE_GAPS`.
        start: First settlement date.
        end: Last settlement date, inclusive.

    Returns:
        The periods with no data point (as `YYYY-MM-DD:period`), and which of the known source gaps
        inside the window are still missing and which are now present.
    """
    present = {
        (row[0], row[1]) for row in status.filter(pl.col("data_point_count") > 0).iter_rows()
    }
    missing = []
    for day in days_between(start=start, end=end):
        count = len(expected_settlement_starts(start=day, end=day))
        missing += [
            f"{day.isoformat()}:{period}"
            for period in range(1, count + 1)
            if (day, period) not in present
        ]
    known = [
        f"{day}:{period}"
        for day, period in KNOWN_SOURCE_GAPS[dataset]
        if start <= date.fromisoformat(day) <= end
    ]
    return {
        "periods_with_no_data_point": missing,
        "known_gaps_still_missing": [gap for gap in known if gap in missing],
        "known_gaps_now_present": [gap for gap in known if gap not in missing],
        "missing_not_in_known_list": [gap for gap in missing if gap not in known],
    }


def _gap_text(*, gaps: dict[str, Any]) -> str:
    return (
        "- Settlement periods with no data point at the source, according to Elexon's "
        f"data-status endpoint: {len(gaps['periods_with_no_data_point'])} "
        f"({', '.join(gaps['periods_with_no_data_point']) or 'none'}).\n"
        "- Known source gaps (`KNOWN_SOURCE_GAPS` in the script) inside this window, still "
        f"missing: {', '.join(gaps['known_gaps_still_missing']) or 'none'}. Known source gaps now "
        f"present: {', '.join(gaps['known_gaps_now_present']) or 'none'}.\n"
        "- Periods missing at the source and not among the known source gaps: "
        f"{', '.join(gaps['missing_not_in_known_list']) or 'none'}."
    )


def run_dispatch_phase(
    *,
    root: Path,
    bmu_list: Path,
    start: date,
    end: date,
    sources: Sequence[str],
    threads: int,
) -> None:
    """Download BOAV, EBOCF, and BOALF for the BMUs in `bmu_list` and write each folder."""
    bmu_ids = read_bmu_list(path=bmu_list)
    if not bmu_ids:
        raise ValueError(f"{bmu_list} lists no BMUs")
    batches = bmu_batches(bmu_ids=bmu_ids)
    print(f"{len(bmu_ids)} BMUs in {len(batches)} batches")
    days = [day.isoformat() for day in days_between(start=start, end=end)]
    tables: dict[str, dict[str, Any]] = {
        "elexon_boav": {
            "dataset": "BOAV",
            "keys": days,
            "fetch": partial(
                fetch_settlement_day,
                url_fragment="acceptance/volumes",
                parse=parse_boav,
            ),
            "schema": BOAV_SCHEMA,
            "title": "Elexon accepted bid and offer volumes (BOAV), for the listed BMUs",
            "page": "https://bmrs.elexon.co.uk/bid-offer-acceptance-volumes",
        },
        "elexon_ebocf": {
            "dataset": "EBOCF",
            "keys": days,
            "fetch": partial(
                fetch_settlement_day,
                url_fragment="indicative/cashflows",
                parse=parse_ebocf,
            ),
            "schema": EBOCF_SCHEMA,
            "title": "Elexon indicative bid and offer cashflows (EBOCF), for the listed BMUs",
            "page": "https://bmrs.elexon.co.uk/indicative-cashflows",
        },
        "elexon_boalf": {
            "dataset": "BOALF",
            "keys": month_chunks(start=start, end=end),
            "fetch": fetch_boalf_month,
            "schema": BOALF_SCHEMA,
            "title": "Elexon bid-offer acceptance levels (BOALF), for the listed BMUs",
            "page": "https://bmrs.elexon.co.uk/bid-offer-acceptance-level",
        },
    }
    for name in sources:
        table = tables[name]
        output_dir = root / name
        chunks: dict[str, list[str]] = {"fetched": [], "cached": [], "not_published": []}
        parts = []
        for batch in batches:
            cache_dir = output_dir / "_day_cache" / f"batch_{list_hash(bmu_ids=batch)}"
            outcome = fetch_missing_chunks(
                cache_dir=cache_dir,
                keys=table["keys"],
                fetch=partial(table["fetch"], batches=[batch]),
                label=f"{name} {cache_dir.name}",
                threads=threads,
            )
            for kind, keys in outcome.items():
                chunks[kind] += keys
            parts.append(
                read_chunks(cache_dir=cache_dir, keys=table["keys"], schema=table["schema"])
            )
        frame = pl.concat(parts)
        frame = frame.sort("bmu_id", "time_from" if name == "elexon_boalf" else "time")
        write_parquet_atomic(frame=frame, path=output_dir / f"{name}.parquet")
        status_keys = status_chunks(start=start, end=end)
        fetch_missing_chunks(
            cache_dir=output_dir / "_day_cache" / f"data_status_{table['dataset']}",
            keys=status_keys,
            fetch=partial(fetch_status_chunk, dataset=table["dataset"]),
            label=f"{name} data-status",
            threads=threads,
        )
        status = read_chunks(
            cache_dir=output_dir / "_day_cache" / f"data_status_{table['dataset']}",
            keys=status_keys,
            schema=STATUS_SCHEMA,
        )
        gaps = source_gaps(status=status, dataset=table["dataset"], start=start, end=end)
        mismatches = 0 if name == "elexon_boalf" else period_time_mismatches(frame=frame)
        distinct_bmus = frame["bmu_id"].n_unique()
        write_lineage(
            product_dir=output_dir,
            note={
                "source_address": (
                    BOALF_URL
                    if name == "elexon_boalf"
                    else f"{ELEXON_API}/balancing/settlement/..."
                ),
                "request": f"{len(bmu_ids)} BMUs in {len(batches)} hash batches, {start} to {end}",
                "window": [start.isoformat(), end.isoformat()],
                "script": SCRIPT_PATH,
                "bmus_requested": bmu_ids,
                "bmu_list_hash": list_hash(bmu_ids=bmu_ids),
                "rows": frame.height,
                "bmus_with_rows": distinct_bmus,
                "chunks_not_published": sorted(set(chunks["not_published"])),
                "source_gaps": gaps,
                "rows_whose_time_differs_from_date_and_period": mismatches,
            },
        )
        _write_dispatch_readme(
            name=name,
            table=table,
            output_dir=output_dir,
            frame=frame,
            bmu_count=len(bmu_ids),
            distinct_bmus=distinct_bmus,
            gaps=gaps,
            mismatches=mismatches,
            start=start,
            end=end,
        )
        print(f"{name}: wrote {frame.height} rows for {distinct_bmus} of {len(bmu_ids)} BMUs")


def _write_dispatch_readme(
    *,
    name: str,
    table: dict[str, Any],
    output_dir: Path,
    frame: pl.DataFrame,
    bmu_count: int,
    distinct_bmus: int,
    gaps: dict[str, Any],
    mismatches: int,
    start: date,
    end: date,
) -> None:
    """Write the README of one dispatch table."""
    settlement = name != "elexon_boalf"
    timestamp = (
        "`time` is the UTC start of the settlement period (Elexon's `startTime`). "
        "`settlement_date` and `settlement_period` are Elexon's own: period 1 starts at 00:00 UK "
        "local time, so a settlement date has 48 periods, 46 on the day the clocks go forward, and "
        "50 on the day they go back. The window is by settlement date. "
        f"Rows whose `time` differs from the UTC start computed from date and period: {mismatches}."
        if settlement
        else "`time_from` and `time_to` are UTC instants (Elexon's `timeFrom` and `timeTo`), to "
        "the minute. The window is by calendar month of `time_from` in UTC. `settlement_date` and "
        "the period numbers are Elexon's own, with period 1 starting at 00:00 UK local time."
    )
    columns = {
        "bmu_id": "Elexon Balancing Mechanism Unit (BMU) identifier",
        **(
            {
                "time": "Start of the settlement period, UTC",
                "settlement_date": "Elexon settlement date (UK local day)",
                "settlement_period": "Elexon settlement period within the date, from 1",
                "side": "`bid` (the BMU was paid to reduce output) or `offer` (to increase it)",
                "created_time": (
                    "When Elexon created the row, which marks the settlement run that produced "
                    "it, UTC"
                ),
            }
            if settlement
            else {
                "acceptance_number": "Acceptance identifier",
                "acceptance_time": "When the acceptance was issued, UTC",
                "settlement_date": "Elexon settlement date of `time_from`",
                "settlement_period_from": "Settlement period at the start of the ramp",
                "settlement_period_to": "Settlement period at the end of the ramp",
                "time_from": "Start of the ramp, UTC",
                "time_to": "End of the ramp, UTC",
                "level_from_mw": "Output level at `time_from`, MW",
                "level_to_mw": "Output level at `time_to`, MW",
                "deemed_bo_flag": "Deemed bid-offer acceptance",
                "so_flag": "System operator flag: the acceptance was for a transmission constraint",
                "amendment_flag": "Elexon amendment flag (ORI original, UPD updated, DEL deleted)",
                "stor_flag": "Short-term operating reserve flag",
                "rr_flag": "Replacement reserve flag",
            }
        ),
    }
    if name == "elexon_boav":
        columns |= {
            "acceptance_id": "Acceptance identifier",
            "acceptance_duration": "Elexon acceptance duration code (S short, L long)",
            "total_volume_accepted_mwh": "Total accepted volume in the period, MWh (bids negative)",
            "pair_volume_neg1..pos6": "Volume accepted in bid-offer pairs -1..-6 and 1..6, MWh",
        }
    if name == "elexon_ebocf":
        columns |= {
            "total_cashflow_gbp": (
                "Total indicative cashflow in the period, GBP. The table holds no price. A user "
                "of the table derives the price as `total_cashflow_gbp` divided by "
                "`total_volume_accepted_mwh` in `elexon_boav`"
            ),
            "pair_cashflow_neg1..pos6": "Cashflow in bid-offer pairs -1..-6 and 1..6, GBP",
        }
    row_lines = (
        f"- Rows written: {frame.height}, from {distinct_bmus} of the {bmu_count} BMUs requested. "
        "A BMU has a row only when it had an acceptance, so row counts are not expected to be "
        "complete grids.\n" + _gap_text(gaps=gaps)
    )
    write_readme(
        product_dir=output_dir,
        title=table["title"],
        source_page=table["page"],
        script_path=SCRIPT_PATH,
        attribution=ELEXON_ATTRIBUTION,
        cache_hint="_day_cache/batch_<hash>/",
        licence=ELEXON_LICENCE,
        timestamp_convention=timestamp,
        columns=columns,
        row_summary=row_lines + f"\n- Window: {start} to {end}.",
        gotchas=[
            (
                "The BMU list is `elexon_bmu_reference/bmu_candidates.csv`, or the copy passed "
                "with `--bmu-list`. `lineage.json` lists the BMUs requested. Each "
                "batch of BMUs is cached separately, so removing a BMU refetches one batch."
            ),
            (
                "Elexon revises settlement values. The table holds what the API returned on the "
                "retrieval date in `lineage.json`."
            ),
        ],
    )


def main() -> None:
    """Run the reference phase or the dispatch phase."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--phase", choices=("reference", "dispatch"), required=True)
    parser.add_argument("--output-root", type=Path, default=MARKET_DOWNLOADS_DIR)
    parser.add_argument("--overwrite-candidates", action="store_true")
    parser.add_argument("--bmu-list", type=Path, default=None)
    parser.add_argument("--start", type=date.fromisoformat, default=WINDOW_START)
    parser.add_argument("--end", type=date.fromisoformat, default=WINDOW_END)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("elexon_boav", "elexon_ebocf", "elexon_boalf"),
        default=["elexon_boav", "elexon_ebocf", "elexon_boalf"],
    )
    parser.add_argument("--threads", type=int, default=FETCH_THREADS)
    args = parser.parse_args()
    if args.phase == "reference":
        run_reference_phase(root=args.output_root, overwrite_candidates=args.overwrite_candidates)
    else:
        run_dispatch_phase(
            root=args.output_root,
            bmu_list=args.bmu_list
            or args.output_root / "elexon_bmu_reference" / "bmu_candidates.csv",
            start=args.start,
            end=args.end,
            sources=args.sources,
            threads=args.threads,
        )
    print(f"Finished at {datetime.now(UTC):%Y-%m-%d %H:%M} UTC")


if __name__ == "__main__":
    main()
