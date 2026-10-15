r"""Download Sheffield Solar's PV_Live solar generation for the two grid supply points in our data.

PV_Live estimates the solar generation behind each grid supply point (GSP) every half-hour. The
two GSPs are Walpole (PV_Live's `WALP_B`) and Bicker Fen (PV_Live's `BICF_1`). Run it for
2019-01-01 to yesterday, or pass `--start`, `--end` (both inclusive), and `--output-root` for a
small test run:

    uv run python studies/market_downloads/fetch_pv_live_gsp.py

Each (GSP, month) is one request, written to `_month_cache/` the moment it arrives, so a crash
loses at most one request and a re-run fetches only what is missing.
"""

import argparse
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import polars as pl
from fetch_bmu_dispatch import month_chunks
from fetch_system_series import UTC_TIME
from market_common import (
    FETCH_THREADS,
    IncompleteChunkError,
    expected_half_hour_starts,
    fetch_missing_chunks,
    get_json,
    read_chunks,
    summarise_gaps,
    write_lineage,
    write_parquet_atomic,
    write_readme,
)
from studies.sources import MARKET_DOWNLOADS_DIR

SCRIPT_PATH: Final[str] = "studies/market_downloads/fetch_pv_live_gsp.py"
SOURCE_PAGE: Final[str] = "https://www.solar.sheffield.ac.uk/pvlive/"
API_URL: Final[str] = "https://api.pvlive.uk/pvlive/api/v4/gsp/{gsp_id}"
GSP_LIST_URL: Final[str] = "https://api.pvlive.uk/pvlive/api/v4/gsp_list"
OUR_GSPS: Final[dict[int, str]] = {298: "WALP_B", 24: "BICF_1"}
"""The two GSPs in the NGED metadata table, as PV_Live numbers them, mapped to PV_Live's name.

298 is Walpole and 24 is Bicker Fen. `check_gsp_list` compares this table with PV_Live's own
`gsp_list` before each run.
"""
FIRST_DAY: Final[date] = date(2019, 1, 1)
HALF_HOUR: Final[timedelta] = timedelta(minutes=30)
REQUEST_TIME: Final[str] = "%Y-%m-%dT%H:%M:%S"
PURPOSE: Final[str] = (
    "Public data for ml-experiments that use the GSP's solar estimate as a feature for the "
    "Primary and BSP demand forecasts."
)
ATTRIBUTION: Final[str] = (
    "Sheffield Solar PV_Live (<https://www.solar.sheffield.ac.uk/pvlive/>), licensed under "
    "CC BY 4.0."
)
LICENCE: Final[str] = (
    "PV_Live's web page states that its data are licensed under Creative Commons Attribution 4.0 "
    "(CC BY 4.0), credited to Sheffield Solar. The API needs no key or payment."
)
SCHEMA: Final[dict[str, Any]] = {
    "time": UTC_TIME,
    "gsp_id": pl.Int16,
    "gsp_name": pl.String,
    "generation_mw": pl.Float64,
    "installed_capacity_mwp": pl.Float64,
    "capacity_mwp": pl.Float64,
    "generation_percent_of_capacity": pl.Float64,
    "updated_at": UTC_TIME,
}
COLUMN_DOCS: Final[dict[str, str]] = {
    "time": "UTC start of the half-hour. PV_Live labels each half-hour by its end, so this "
    "column is PV_Live's `datetime_gmt` minus 30 minutes.",
    "gsp_id": "PV_Live's number for the GSP.",
    "gsp_name": "PV_Live's name for the GSP (`WALP_B` is Walpole, `BICF_1` is Bicker Fen).",
    "generation_mw": "Estimated solar generation behind the GSP, MW, averaged over the half-hour.",
    "installed_capacity_mwp": "Installed PV capacity PV_Live assumed (DC), MWp.",
    "capacity_mwp": "PV_Live's `capacity_mwp`, the capacity it normalises against, MWp.",
    "generation_percent_of_capacity": "`100 * generation_mw / capacity_mwp`, computed by this "
    "script because the API has no percent column; null where `capacity_mwp` is zero or null.",
    "updated_at": "When PV_Live last updated this estimate (UTC).",
}


def check_gsp_list() -> None:
    """Raise if PV_Live's own list does not give each ID in `OUR_GSPS` the same name."""
    body = get_json(url=GSP_LIST_URL)
    names = {gsp_id: name for gsp_id, name, _ in body["data"]}
    wrong = {i: names.get(i) for i, name in OUR_GSPS.items() if names.get(i) != name}
    if wrong:
        raise ValueError(f"OUR_GSPS disagrees with PV_Live's gsp_list: {wrong}")


def parse_gsp_rows(*, body: object, gsp_id: int) -> pl.DataFrame:
    """Turn one PV_Live response into the tidy table, shifting each label back to the start."""
    if not isinstance(body, dict) or "data" not in body:
        raise TypeError(f"GSP {gsp_id}: expected a PV_Live response, got {str(body)[:200]!r}")
    frame = pl.DataFrame(body["data"], schema=dict.fromkeys(body["meta"], pl.String), orient="row")
    if frame.is_empty():
        return pl.DataFrame(schema=SCHEMA)
    capacity = pl.col("capacity_mwp").cast(pl.Float64)
    generation = pl.col("generation_mw").cast(pl.Float64)
    return frame.select(
        time=pl.col("datetime_gmt").str.to_datetime(
            format="%Y-%m-%dT%H:%M:%SZ", time_zone="UTC", time_unit="us"
        )
        - HALF_HOUR,
        gsp_id=pl.col("gsp_id").cast(pl.Int16),
        gsp_name=pl.lit(OUR_GSPS[gsp_id]),
        generation_mw=generation,
        installed_capacity_mwp=pl.col("installedcapacity_mwp").cast(pl.Float64),
        capacity_mwp=capacity,
        generation_percent_of_capacity=pl.when(capacity > 0)
        .then(100 * generation / capacity)
        .otherwise(None),
        updated_at=pl.col("updated_gmt").str.to_datetime(
            format="%Y-%m-%dT%H:%M:%SZ", time_zone="UTC", time_unit="us"
        ),
    ).sort("time")


def fetch_chunk(key: str) -> pl.DataFrame:
    """Fetch one `<gsp_id>_<first>_<day after last>` chunk and keep the half-hours it names."""
    gsp_text, first_text, after_text = key.split("_")
    gsp_id = int(gsp_text)
    first, after = date.fromisoformat(first_text), date.fromisoformat(after_text)
    window_start = datetime(first.year, first.month, first.day, tzinfo=UTC)
    window_end = datetime(after.year, after.month, after.day, tzinfo=UTC)
    # PV_Live labels a half-hour by its end, so half-hours starting in [first, after) carry labels
    # from first + 30 minutes to after, inclusive.
    body = get_json(
        url=API_URL.format(gsp_id=gsp_id),
        params=[
            ("start", f"{window_start + HALF_HOUR:{REQUEST_TIME}}"),
            ("end", f"{window_end:{REQUEST_TIME}}"),
            ("extra_fields", "installedcapacity_mwp,capacity_mwp,updated_gmt"),
            ("data_format", "json"),
        ],
    )
    frame = parse_gsp_rows(body=body, gsp_id=gsp_id)
    frame = frame.filter(pl.col("time").is_between(window_start, window_end, closed="left"))
    expected = int((window_end - window_start) / HALF_HOUR)
    if frame["time"].n_unique() < expected:
        raise IncompleteChunkError(
            f"GSP {gsp_id} chunk {key} has {frame.height} of {expected} rows"
        )
    return frame


def chunk_keys(*, start: date, end: date) -> list[str]:
    """Return one key per GSP and calendar month of `start..end`, GSP by GSP."""
    return [
        f"{gsp_id}_{month}" for gsp_id in OUR_GSPS for month in month_chunks(start=start, end=end)
    ]


def run(*, root: Path, start: date, end: date, threads: int) -> pl.DataFrame:
    """Download the two GSPs, write the parquet, the lineage note, and the README."""
    check_gsp_list()
    output_dir = root / "pv_live_gsp"
    cache_dir = output_dir / "_month_cache"
    keys = chunk_keys(start=start, end=end)
    outcome = fetch_missing_chunks(
        cache_dir=cache_dir, keys=keys, fetch=fetch_chunk, label="pv_live_gsp", threads=threads
    )
    frame = read_chunks(cache_dir=cache_dir, keys=keys, schema=SCHEMA).sort("gsp_id", "time")
    write_parquet_atomic(frame=frame, path=output_dir / "pv_live_gsp.parquet")
    expected = expected_half_hour_starts(start=start, end=end)
    checks = {
        name: summarise_gaps(
            expected=expected, actual=frame.filter(pl.col("gsp_name") == name)["time"]
        )
        for name in OUR_GSPS.values()
    }
    write_lineage(
        product_dir=output_dir,
        note={
            "source_address": API_URL,
            "window": [start.isoformat(), end.isoformat()],
            "script": SCRIPT_PATH,
            "gsps": OUR_GSPS,
            "chunks_not_published": sorted(outcome["not_published"]),
            "rows": frame.height,
            "null_counts": dict(zip(frame.columns, frame.null_count().row(0), strict=True)),
            "latest_update": frame["updated_at"].max(),
            "checks": checks,
            "attribution": ATTRIBUTION,
        },
    )
    write_readme(
        product_dir=output_dir,
        title="Sheffield Solar PV_Live solar generation, Walpole and Bicker Fen GSPs",
        source_page=SOURCE_PAGE,
        script_path=SCRIPT_PATH,
        attribution=ATTRIBUTION,
        cache_hint="_month_cache/",
        licence=LICENCE,
        timestamp_convention=COLUMN_DOCS["time"],
        columns=COLUMN_DOCS,
        row_summary="\n".join(
            f"- {name}: {gaps['distinct_rows']} of {gaps['expected_rows']} half-hours, "
            f"{gaps['missing_count']} missing."
            for name, gaps in checks.items()
        ),
        gotchas=[
            (
                "PV_Live revises its estimates, so a re-download can change values; "
                "`updated_at` says when each value was last revised."
            ),
            "Join on `gsp_id` or `gsp_name`, never on a position in the file.",
        ],
        purpose=PURPOSE,
    )
    print(f"pv_live_gsp: wrote {frame.height} rows")
    return frame


def main() -> None:
    """Parse the command line and run the download."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--output-root", type=Path, default=MARKET_DOWNLOADS_DIR)
    parser.add_argument("--start", type=date.fromisoformat, default=FIRST_DAY)
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=datetime.now(UTC).date() - timedelta(days=1),
    )
    parser.add_argument("--threads", type=int, default=min(FETCH_THREADS, 2))
    args = parser.parse_args()
    run(root=args.output_root, start=args.start, end=args.end, threads=args.threads)
    print(f"Finished at {datetime.now(UTC):%Y-%m-%d %H:%M} UTC")


if __name__ == "__main__":
    main()
