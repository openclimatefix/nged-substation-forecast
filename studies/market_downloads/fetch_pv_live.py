r"""Download Sheffield Solar's PV_Live solar generation for NGED's four licence areas.

Written for the study of Elexon's indicated generation and demand. PV_Live estimates the solar
generation of each distribution licence area every half-hour. Run it for 2025-09-01 to 2026-09-30,
or pass `--start`, `--end` (both inclusive), and `--output-root` for a small test run:

    uv run python studies/market_downloads/fetch_pv_live.py

Each (area, month) is one request, written to `_month_cache/` the moment it arrives, so a crash
loses at most one request and a re-run fetches only what is missing. PV_Live refuses a request
that spans more than 366 days.
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
    WINDOW_END,
    WINDOW_START,
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

SCRIPT_PATH: Final[str] = "studies/market_downloads/fetch_pv_live.py"
SOURCE_PAGE: Final[str] = "https://www.solar.sheffield.ac.uk/pvlive/"
API_URL: Final[str] = "https://api.pvlive.uk/pvlive/api/v4/pes/{pes_id}"
PES_LIST_URL: Final[str] = "https://api.pvlive.uk/pvlive/api/v4/pes_list"
NGED_AREAS: Final[dict[int, str]] = {
    11: "_B",
    14: "_E",
    21: "_K",
    22: "_L",
}
"""NGED's four licence areas as PV_Live numbers them, mapped to the GSP group letter.

The numbers do not follow the letters: PV_Live's 19 and 20 are `_J` and `_H`, in southern England.
`check_pes_list` compares this table with PV_Live's own `pes_list` before each run.
"""
HALF_HOUR: Final[timedelta] = timedelta(minutes=30)
REQUEST_TIME: Final[str] = "%Y-%m-%dT%H:%M:%S"
PURPOSE: Final[str] = (
    "Public data for the study of Elexon's indicated generation and demand "
    "(<https://github.com/openclimatefix/nged-substation-forecast/issues/1108>)."
)
LICENCE: Final[str] = (
    "The maintainer accepted PV_Live's published terms for this download. The terms and any "
    "attribution wording have not been independently verified."
)
SCHEMA: Final[dict[str, Any]] = {
    "time": UTC_TIME,
    "pes_id": pl.Int16,
    "gsp_group": pl.String,
    "generation_mw": pl.Float64,
    "installed_capacity_mwp": pl.Float64,
    "updated_at": UTC_TIME,
}
COLUMN_DOCS: Final[dict[str, str]] = {
    "time": "UTC start of the half-hour. PV_Live labels each half-hour by its end, so this "
    "column is PV_Live's `datetime_gmt` minus 30 minutes.",
    "pes_id": "PV_Live's number for the distribution licence area.",
    "gsp_group": "The GSP group letter of the area, from PV_Live's `pes_list`.",
    "generation_mw": "Estimated solar generation of the area, MW, averaged over the half-hour.",
    "installed_capacity_mwp": "Installed capacity PV_Live assumed for the area, MWp.",
    "updated_at": "When PV_Live last updated this estimate (UTC).",
}


def check_pes_list() -> None:
    """Raise if PV_Live's own list does not give each area in `NGED_AREAS` the same GSP letter."""
    body = get_json(url=PES_LIST_URL)
    letters = {pes_id: letter for pes_id, letter, _ in body["data"]}
    wrong = {
        pes_id: letters.get(pes_id)
        for pes_id, letter in NGED_AREAS.items()
        if letters.get(pes_id) != letter
    }
    if wrong:
        raise ValueError(f"NGED_AREAS disagrees with PV_Live's pes_list: {wrong}")


def parse_pes_rows(*, body: object, pes_id: int) -> pl.DataFrame:
    """Turn one PV_Live response into the tidy table, shifting each label back to the start."""
    if not isinstance(body, dict) or "data" not in body:
        raise TypeError(f"PES {pes_id}: expected a PV_Live response, got {str(body)[:200]!r}")
    frame = pl.DataFrame(
        body["data"],
        schema=dict.fromkeys(body["meta"], pl.String),
        orient="row",
    )
    return frame.select(
        time=pl.col("datetime_gmt").str.to_datetime(
            format="%Y-%m-%dT%H:%M:%SZ", time_zone="UTC", time_unit="us"
        )
        - HALF_HOUR,
        pes_id=pl.col("pes_id").cast(pl.Int16),
        gsp_group=pl.lit(NGED_AREAS[pes_id]),
        generation_mw=pl.col("generation_mw").cast(pl.Float64),
        installed_capacity_mwp=pl.col("installedcapacity_mwp").cast(pl.Float64),
        updated_at=pl.col("updated_gmt").str.to_datetime(
            format="%Y-%m-%dT%H:%M:%SZ", time_zone="UTC", time_unit="us"
        ),
    ).sort("time")


def fetch_chunk(key: str) -> pl.DataFrame:
    """Fetch one `<pes_id>_<first>_<day after last>` chunk and keep the half-hours it names."""
    pes_text, first_text, after_text = key.split("_")
    pes_id = int(pes_text)
    first, after = date.fromisoformat(first_text), date.fromisoformat(after_text)
    # PV_Live labels a half-hour by its end, so the half-hours starting in [first, after) carry
    # labels from first + 30 minutes to after, inclusive.
    window_start = datetime(first.year, first.month, first.day, tzinfo=UTC)
    window_end = datetime(after.year, after.month, after.day, tzinfo=UTC)
    body = get_json(
        url=API_URL.format(pes_id=pes_id),
        params=[
            ("start", f"{window_start + HALF_HOUR:{REQUEST_TIME}}"),
            ("end", f"{window_end:{REQUEST_TIME}}"),
            ("extra_fields", "installedcapacity_mwp,updated_gmt"),
            ("data_format", "json"),
        ],
    )
    frame = parse_pes_rows(body=body, pes_id=pes_id)
    frame = frame.filter(pl.col("time").is_between(window_start, window_end, closed="left"))
    expected = int((window_end - window_start) / HALF_HOUR)
    if frame["time"].n_unique() < expected:
        raise IncompleteChunkError(
            f"PES {pes_id} chunk {key} has {frame.height} of {expected} rows"
        )
    return frame


def chunk_keys(*, start: date, end: date) -> list[str]:
    """Return one key per area and calendar month of `start..end`."""
    return [
        f"{pes_id}_{month}" for pes_id in NGED_AREAS for month in month_chunks(start=start, end=end)
    ]


def run(*, root: Path, start: date, end: date, threads: int) -> pl.DataFrame:
    """Download the four areas, write the parquet, the lineage note, and the README."""
    check_pes_list()
    output_dir = root / "pv_live"
    cache_dir = output_dir / "_month_cache"
    keys = chunk_keys(start=start, end=end)
    outcome = fetch_missing_chunks(
        cache_dir=cache_dir, keys=keys, fetch=fetch_chunk, label="pv_live", threads=threads
    )
    frame = read_chunks(cache_dir=cache_dir, keys=keys, schema=SCHEMA).sort("pes_id", "time")
    write_parquet_atomic(frame=frame, path=output_dir / "pv_live.parquet")
    expected = expected_half_hour_starts(start=start, end=end)
    checks = {
        letter: summarise_gaps(
            expected=expected,
            actual=frame.filter(pl.col("gsp_group") == letter)["time"],
        )
        for letter in NGED_AREAS.values()
    }
    write_lineage(
        product_dir=output_dir,
        note={
            "source_address": API_URL,
            "window": [start.isoformat(), end.isoformat()],
            "script": SCRIPT_PATH,
            "areas": NGED_AREAS,
            "chunks_not_published": sorted(outcome["not_published"]),
            "rows": frame.height,
            "null_generation_mw": frame["generation_mw"].null_count(),
            "latest_update": frame["updated_at"].max(),
            "checks": checks,
        },
    )
    write_readme(
        product_dir=output_dir,
        title="Sheffield Solar PV_Live solar generation, NGED's four licence areas",
        source_page=SOURCE_PAGE,
        script_path=SCRIPT_PATH,
        attribution=None,
        cache_hint="_month_cache/",
        licence=LICENCE,
        timestamp_convention=COLUMN_DOCS["time"],
        columns=COLUMN_DOCS,
        row_summary="\n".join(
            f"- {letter}: {gaps['distinct_rows']} of {gaps['expected_rows']} half-hours, "
            f"{gaps['missing_count']} missing."
            for letter, gaps in checks.items()
        ),
        gotchas=[
            (
                "PV_Live revises its estimates, so a re-download can change values; "
                "`updated_at` says when each value was last revised."
            ),
            (
                "PV_Live numbers the areas in a different order from the GSP group letters, so "
                "always join on `gsp_group`."
            ),
        ],
        purpose=PURPOSE,
    )
    print(f"pv_live: wrote {frame.height} rows")
    return frame


def main() -> None:
    """Parse the command line and run the download."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--output-root", type=Path, default=MARKET_DOWNLOADS_DIR)
    parser.add_argument("--start", type=date.fromisoformat, default=WINDOW_START)
    parser.add_argument("--end", type=date.fromisoformat, default=WINDOW_END)
    parser.add_argument("--threads", type=int, default=FETCH_THREADS)
    args = parser.parse_args()
    run(root=args.output_root, start=args.start, end=args.end, threads=args.threads)
    print(f"Finished at {datetime.now(UTC):%Y-%m-%d %H:%M} UTC")


if __name__ == "__main__":
    main()
