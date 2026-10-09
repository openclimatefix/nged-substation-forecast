r"""Download the final Physical Notifications (PNs) of every BMU, for a few chosen days.

Written for the study of Elexon's indicated generation and demand, which sums the PNs of every
Balancing Mechanism Unit (BMU). Summing the PNs by the GSP group of each BMU shows which GSP
groups sit inside which of INDDEM's zones. One request returns the PN segments of every BMU for one
settlement period, about 2,500 rows, so a day takes 48 requests. The default days are one weekday
in each season:

    uv run python studies/market_downloads/fetch_pn_all_bmus_sample.py
    uv run python studies/market_downloads/fetch_pn_all_bmus_sample.py --days 2026-03-04

Each day is one chunk under `_day_cache/`, written the moment it arrives, so a crash loses at most
one day and a re-run fetches only the missing days.
"""

import argparse
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import polars as pl
from fetch_system_series import UTC_TIME
from market_common import (
    ELEXON_API,
    ELEXON_ATTRIBUTION,
    ELEXON_LICENCE,
    FETCH_THREADS,
    IncompleteChunkError,
    fetch_missing_chunks,
    get_json,
    periods_in_settlement_day,
    read_chunks,
    write_lineage,
    write_parquet_atomic,
    write_readme,
)
from studies.sources import MARKET_DOWNLOADS_DIR

SCRIPT_PATH: Final[str] = "studies/market_downloads/fetch_pn_all_bmus_sample.py"
SOURCE_PAGE: Final[str] = "https://bmrs.elexon.co.uk/physical-notifications"
PN_URL: Final[str] = f"{ELEXON_API}/datasets/PN"
DEFAULT_DAYS: Final[tuple[date, ...]] = (
    date(2025, 12, 10),
    date(2026, 3, 4),
    date(2026, 6, 17),
    date(2026, 9, 16),
)
"""One Wednesday in each season, inside the study window."""
PURPOSE: Final[str] = (
    "Public data for the study of Elexon's indicated generation and demand "
    "(<https://github.com/openclimatefix/nged-substation-forecast/issues/1108>)."
)
SCHEMA: Final[dict[str, object]] = {
    "settlement_date": pl.Date,
    "settlement_period": pl.Int16,
    "bmu_id": pl.String,
    "time_from": UTC_TIME,
    "time_to": UTC_TIME,
    "level_from_mw": pl.Float64,
    "level_to_mw": pl.Float64,
}
COLUMN_DOCS: Final[dict[str, str]] = {
    "settlement_date": "Settlement date (UK clock-time day).",
    "settlement_period": "Settlement period within the settlement date.",
    "bmu_id": "Elexon BMU identifier (`bmUnit`). Supplier base BMUs start with `2__`.",
    "time_from": "UTC start of the PN segment.",
    "time_to": "UTC end of the PN segment.",
    "level_from_mw": "Notified level at the segment start, MW. Negative is import.",
    "level_to_mw": "Notified level at the segment end, MW. The level changes linearly between "
    "the two ends.",
}


def fetch_period(*, settlement_date: date, period: int) -> pl.DataFrame:
    """Fetch the PN segments of every BMU for one settlement period."""
    body = get_json(
        url=PN_URL,
        params=[
            ("settlementDate", settlement_date.isoformat()),
            ("settlementPeriod", str(period)),
            ("format", "json"),
        ],
    )
    if not isinstance(body, dict) or not isinstance(body.get("data"), list):
        raise TypeError(f"PN {settlement_date} P{period}: unexpected body {str(body)[:200]!r}")
    rows = body["data"]
    frame = pl.DataFrame(
        {
            "settlement_date": [row["settlementDate"] for row in rows],
            "settlement_period": [row["settlementPeriod"] for row in rows],
            "bmu_id": [row["bmUnit"] for row in rows],
            "time_from": [row["timeFrom"] for row in rows],
            "time_to": [row["timeTo"] for row in rows],
            "level_from_mw": [row["levelFrom"] for row in rows],
            "level_to_mw": [row["levelTo"] for row in rows],
        },
        schema={
            "settlement_date": pl.String,
            "settlement_period": pl.Int16,
            "bmu_id": pl.String,
            "time_from": pl.String,
            "time_to": pl.String,
            "level_from_mw": pl.Float64,
            "level_to_mw": pl.Float64,
        },
    )
    return frame.with_columns(
        settlement_date=pl.col("settlement_date").str.to_date(format="%Y-%m-%d"),
        time_from=pl.col("time_from").str.to_datetime(
            format="%Y-%m-%dT%H:%M:%SZ", time_zone="UTC", time_unit="us"
        ),
        time_to=pl.col("time_to").str.to_datetime(
            format="%Y-%m-%dT%H:%M:%SZ", time_zone="UTC", time_unit="us"
        ),
    )


def fetch_day(key: str) -> pl.DataFrame:
    """Fetch every settlement period of the settlement date named by `key`."""
    settlement_date = date.fromisoformat(key)
    periods = periods_in_settlement_day(settlement_date=settlement_date)
    frames = [
        fetch_period(settlement_date=settlement_date, period=period)
        for period in range(1, periods + 1)
    ]
    empty = [period for period, frame in enumerate(frames, start=1) if frame.is_empty()]
    if empty:
        raise IncompleteChunkError(f"PN {key}: periods {empty} have no rows")
    return pl.concat(frames)


def run(*, root: Path, days: list[date], threads: int) -> pl.DataFrame:
    """Download the days, then write the parquet, the lineage note, and the README."""
    output_dir = root / "elexon_pn_sample"
    cache_dir = output_dir / "_day_cache"
    keys = [day.isoformat() for day in days]
    outcome = fetch_missing_chunks(
        cache_dir=cache_dir, keys=keys, fetch=fetch_day, label="pn_sample", threads=threads
    )
    frame = read_chunks(cache_dir=cache_dir, keys=keys, schema=SCHEMA).sort(
        "settlement_date", "settlement_period", "bmu_id", "time_from"
    )
    write_parquet_atomic(frame=frame, path=output_dir / "elexon_pn_sample.parquet")
    per_day = dict(
        frame.group_by("settlement_date")
        .agg(
            rows=pl.len(),
            periods=pl.col("settlement_period").n_unique(),
            bmus=pl.col("bmu_id").n_unique(),
        )
        .sort("settlement_date")
        .select(pl.col("settlement_date").cast(pl.String), pl.struct("rows", "periods", "bmus"))
        .iter_rows()
    )
    write_lineage(
        product_dir=output_dir,
        note={
            "source_address": PN_URL,
            "days": keys,
            "script": SCRIPT_PATH,
            "rows": frame.height,
            "chunks_not_published": sorted(outcome["not_published"]),
            "per_day": per_day,
        },
    )
    write_readme(
        product_dir=output_dir,
        title="Elexon final Physical Notifications of every BMU, four sample days",
        source_page=SOURCE_PAGE,
        script_path=SCRIPT_PATH,
        attribution=ELEXON_ATTRIBUTION,
        cache_hint="_day_cache/",
        licence=ELEXON_LICENCE,
        timestamp_convention=(
            "`time_from` and `time_to` are UTC. The segments of one BMU tile the settlement "
            "period, with the level changing linearly between the two ends of a segment."
        ),
        columns=COLUMN_DOCS,
        row_summary="\n".join(
            f"- {day}: {counts['rows']} rows, {counts['periods']} periods, {counts['bmus']} BMUs."
            for day, counts in per_day.items()
        ),
        gotchas=[
            (
                "The values are the PNs as Elexon serves them now, which are the final PNs. The "
                "PNs in force when an INDDEM or INDGEN issue was published can differ."
            ),
        ],
        purpose=PURPOSE,
    )
    print(f"elexon_pn_sample: wrote {frame.height} rows")
    return frame


def main() -> None:
    """Parse the command line and run the download."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--output-root", type=Path, default=MARKET_DOWNLOADS_DIR)
    parser.add_argument("--days", nargs="+", type=date.fromisoformat, default=list(DEFAULT_DAYS))
    parser.add_argument("--threads", type=int, default=FETCH_THREADS)
    args = parser.parse_args()
    run(root=args.output_root, days=args.days, threads=args.threads)
    print(f"Finished at {datetime.now(UTC):%Y-%m-%d %H:%M} UTC")


if __name__ == "__main__":
    main()
