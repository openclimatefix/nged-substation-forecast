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
from typing import Any, Final

import polars as pl
from fetch_system_series import UTC_TIME, Col, frame_from_rows, schema_of
from market_common import (
    ELEXON_API,
    ELEXON_ATTRIBUTION,
    ELEXON_LICENCE,
    FETCH_THREADS,
    SETTLEMENT_PERIOD,
    IncompleteChunkError,
    fetch_missing_chunks,
    get_json,
    period_start_utc,
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
COLUMNS: Final[tuple[Col, ...]] = (
    Col("settlementDate", "settlement_date", pl.Date, "Settlement date (UK clock-time day)."),
    Col(
        "settlementPeriod",
        "settlement_period",
        pl.Int16,
        "Settlement period within the settlement date.",
    ),
    Col(
        "nationalGridBmUnit",
        "national_grid_bmu_id",
        pl.String,
        "National Grid BMU identifier (`nationalGridBmUnit`). Never null, so it is the key.",
    ),
    Col(
        "bmUnit",
        "bmu_id",
        pl.String,
        (
            "Elexon BMU identifier (`bmUnit`). Null for BMUs Elexon gives no Elexon identifier. "
            "Supplier base BMUs start with `2__`."
        ),
    ),
    Col("timeFrom", "time_from", UTC_TIME, "UTC start of the PN segment."),
    Col("timeTo", "time_to", UTC_TIME, "UTC end of the PN segment."),
    Col(
        "levelFrom",
        "level_from_mw",
        pl.Float64,
        "Notified level at the segment start, MW. Negative is import.",
    ),
    Col(
        "levelTo",
        "level_to_mw",
        pl.Float64,
        "Notified level at the segment end, MW. The level changes linearly within one segment.",
    ),
)
SCHEMA: Final[dict[str, Any]] = schema_of(columns=COLUMNS)
COLUMN_DOCS: Final[dict[str, str]] = {col.name: col.doc for col in COLUMNS}


def untiled_bmus(*, frame: pl.DataFrame, settlement_date: date, period: int) -> int:
    """Count the BMUs whose segments do not tile the settlement period exactly."""
    start = period_start_utc(settlement_date=settlement_date, settlement_period=period)
    bad = (
        frame.sort("national_grid_bmu_id", "time_from")
        .group_by("national_grid_bmu_id")
        .agg(
            first=pl.col("time_from").min(),
            last=pl.col("time_to").max(),
            seams=(pl.col("time_from").shift(-1) == pl.col("time_to")).drop_nulls().all(),
            wrong_period=(
                (pl.col("settlement_date") != settlement_date)
                | (pl.col("settlement_period") != period)
            ).any(),
        )
        .filter(
            (pl.col("first") != start)
            | (pl.col("last") != start + SETTLEMENT_PERIOD)
            | ~pl.col("seams")
            | pl.col("wrong_period")
        )
    )
    return bad.height


def fetch_period(*, settlement_date: date, period: int) -> pl.DataFrame:
    """Fetch the PN segments of every BMU for one settlement period.

    Args:
        settlement_date: The settlement date.
        period: The settlement period within the date.

    Returns:
        One row for each PN segment of each BMU in the period.

    Raises:
        TypeError: If the response is not an object holding a list of rows.
        ValueError: If any BMU's segments do not tile the settlement period.
    """
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
    frame = frame_from_rows(
        rows=body["data"], columns=COLUMNS, what=f"PN {settlement_date} P{period}"
    )
    untiled = untiled_bmus(frame=frame, settlement_date=settlement_date, period=period)
    if untiled and not frame.is_empty():
        raise ValueError(f"PN {settlement_date} P{period}: {untiled} BMUs do not tile the period")
    return frame


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


def day_summaries(*, frame: pl.DataFrame) -> dict[str, dict[str, int]]:
    """Measure each sampled day: rows, periods against the expected count, and BMUs per period."""
    per_period = frame.group_by("settlement_date", "settlement_period").agg(
        bmus=pl.col("national_grid_bmu_id").n_unique()
    )
    per_day = (
        frame.group_by("settlement_date")
        .agg(
            rows=pl.len(),
            periods=pl.col("settlement_period").n_unique(),
            bmus_without_elexon_id=pl.col("national_grid_bmu_id")
            .filter(pl.col("bmu_id").is_null())
            .n_unique(),
        )
        .join(
            per_period.group_by("settlement_date").agg(
                bmus_min_per_period=pl.col("bmus").min(), bmus_max_per_period=pl.col("bmus").max()
            ),
            on="settlement_date",
        )
        .sort("settlement_date")
    )
    return {
        str(row["settlement_date"]): {
            **{key: value for key, value in row.items() if key != "settlement_date"},
            "periods_expected": periods_in_settlement_day(settlement_date=row["settlement_date"]),
        }
        for row in per_day.iter_rows(named=True)
    }


def run(*, root: Path, days: list[date], threads: int) -> pl.DataFrame:
    """Download the days, then write the parquet, the lineage note, and the README."""
    output_dir = root / "elexon_pn_sample"
    cache_dir = output_dir / "_day_cache"
    keys = [day.isoformat() for day in days]
    outcome = fetch_missing_chunks(
        cache_dir=cache_dir, keys=keys, fetch=fetch_day, label="pn_sample", threads=threads
    )
    frame = read_chunks(cache_dir=cache_dir, keys=keys, schema=SCHEMA).sort(
        "settlement_date", "settlement_period", "national_grid_bmu_id", "time_from"
    )
    write_parquet_atomic(frame=frame, path=output_dir / "elexon_pn_sample.parquet")
    per_day = day_summaries(frame=frame)
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
        title=f"Elexon final Physical Notifications of every BMU, {len(keys)} sample days",
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
            f"- {day}: {counts['rows']} rows, {counts['periods']} of {counts['periods_expected']} "
            f"periods, {counts['bmus_min_per_period']} to {counts['bmus_max_per_period']} BMUs "
            f"in a period, {counts['bmus_without_elexon_id']} of them without an Elexon "
            "identifier."
            for day, counts in per_day.items()
        ),
        gotchas=[
            (
                "The values are the PNs as Elexon serves them now, which are the final PNs. The "
                "PNs in force when an INDDEM or INDGEN issue was published can differ."
            ),
            (
                "Some BMUs have no Elexon identifier (`bmu_id` is null), so key on "
                "`national_grid_bmu_id`. The count is in the row summary above."
            ),
            (
                "A BMU's level can jump where one segment meets the next, at period boundaries "
                "especially, so average each segment on its own and never interpolate across "
                "segments."
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
