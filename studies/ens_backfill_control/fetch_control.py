"""Fetch the ECMWF ENS control member for every complete date of Dynamical.org's staged archive.

The staged archive on Source Cooperative holds 00 UTC ENS runs from 2021-03-21 to 2024-03-31. This
script fetches the control member's 12 variables at 85 steps (1,020 messages, about 485 MB over the
network) for each complete date in a date range, crops them to the box around Great Britain, and
writes one checkpoint file per date under `<output folder>/control/`. It reuses the pilot's fetch
code, `studies.ens_backfill_fetch`.

Dates run in two tranches, the second validation year (2023-07-01 onwards) first, then the earlier
dates. A date already written is skipped, so a crashed or stopped run resumes where it left off.
`--shard` and `--shard-count` split the dates between processes that run side by side.
"""

import argparse
import json
import os
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

from studies.ens_backfill_fetch import (
    MAX_WORKERS,
    FetchError,
    RequestStats,
    RetriesExhaustedError,
    fetch_planned,
    list_bucket,
    load_plan,
    prefix_bytes,
    shard_dates,
    write_checkpoint,
)
from studies.ens_grib_source import complete_dates, parse_listing

DEFAULT_OUTPUT_DIR: Final[Path] = Path("/mnt/wd_18tb/data/nwp-archive/ECMWF-ENS-BACKFILL-CONTROL")
"""The destination on the 18 TB USB disk, beside the other products of the weather archive."""

FIRST_DATE: Final[date] = date(2021, 3, 21)
"""The first date of the solidly complete part of the archive."""

LAST_DATE: Final[date] = date(2024, 3, 31)
"""The last staged date, the day before Dynamical.org's own archive begins."""

FIRST_TRANCHE_START: Final[date] = date(2023, 7, 1)
"""The start of the second validation year, which is fetched before the earlier dates."""

MAX_CONSECUTIVE_FAILURES: Final[int] = 5
"""A run stops after this many dates in a row whose retries all failed, a sign of an outage."""


def stamp() -> str:
    """Return the current UTC time for a log line."""
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%SZ")


def listed_dates(*, output_dir: Path) -> list[date]:
    """Return the archive's complete dates, from the saved listing or a fresh bucket listing.

    The first process to list the bucket saves the result, and every other process and every
    resumed run reads that file, so that all of them agree on the dates.
    """
    saved = output_dir / "complete_dates.json"
    if not saved.exists():
        complete = complete_dates(files_by_date=parse_listing(text=list_bucket()))
        output_dir.mkdir(parents=True, exist_ok=True)
        partial = saved.with_name(f"complete_dates.json.{os.getpid()}.partial")
        partial.write_text(
            json.dumps(
                {
                    "listed_at_utc": datetime.now(UTC).isoformat(),
                    "dates": [day.isoformat() for day in complete],
                },
                indent=2,
            )
        )
        partial.rename(saved)
    return [date.fromisoformat(text) for text in json.loads(saved.read_text())["dates"]]


def order_dates(*, complete: list[date], first: date, last: date) -> list[date]:
    """Return the complete dates from `first` to `last`, the second validation year first.

    Each tranche runs oldest first.
    """
    in_range = sorted(day for day in complete if first <= day <= last)
    return [day for day in in_range if day >= FIRST_TRANCHE_START] + [
        day for day in in_range if day < FIRST_TRANCHE_START
    ]


def fetch_date(*, day: date, output_dir: Path, workers: int, dry_run: bool) -> bool | None:
    """Fetch one date, write its checkpoint, and log a summary line.

    Returns:
        True on success, False on a failure that points at the server or network (every retry
        failed), and None on a failure confined to this date's data, such as a bad sidecar.
    """
    out_path = output_dir / "control" / f"{day.isoformat()}.npz"
    if out_path.exists():
        print(f"{stamp()} {day}: already written, skipping", flush=True)
        return True
    started = time.monotonic()
    stats = RequestStats()
    try:
        planned = load_plan(day=day, members="control", stats=stats)
        if dry_run:
            print(
                f"{stamp()} {day}: {len(planned)} messages of {prefix_bytes()} bytes = "
                f"{len(planned) * prefix_bytes() / 1e6:.0f} MB",
                flush=True,
            )
            return True
        arrays = fetch_planned(
            day=day, planned=planned, members="control", workers=workers, stats=stats
        )
        write_checkpoint(arrays=arrays, out_path=out_path)
    except (FetchError, ValueError) as error:
        print(f"{stamp()} {day}: FAILED after {stats.requests} requests: {error}", flush=True)
        return False if isinstance(error, RetriesExhaustedError) else None
    seconds = time.monotonic() - started
    print(
        f"{stamp()} {day}: messages={len(planned)} requests={stats.requests} "
        f"MB={stats.bytes / 1e6:.1f} seconds={seconds:.1f} retries={stats.retries} "
        f"extended_prefixes={stats.extended_prefixes} MB/s={stats.bytes / 1e6 / seconds:.2f}",
        flush=True,
    )
    return True


def main() -> int:
    """Parse the command line and fetch every requested date."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start", type=date.fromisoformat, default=FIRST_DATE)
    parser.add_argument("--end", type=date.fromisoformat, default=LAST_DATE)
    parser.add_argument(
        "--max-dates", type=int, help="stop after this many dates, for a timing run"
    )
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--workers", type=int, default=12, help=f"concurrent connections, at most {MAX_WORKERS}"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="fetch the sidecars and print the plan"
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be between 1 and {MAX_WORKERS}")
    if not 0 <= args.shard < args.shard_count:
        parser.error("--shard must be in 0 to --shard-count minus 1")

    ordered = order_dates(
        complete=listed_dates(output_dir=args.output_dir), first=args.start, last=args.end
    )
    dates = shard_dates(dates=ordered, shard=args.shard, shard_count=args.shard_count)
    if args.max_dates is not None:
        dates = dates[: args.max_dates]
    calendar_days = (args.end - args.start).days + 1
    print(
        f"{stamp()} {calendar_days - len(ordered)} calendar days in range are not complete in "
        f"the archive listing",
        flush=True,
    )
    print(
        f"{stamp()} shard {args.shard}/{args.shard_count}: {len(dates)} of {len(ordered)} dates, "
        f"{args.workers} connections, output {args.output_dir}",
        flush=True,
    )
    started = time.monotonic()
    failed: list[date] = []
    consecutive_failures = 0
    for day in dates:
        outcome = fetch_date(
            day=day, output_dir=args.output_dir, workers=args.workers, dry_run=args.dry_run
        )
        if outcome is not False:
            consecutive_failures = 0
        if outcome is None or outcome is False:
            failed.append(day)
        if outcome is False:
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(
                    f"{stamp()} Stopping after {consecutive_failures} failures in a row", flush=True
                )
                break
    print(
        f"{stamp()} done in {time.monotonic() - started:.0f} s; "
        f"{len(failed)} failed dates: {[day.isoformat() for day in failed]}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
