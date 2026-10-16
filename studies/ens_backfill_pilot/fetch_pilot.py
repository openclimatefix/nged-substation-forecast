"""Fetch a band of grid rows from Dynamical.org's staged ECMWF ENS GRIB files, for the pilot.

Run `uv run python studies/ens_backfill_pilot/fetch_pilot.py --dry-run` first. The pilot fetches
the control member's 12 variables at 85 steps (1,020 messages) for 23 dates, one range request
per message, and writes one checkpoint file per date under
`data/studies/per_study/ens_backfill_pilot/`. A date already written is skipped, so a crashed run
resumes where it stopped.
"""

import argparse
import json
import sys
import time
from datetime import UTC, date, datetime

from studies.ens_backfill_fetch import (
    ALWAYS_INCLUDED_DATES,
    DATE_SEED,
    MAX_WORKERS,
    PILOT_END,
    PILOT_START,
    PILOT_VARIABLES,
    RANDOM_DATE_COUNT,
    STEPS_HOURS,
    FetchError,
    MembersType,
    RequestStats,
    fetch_planned,
    list_bucket,
    load_plan,
    members_for,
    pilot_file_path,
    prefix_bytes,
    write_checkpoint,
)
from studies.ens_grib_source import choose_pilot_dates, complete_dates, parse_listing
from studies.sources import ENS_BACKFILL_PILOT_DIR


def resolve_dates(*, override: str | None) -> list[date]:
    """Return the dates to fetch: the `--dates` override, or the saved or freshly drawn pilot dates.

    The drawn dates are saved to `pilot_dates.json` and reused, because the bucket is still being
    filled and a fresh listing could change what the seeded draw picks. Delete the file to redraw.
    """
    if override:
        return [date.fromisoformat(text) for text in override.split(",")]
    saved = ENS_BACKFILL_PILOT_DIR / "pilot_dates.json"
    if saved.exists():
        return [date.fromisoformat(text) for text in json.loads(saved.read_text())["dates"]]
    complete = complete_dates(files_by_date=parse_listing(text=list_bucket()))
    chosen = choose_pilot_dates(
        complete=complete,
        first=PILOT_START,
        last=PILOT_END,
        sample_size=RANDOM_DATE_COUNT,
        seed=DATE_SEED,
        always=ALWAYS_INCLUDED_DATES,
    )
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_text(
        json.dumps(
            {
                "seed": DATE_SEED,
                "listed_at_utc": datetime.now(UTC).isoformat(),
                "complete_dates_in_bucket": len(complete),
                "dates": [day.isoformat() for day in chosen],
            },
            indent=2,
        )
    )
    return chosen


def fetch_date(*, day: date, members: MembersType, workers: int, dry_run: bool) -> bool:
    """Fetch one date, write its checkpoint, and print a summary line. Return False on failure."""
    out_path = pilot_file_path(members=members, day=day)
    if out_path.exists() and not dry_run:
        print(f"{day}: already written, skipping")
        return True
    started = time.monotonic()
    stats = RequestStats()
    try:
        planned = load_plan(day=day, members=members, stats=stats)
        if dry_run:
            print(
                f"{day}: {len(planned)} messages of {prefix_bytes()} bytes = "
                f"{len(planned) * prefix_bytes() / 1e6:.0f} MB, plus "
                f"{stats.requests} idx requests ({stats.bytes / 1e3:.0f} kB)"
            )
            return True
        arrays = fetch_planned(
            day=day, planned=planned, members=members, workers=workers, stats=stats
        )
    except (FetchError, ValueError) as error:
        print(f"{day}: FAILED after {stats.requests} requests: {error}", file=sys.stderr)
        return False
    write_checkpoint(arrays=arrays, out_path=out_path)
    seconds = time.monotonic() - started
    print(
        f"{day}: messages={len(planned)} requests={stats.requests} "
        f"MB={stats.bytes / 1e6:.1f} seconds={seconds:.1f} retries={stats.retries} "
        f"extended_prefixes={stats.extended_prefixes} "
        f"req/s={stats.requests / seconds:.1f} MB/s={stats.bytes / 1e6 / seconds:.2f}"
    )
    return True


def main() -> int:
    """Parse the command line and fetch every requested date."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list dates, count requests, fetch nothing but sidecars",
    )
    parser.add_argument(
        "--dates", help="comma-separated ISO dates, replacing the drawn pilot dates"
    )
    parser.add_argument("--members", choices=["control", "all"], default="control")
    parser.add_argument(
        "--workers", type=int, default=8, help=f"concurrent connections, at most {MAX_WORKERS}"
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be between 1 and {MAX_WORKERS}")

    dates = resolve_dates(override=args.dates)
    print(f"{len(dates)} dates: {', '.join(day.isoformat() for day in dates)}")
    started = time.monotonic()
    outcomes: list[bool] = []
    for day in dates:
        outcomes.append(
            fetch_date(day=day, members=args.members, workers=args.workers, dry_run=args.dry_run)
        )
        if not outcomes[-1]:
            print(f"Stopping after the first failed date, {day}", file=sys.stderr)
            break
    messages = len(PILOT_VARIABLES) * len(members_for(members=args.members)) * len(STEPS_HOURS)
    print(
        f"{len(dates)} dates planned x {messages} messages; "
        f"{sum(outcomes)} dates ok in {time.monotonic() - started:.0f} s"
    )
    return 0 if all(outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
