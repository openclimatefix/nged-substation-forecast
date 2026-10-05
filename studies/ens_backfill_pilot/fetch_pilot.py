"""Fetch a band of grid rows from Dynamical.org's staged ECMWF ENS GRIB files, for the pilot.

Run `uv run python studies/ens_backfill_pilot/fetch_pilot.py --dry-run` first. The pilot fetches
the control member's 12 variables at 85 steps (1,020 messages) for 23 dates, one range request
per message, and writes one checkpoint file per date under `data/studies/ens_backfill_pilot/`.
A date already written is skipped, so a crashed run resumes where it stopped.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
from pilot_common import (
    ALWAYS_INCLUDED_DATES,
    BUCKET_PREFIX,
    CROP_COLUMNS,
    CROP_LATITUDES,
    CROP_LONGITUDES,
    CROP_LONGITUDES_DEGREES_EAST,
    DATE_SEED,
    FIRST_ROW,
    LAST_ROW,
    MAX_WORKERS,
    PILOT_END,
    PILOT_START,
    PILOT_VARIABLES,
    RANDOM_DATE_COUNT,
    STEPS_HOURS,
    FetchError,
    MembersType,
    PlannedMessage,
    RequestStats,
    fetch_idx,
    fetch_message_prefix,
    files_needed,
    members_for,
    pilot_file_path,
    plan_date,
    prefix_bytes,
)
from studies.ens_grib_source import (
    IdxEntry,
    choose_pilot_dates,
    complete_dates,
    find_gaps,
    parse_listing,
)
from studies.grib1_simple import decode_values, unpack_rows
from studies.sources import ENS_BACKFILL_PILOT_DIR


def list_bucket() -> str:
    """List every object under the ENS prefix with the AWS command line, anonymously."""
    result = subprocess.run(
        ["aws", "s3", "ls", "--no-sign-request", "--recursive", BUCKET_PREFIX],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


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


def load_plan(*, day: date, members: MembersType, stats: RequestStats) -> list[PlannedMessage]:
    """Fetch the sidecars of one date, check they tile their files, and plan the messages."""
    idx_by_file: dict[str, list[IdxEntry]] = {}
    for file_name in files_needed(members=members):
        entries = fetch_idx(day=day, file_name=file_name, stats=stats)
        gaps = find_gaps(entries=entries)
        if gaps:
            raise FetchError(f"{day} {file_name}.idx has {len(gaps)} gaps, the first: {gaps[0]}")
        idx_by_file[file_name] = entries
    return plan_date(idx_by_file=idx_by_file, members=members)


def fetch_one(*, day: date, message: PlannedMessage, stats: RequestStats) -> dict[str, object]:
    """Fetch, check and decode one message, returning what goes in the output arrays."""
    body, header = fetch_message_prefix(day=day, message=message, stats=stats)
    packed = unpack_rows(message=body, header=header, first_row=FIRST_ROW, last_row=LAST_ROW)
    cropped = packed[:, CROP_COLUMNS]
    return {
        "raw_x": cropped,
        "values": decode_values(packed=cropped, header=header).astype(np.float32),
        "reference_value": header.reference_value,
        "binary_scale": header.binary_scale,
        "decimal_scale": header.decimal_scale,
        "bits_per_value": header.bits_per_value,
        "sha256": hashlib.sha256(body).hexdigest(),
        "range_bytes": len(body),
    }


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
        arrays = _fetch_planned(
            day=day, planned=planned, members=members, workers=workers, stats=stats
        )
    except (FetchError, ValueError) as error:
        print(f"{day}: FAILED after {stats.requests} requests: {error}", file=sys.stderr)
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_suffix(".npz.partial")
    with partial.open("wb") as file:
        np.savez_compressed(file, **arrays)
    partial.rename(out_path)
    seconds = time.monotonic() - started
    print(
        f"{day}: messages={len(planned)} requests={stats.requests} "
        f"MB={stats.bytes / 1e6:.1f} seconds={seconds:.1f} retries={stats.retries} "
        f"extended_prefixes={stats.extended_prefixes} "
        f"req/s={stats.requests / seconds:.1f} MB/s={stats.bytes / 1e6 / seconds:.2f}"
    )
    return True


def _fetch_planned(
    *,
    day: date,
    planned: list[PlannedMessage],
    members: MembersType,
    workers: int,
    stats: RequestStats,
) -> dict[str, Any]:
    member_numbers = members_for(members=members)
    shape = (len(PILOT_VARIABLES), len(member_numbers), len(STEPS_HOURS))
    rows = LAST_ROW - FIRST_ROW + 1
    arrays: dict[str, Any] = {
        "values": np.full((*shape, rows, len(CROP_COLUMNS)), np.nan, dtype=np.float32),
        "raw_x": np.zeros((*shape, rows, len(CROP_COLUMNS)), dtype=np.uint16),
        "reference_value": np.full(shape, np.nan),
        "binary_scale": np.zeros(shape, dtype=np.int16),
        "decimal_scale": np.zeros(shape, dtype=np.int16),
        "bits_per_value": np.zeros(shape, dtype=np.uint8),
        "sha256": np.full(shape, "", dtype="U64"),
        "range_bytes": np.zeros(shape, dtype=np.int64),
        "message_offset": np.zeros(shape, dtype=np.int64),
        "message_length": np.zeros(shape, dtype=np.int64),
        "file_name": np.full(shape, "", dtype="U16"),
    }
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_one, day=day, message=message, stats=stats): message
            for message in planned
        }
        try:
            for future in as_completed(futures):
                message = futures[future]
                position = (
                    PILOT_VARIABLES.index(message.variable),
                    member_numbers.index(message.member),
                    STEPS_HOURS.index(message.step_hours),
                )
                result = future.result()
                for key, value in result.items():
                    arrays[key][position] = value
                arrays["message_offset"][position] = message.entry.offset
                arrays["message_length"][position] = message.entry.length
                arrays["file_name"][position] = message.file_name
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    arrays["latitudes"] = CROP_LATITUDES
    arrays["longitudes"] = CROP_LONGITUDES
    arrays["longitudes_degrees_east"] = CROP_LONGITUDES_DEGREES_EAST
    arrays["variables"] = np.array(PILOT_VARIABLES)
    arrays["members"] = np.array(member_numbers)
    arrays["steps_hours"] = np.array(STEPS_HOURS)
    arrays["date"] = np.array(day.isoformat())
    return arrays


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
