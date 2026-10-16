"""Fetching a band of grid rows from Dynamical.org's staged ECMWF ENS GRIB files.

The ENS backfill pilot and the control-member backfill fetch both download the same band of grid
rows from each message. `studies.grib1_simple` reads the messages, `studies.ens_grib_source` reads
the sidecars, and this module holds the constants, the HTTP code, and the one-date fetch they share.
"""

import hashlib
import os
import random
import subprocess
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import requests

from studies.ens_grib_source import IdxEntry, find_gaps, parse_idx, prefix_range_header
from studies.grib1_simple import (
    Grib1Header,
    bytes_needed_for_rows,
    decode_values,
    parse_header,
    unpack_rows,
)
from studies.sources import ENS_BACKFILL_PILOT_DIR

PROXY_URL: Final[str] = "https://data.source.coop/dynamical/ecmwf-ifs-grib/ecmwf-ifs-ens"
"""Source Cooperative's Cloudflare proxy for the staged files, about 30 times faster than direct
S3 from the UK."""

BUCKET_PREFIX: Final[str] = (
    "s3://us-west-2.opendata.source.coop/dynamical/ecmwf-ifs-grib/ecmwf-ifs-ens/"
)
"""The bucket prefix that `aws s3 ls --no-sign-request` lists."""

USER_AGENT: Final[str] = (
    "nged-substation-forecast-ens-backfill/0.1 (+https://github.com/openclimatefix/nged-substation-forecast)"
)
"""The proxy answers 403 to Python's default user agent."""


def list_bucket() -> str:
    """List every object under the ENS prefix with the AWS command line, anonymously."""
    result = subprocess.run(
        ["aws", "s3", "ls", "--no-sign-request", "--recursive", BUCKET_PREFIX],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


MembersType = Literal["control", "all"]
"""Which ensemble members a run fetches."""

PILOT_START: Final[date] = date(2021, 3, 21)
"""The first date of the solidly complete part of the archive."""

PILOT_END: Final[date] = date(2024, 3, 31)
"""The last staged date, the day before the open-data feed that Dynamical.org serves begins."""

ALWAYS_INCLUDED_DATES: Final[tuple[date, ...]] = (
    date(2023, 6, 26),
    date(2023, 6, 28),
    date(2024, 3, 31),
)
"""The two dates either side of the 2023-06-27 change to a 9 km native model, and the last date."""

RANDOM_DATE_COUNT: Final[int] = 20
DATE_SEED: Final[int] = 20260926

PILOT_VARIABLES: Final[tuple[str, ...]] = (
    "2t",
    "2d",
    "10u",
    "10v",
    "100u",
    "100v",
    "sp",
    "msl",
    "tp",
    "strd",
    "ssrd",
    "z500",
)
"""The messages the `Nwp` contract needs, under their ECMWF short names. `z500` is geopotential on
the 500 hPa level, which the pipeline divides by g to get `geopotential_height_500hpa`."""

ACCUMULATED_VARIABLES: Final[tuple[str, ...]] = ("tp", "strd", "ssrd")
"""The variables ECMWF accumulates from the start of the forecast."""

VARIABLE_MESSAGE: Final[dict[str, tuple[str, str, int]]] = {
    name: ("sfc", name, 0) for name in PILOT_VARIABLES if name != "z500"
} | {"z500": ("pl", "z", 500)}
"""Each variable's `(levtype, idx param, level)`."""

STEPS_HOURS: Final[tuple[int, ...]] = tuple(range(0, 145, 3)) + tuple(range(150, 361, 6))
"""The 85 forecast steps, identical to `contracts.weather_schemas.ECMWF_ENS_LEAD_TIME_HOURS`."""

FIRST_ROW: Final[int] = 115
"""Row 115 is latitude 61.25, the northern edge of the pipeline's grid."""

LAST_ROW: Final[int] = 164
"""Row 164 is latitude 49.0, half a degree south of the pipeline's grid."""

FIRST_COLUMN_LONGITUDE: Final[float] = -12.0
LAST_COLUMN_LONGITUDE: Final[float] = 5.0
GRID_STEP_DEGREES: Final[float] = 0.25
GLOBAL_COLUMNS: Final[int] = 1440
GLOBAL_ROWS: Final[int] = 721
ROW_BYTES: Final[int] = GLOBAL_COLUMNS * 2

EXPECTED_DATA_START: Final[int] = 157
"""Where the packed values start in every message seen so far (8 + 106 + 32 + 11)."""

MEMBER_COUNT: Final[int] = 51

_LONGITUDE_STEPS: Final[np.ndarray] = np.arange(
    round(FIRST_COLUMN_LONGITUDE / GRID_STEP_DEGREES),
    round(LAST_COLUMN_LONGITUDE / GRID_STEP_DEGREES) + 1,
)
CROP_COLUMNS: Final[np.ndarray] = _LONGITUDE_STEPS % GLOBAL_COLUMNS
"""The 69 columns from 12 degrees west to 5 degrees east, in west-to-east order, which cross the
start of the 0 to 359.75 longitude axis."""

CROP_LONGITUDES_DEGREES_EAST: Final[np.ndarray] = CROP_COLUMNS * GRID_STEP_DEGREES
CROP_LONGITUDES: Final[np.ndarray] = _LONGITUDE_STEPS * GRID_STEP_DEGREES
"""The cropped columns' longitudes on the -180 to 180 axis the pipeline uses."""

CROP_LATITUDES: Final[np.ndarray] = 90.0 - np.arange(FIRST_ROW, LAST_ROW + 1) * GRID_STEP_DEGREES

MAX_WORKERS: Final[int] = 16
MAX_ATTEMPTS: Final[int] = 8


def pilot_file_path(*, members: MembersType, day: date) -> Path:
    """Return the path of one date's checkpoint file."""
    folder = "control" if members == "control" else "all_members"
    return ENS_BACKFILL_PILOT_DIR / folder / f"{day.isoformat()}.npz"


class FetchError(Exception):
    """A request failed in a way that retrying will not fix."""


class RetriesExhaustedError(FetchError):
    """Every attempt at a request failed, which points at the server or the network."""


@dataclass
class RequestStats:
    """Thread-safe counters for one run of requests."""

    requests: int = 0
    bytes: int = 0
    retries: int = 0
    extended_prefixes: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, *, requests_made: int = 0, byte_count: int = 0, retries: int = 0) -> None:
        """Add to the counters."""
        with self.lock:
            self.requests += requests_made
            self.bytes += byte_count
            self.retries += retries


_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
        _local.session.headers["User-Agent"] = USER_AGENT
    return _local.session


def get_bytes(
    *, url: str, byte_range: str | None, expected_length: int | None, stats: RequestStats
) -> bytes:
    """GET a URL with retries, and return the body.

    A range request must be answered with 206 and the exact number of bytes asked for. A 200 to a
    range request means the server ignored the range and is about to send the whole file, so the
    connection is closed unread. Request errors (including a connection reset part-way through a
    body), 429, 5xx and a short body are retried with exponential backoff and jitter. Any other
    status raises.

    Args:
        url: The URL to fetch.
        byte_range: A `Range` header value, or `None` for a whole small file.
        expected_length: The number of bytes a range must return, or `None`.
        stats: Counters to update.

    Returns:
        The response body.

    Raises:
        FetchError: On a status that is not retried, or when every attempt failed.
    """
    headers = {"Range": byte_range} if byte_range else {}
    last_problem = ""
    for attempt in range(MAX_ATTEMPTS):
        if attempt:
            stats.add(retries=1)
            time.sleep(min(60.0, 2.0**attempt) * (0.5 + random.random()))
        try:
            with _session().get(url, headers=headers, stream=True, timeout=(15, 90)) as response:
                stats.add(requests_made=1)
                if response.status_code == 429 or response.status_code >= 500:
                    last_problem = f"HTTP {response.status_code}"
                    continue
                wanted_status = 206 if byte_range else 200
                if response.status_code != wanted_status:
                    raise FetchError(f"{url} {byte_range}: HTTP {response.status_code}")
                body = response.content
        except requests.RequestException as error:
            last_problem = f"{type(error).__name__}: {error}"
            continue
        if expected_length is not None and len(body) != expected_length:
            last_problem = f"got {len(body)} bytes, expected {expected_length}"
            continue
        stats.add(byte_count=len(body))
        return body
    raise RetriesExhaustedError(
        f"{url} {byte_range}: gave up after {MAX_ATTEMPTS} attempts ({last_problem})"
    )


def file_url(*, day: date, file_name: str) -> str:
    """Return the proxy URL of one staged file."""
    return f"{PROXY_URL}/{day.isoformat()}/{file_name}"


def source_file_name(*, member: int, levtype: str) -> str:
    """Return the staged file that holds `member`'s messages of `levtype`, for a member 0 to 50."""
    if levtype == "pl":
        return "cf_pl.grib" if member == 0 else "pf_pl.grib"
    if member == 0:
        return "cf_sfc.grib"
    return "pf_sfc_0.grib" if member <= 25 else "pf_sfc_1.grib"


@dataclass(frozen=True)
class PlannedMessage:
    """One message to fetch, and where its rows go in the output arrays."""

    variable: str
    member: int
    step_hours: int
    file_name: str
    entry: IdxEntry


def members_for(*, members: MembersType) -> tuple[int, ...]:
    """Return the member numbers a run fetches."""
    return (0,) if members == "control" else tuple(range(MEMBER_COUNT))


def fetch_idx(*, day: date, file_name: str, stats: RequestStats) -> list[IdxEntry]:
    """Fetch and parse one file's sidecar."""
    body = get_bytes(
        url=file_url(day=day, file_name=f"{file_name}.idx"),
        byte_range=None,
        expected_length=None,
        stats=stats,
    )
    return parse_idx(text=body.decode())


def plan_date(
    *, idx_by_file: dict[str, list[IdxEntry]], members: MembersType
) -> list[PlannedMessage]:
    """Choose the messages to fetch for one date, one for every (variable, member, step).

    Args:
        idx_by_file: The parsed sidecars, keyed by GRIB file name.
        members: Which members to fetch.

    Returns:
        The messages, ordered by variable, then member, then step.

    Raises:
        FetchError: If any (variable, member, step) has no message or more than one.
    """
    wanted_members = members_for(members=members)
    planned = []
    for variable in PILOT_VARIABLES:
        levtype, param, level = VARIABLE_MESSAGE[variable]
        for member in wanted_members:
            file_name = source_file_name(member=member, levtype=levtype)
            found: dict[int, list[IdxEntry]] = {}
            for entry in idx_by_file[file_name]:
                if (
                    entry.parameter == param
                    and entry.level_type == levtype
                    and entry.level == level
                    and entry.member == member
                ):
                    found.setdefault(entry.step_hours, []).append(entry)
            for step in STEPS_HOURS:
                if len(found.get(step, [])) != 1:
                    raise FetchError(
                        f"{file_name}: {variable} member {member} step {step} has "
                        f"{len(found.get(step, []))} idx entries, expected 1"
                    )
                planned.append(
                    PlannedMessage(
                        variable=variable,
                        member=member,
                        step_hours=step,
                        file_name=file_name,
                        entry=found[step][0],
                    )
                )
    return planned


def files_needed(*, members: MembersType) -> tuple[str, ...]:
    """Return the GRIB files a run reads."""
    names = {
        source_file_name(member=member, levtype=levtype)
        for member in members_for(members=members)
        for levtype in ("sfc", "pl")
    }
    return tuple(sorted(names))


def prefix_bytes() -> int:
    """Return the request length in bytes: the message start up to the end of the last row."""
    return EXPECTED_DATA_START + (LAST_ROW + 1) * ROW_BYTES


def fetch_message_prefix(
    *, day: date, message: PlannedMessage, stats: RequestStats
) -> tuple[bytes, Grib1Header]:
    """Fetch the top of one message with one range request and check its header against the idx.

    The request is sized on the assumption that the packed values start at byte 157. If the
    parsed header says they start elsewhere, the message is requested again with the right length.

    Args:
        day: The forecast date, which the header's reference time must match at 00 UTC.
        message: The message to fetch.
        stats: Counters to update.

    Returns:
        The bytes fetched, and the header parsed from them.

    Raises:
        FetchError: If the request fails.
        ValueError: If the header is not one this module can decode, or disagrees with the idx
            entry about the message length, parameter, level, step or member, or the grid is not
            the 1440 by 721 grid the crop assumes.
    """
    entry = message.entry
    length = prefix_bytes()
    body = b""
    header = None
    for _ in range(2):
        body = get_bytes(
            url=file_url(day=day, file_name=message.file_name),
            byte_range=prefix_range_header(entry=entry, prefix_bytes=length),
            expected_length=length,
            stats=stats,
        )
        header = parse_header(message=body)
        needed = bytes_needed_for_rows(header=header, last_row=LAST_ROW)
        if needed <= length:
            break
        with stats.lock:
            stats.extended_prefixes += 1
        length = needed
    assert header is not None
    _check_header_against_idx(day=day, message=message, header=header)
    return body[:length], header


def _check_header_against_idx(*, day: date, message: PlannedMessage, header: Grib1Header) -> None:
    levtype, param, level = VARIABLE_MESSAGE[message.variable]
    expected_level_type = 100 if levtype == "pl" else 1
    problems = []
    if header.total_length != message.entry.length:
        problems.append(f"length {header.total_length} != idx {message.entry.length}")
    if header.parameter != param:
        problems.append(f"parameter {header.parameter} != {param}")
    if (header.level_type, header.level) != (expected_level_type, level):
        problems.append(
            f"level {header.level_type}/{header.level} != {expected_level_type}/{level}"
        )
    if header.step_hours != message.step_hours:
        problems.append(f"step {header.step_hours} != {message.step_hours}")
    if header.member != message.member:
        problems.append(f"member {header.member} != {message.member}")
    if (header.reference_time.date(), header.reference_time.hour) != (day, 0):
        problems.append(f"reference time {header.reference_time} != {day} 00Z")
    if (header.ni, header.nj) != (GLOBAL_COLUMNS, GLOBAL_ROWS):
        problems.append(f"grid {header.ni}x{header.nj}")
    if (header.first_latitude_degrees, header.first_longitude_degrees) != (90.0, 0.0):
        problems.append("the grid does not start at 90N, 0E")
    if (header.latitude_increment_degrees, header.longitude_increment_degrees) != (0.25, 0.25):
        problems.append("the grid spacing is not 0.25 degrees")
    if problems:
        raise ValueError(
            f"{day} {message.file_name} {message.variable} member {message.member} step "
            f"{message.step_hours}: " + "; ".join(problems)
        )


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


def fetch_planned(
    *,
    day: date,
    planned: list[PlannedMessage],
    members: MembersType,
    workers: int,
    stats: RequestStats,
) -> dict[str, Any]:
    """Fetch every planned message of one date and return the arrays of its checkpoint file."""
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


def write_checkpoint(*, arrays: dict[str, Any], out_path: Path) -> None:
    """Write one date's arrays through a `.partial` file, so a crash leaves no half-written file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_suffix(".npz.partial")
    with partial.open("wb") as file:
        np.savez_compressed(file, **arrays)
        file.flush()
        os.fsync(file.fileno())
    partial.rename(out_path)


def shard_dates(*, dates: Sequence[date], shard: int, shard_count: int) -> list[date]:
    """Return every `shard_count`-th date from `shard`, so parallel processes interleave.

    Interleaving keeps the processes working through the dates in the same order, so an early
    date range finishes first however many processes run.

    Args:
        dates: The dates in the order they should be fetched.
        shard: This process's number, from 0 to `shard_count - 1`.
        shard_count: How many processes share the dates.

    Returns:
        This shard's dates, in the original order.

    Raises:
        ValueError: If `shard` is outside 0 to `shard_count - 1`.
    """
    if not 0 <= shard < shard_count:
        raise ValueError(f"shard {shard} is not in 0 to {shard_count - 1}")
    return list(dates)[shard::shard_count]
