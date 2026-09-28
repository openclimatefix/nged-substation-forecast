"""Finding and addressing messages in the staged ECMWF ENS GRIB files on Source Cooperative.

Dynamical.org stages one folder per forecast date under `ecmwf-ifs-ens/`, holding five GRIB files
(`cf_sfc`, `cf_pl`, `pf_sfc_0`, `pf_sfc_1`, `pf_pl`) and one `.idx` sidecar per file. Each sidecar
line is a JSON object naming one message's parameter, step, ensemble member, byte offset and byte
length. This module parses those sidecars and the bucket listing, chooses which dates to sample,
and builds the single-range `Range` header that fetches the top of one message.
"""

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Final

GRIB_FILE_NAMES: Final[tuple[str, ...]] = (
    "cf_sfc.grib",
    "cf_pl.grib",
    "pf_sfc_0.grib",
    "pf_sfc_1.grib",
    "pf_pl.grib",
)
"""The five GRIB files a date must hold, with the `.idx` of each, to count as complete."""


@dataclass(frozen=True)
class IdxEntry:
    """One line of an `.idx` sidecar: where one message sits in its GRIB file."""

    parameter: str
    """The short name, such as `2t` or `z`."""
    level_type: str
    """`sfc` for a surface field or `pl` for a pressure-level field."""
    level: int
    """The pressure in hectopascals on a pressure level, and 0 at the surface."""
    step_hours: int
    """The forecast step in hours."""
    member: int
    """The ensemble member: 0 for the control and 1 to 50 for a perturbed member."""
    offset: int
    """The message's first byte within the file."""
    length: int
    """The message length in bytes."""


def parse_idx(*, text: str) -> list[IdxEntry]:
    """Parse the JSON lines of an `.idx` sidecar, in file order.

    The control file's lines carry no `number`, so a missing `number` means member 0, and a
    surface line carries no `levelist`, so a missing `levelist` means level 0. The offset is
    written as a float and is converted to an integer.

    Args:
        text: The whole sidecar.

    Returns:
        One entry per non-blank line.
    """
    entries = []
    for line in text.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        entries.append(
            IdxEntry(
                parameter=record["param"],
                level_type=record["levtype"],
                level=int(record.get("levelist", 0)),
                step_hours=int(record["step"]),
                member=int(record.get("number", 0)),
                offset=int(record["_offset"]),
                length=int(record["_length"]),
            )
        )
    return entries


def find_gaps(*, entries: list[IdxEntry]) -> list[str]:
    """Describe every place where the messages do not follow one another byte for byte.

    The first message must start at byte 0, and each later message must start where the previous
    one (in offset order) ends.

    Args:
        entries: The parsed sidecar.

    Returns:
        One sentence per gap or overlap, empty when the messages tile the file.
    """
    problems = []
    expected_offset = 0
    for entry in sorted(entries, key=lambda item: item.offset):
        if entry.offset != expected_offset:
            problems.append(
                f"{entry.parameter} step {entry.step_hours} member {entry.member} starts at "
                f"{entry.offset} but the previous message ended at {expected_offset}"
            )
        expected_offset = entry.offset + entry.length
    return problems


def prefix_range_header(*, entry: IdxEntry, prefix_bytes: int) -> str:
    """Return the `Range` header value that fetches the first `prefix_bytes` of one message.

    Args:
        entry: The message to fetch.
        prefix_bytes: How many bytes from the message's first byte to fetch.

    Returns:
        A single-range header value, such as `bytes=0-99`.

    Raises:
        ValueError: If `prefix_bytes` is not between 1 and the message length.
    """
    if not 1 <= prefix_bytes <= entry.length:
        raise ValueError(
            f"A prefix of {prefix_bytes} bytes is outside a {entry.length}-byte message"
        )
    return f"bytes={entry.offset}-{entry.offset + prefix_bytes - 1}"


def parse_listing(*, text: str) -> dict[date, set[str]]:
    """Parse the output of `aws s3 ls --recursive` on the `ecmwf-ifs-ens/` prefix.

    Args:
        text: One line per object, with the key as the last whitespace-separated field, and the
            key ending in `<YYYY-MM-DD>/<file name>`.

    Returns:
        The file names each date folder holds. A key whose parent folder is not a date is skipped.
    """
    files_by_date: dict[date, set[str]] = defaultdict(set)
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        *_, folder, name = fields[-1].split("/")
        try:
            folder_date = date.fromisoformat(folder)
        except ValueError:
            continue
        files_by_date[folder_date].add(name)
    return dict(files_by_date)


def complete_dates(*, files_by_date: dict[date, set[str]]) -> list[date]:
    """Return the dates that hold all five GRIB files and all five sidecars, oldest first.

    Args:
        files_by_date: The result of `parse_listing`.

    Returns:
        The sorted complete dates.
    """
    needed = {name for grib in GRIB_FILE_NAMES for name in (grib, f"{grib}.idx")}
    return sorted(day for day, names in files_by_date.items() if needed <= names)


def choose_pilot_dates(
    *,
    complete: list[date],
    first: date,
    last: date,
    sample_size: int,
    seed: int,
    always: tuple[date, ...],
) -> list[date]:
    """Choose the pilot dates: a seeded random sample plus some dates that are always included.

    Args:
        complete: The complete dates, from `complete_dates`.
        first: The first date the sample may draw from, inclusive.
        last: The last date the sample may draw from, inclusive.
        sample_size: How many dates to draw at random.
        seed: The seed of the random draw, so the same listing gives the same dates.
        always: Dates to include whether or not the draw picks them.

    Returns:
        The sorted, deduplicated dates.

    Raises:
        ValueError: If an `always` date is not complete, or fewer than `sample_size` complete
            dates lie between `first` and `last`.
    """
    missing = [day for day in always if day not in complete]
    if missing:
        raise ValueError(f"These required dates are not complete in the bucket: {missing}")
    candidates = [day for day in complete if first <= day <= last]
    if len(candidates) < sample_size:
        raise ValueError(f"Only {len(candidates)} complete dates lie in the range")
    drawn = random.Random(seed).sample(candidates, k=sample_size)
    return sorted(set(drawn) | set(always))
