"""Read the numbers a study's `report.md` prints, so a chart cannot disagree with its page.

A study script prints every number its page quotes into a `report.md`. `ReportText` finds a line
or a table under a heading and raises unless it finds exactly one, and `estimate_in` reads a point
and its interval out of a line. Every lookup fails loudly when the report's wording changes, so a
chart is never drawn from a line that moved.
"""

import re
from pathlib import Path
from typing import Final, NamedTuple

import polars as pl

NUMBER_PATTERN: Final[str] = r"([+-]?\d+\.\d+)"
"""One signed decimal number, as a capture group."""

INTERVAL_PATTERN: Final[str] = rf"\[{NUMBER_PATTERN}, {NUMBER_PATTERN}\]"
"""A bracketed interval `[low, high]`, as two capture groups."""

T_PATTERN: Final[str] = rf"{NUMBER_PATTERN} points; month t-interval {INTERVAL_PATTERN}"
"""A point and its month t-interval, as a report's post hoc section prints them."""

BOOT_PATTERN: Final[str] = rf"{NUMBER_PATTERN} points, bootstrap {INTERVAL_PATTERN}"
"""A point and its bootstrap interval, as a report's stage B section prints them."""

_MIN_TABLE_ROWS: Final[int] = 3
"""A markdown table needs a header, a rule line, and at least one row."""


class Estimate(NamedTuple):
    """A point estimate and its interval."""

    point: float
    low: float
    high: float


def _cell(text: str) -> float | str | None:
    """Read a table cell as a float, a null for `None`, or the text."""
    if text == "None":
        return None
    try:
        return float(text)
    except ValueError:
        return text


class ReportText:
    """A study's `report.md`, with lookups that raise unless they find exactly one match."""

    def __init__(self, *, path: Path) -> None:
        """Read the report.

        Args:
            path: The report's path.
        """
        self.lines: list[str] = path.read_text().splitlines()

    def section(self, *, heading: str) -> list[str]:
        """Return the lines under a heading, up to the next heading.

        Args:
            heading: The heading's text, without its leading hashes.

        Returns:
            The lines.

        Raises:
            ValueError: If the heading is not found exactly once.
        """
        starts = [
            index
            for index, line in enumerate(self.lines)
            if line.startswith("#") and line.lstrip("#").strip() == heading
        ]
        if len(starts) != 1:
            msg = f"heading {heading!r} found {len(starts)} times in the report"
            raise ValueError(msg)
        body: list[str] = []
        for line in self.lines[starts[0] + 1 :]:
            if line.startswith("#"):
                break
            body.append(line)
        return body

    def line(self, *, heading: str, prefix: str) -> str:
        """Return the one line under a heading that starts with a prefix.

        Args:
            heading: The heading.
            prefix: The line's start.

        Returns:
            The line.

        Raises:
            ValueError: If no line, or more than one, matches.
        """
        found = [line for line in self.section(heading=heading) if line.startswith(prefix)]
        if len(found) != 1:
            msg = f"{len(found)} lines start with {prefix!r} under {heading!r}"
            raise ValueError(msg)
        return found[0]

    def table(self, *, heading: str) -> pl.DataFrame:
        """Return the first markdown table under a heading.

        Args:
            heading: The heading.

        Returns:
            The table, with cells that read as numbers as floats and `None` cells as nulls.

        Raises:
            ValueError: If the section holds no table.
        """
        block: list[str] = []
        for line in self.section(heading=heading):
            if line.startswith("|"):
                block.append(line)
            elif block:
                break
        rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in block]
        if len(rows) < _MIN_TABLE_ROWS:
            msg = f"no table under {heading!r}"
            raise ValueError(msg)
        header, body = rows[0], rows[2:]
        return pl.DataFrame(
            {name: [_cell(row[index]) for row in body] for index, name in enumerate(header)},
            strict=False,
        )


def estimate_in(*, line: str, pattern: str) -> Estimate:
    """Read an estimate and an interval out of a report line.

    Args:
        line: The report line.
        pattern: A regular expression with three groups: the point, the low end, and the high end.

    Returns:
        The estimate.

    Raises:
        ValueError: If the pattern does not match.
    """
    found = re.search(pattern, line)
    if found is None:
        msg = f"{pattern!r} does not match {line!r}"
        raise ValueError(msg)
    return Estimate(*(float(group) for group in found.groups()))
