"""Check that every decimal number and `[a, b]` interval on a docs page is in a study's `report.md`.

A study script prints every table the page quotes into a `report.md`, and the page is written by
hand from it. This check closes the gap: it takes part of the page (a whole section by its heading,
or one list item of a section), extracts every decimal number, every bracketed pair, and every
number followed by a bracketed pair, and requires each to equal a number the report printed, rounded
to the precision the page uses.

**Three kinds of page number are checked, from the most specific to the least.**

- **A triple `X points [a, b]`** (also `X [a, b]`, `X% [a, b]`, and the table form `X | [a, b]`) has
  to match one report row's value and both ends of its interval together. A value and an interval
  that are both in the report, but never in the same row, fail.
- **A pair `[a, b]` alone** has to match one report pair, both ends, in order.
- **A bare number** has to match any number the report printed.

**Signs follow one rule.** A pair end with no sign or a plus sign meets a report end that is not
negative, and a pair end with a minus sign meets a negative report end. A value or a bare number
with an explicit `+` or `-` meets a report number of that sign. A value or a bare number with no
sign meets either sign, because the page states the direction in words ("0.27 points lower").
`bare_magnitude_audit` lists the signs of the report numbers behind each bare number, so a reviewer
can check each direction claim by hand. A number that rounds to zero at the page's precision meets
either sign.

**Rounding is half-up, and starts from the full-precision value where one exists.** The report
prints each interval to four decimal places and the page quotes two. Rounding the printed digits
would turn 0.20466 into 0.2047 and then into 0.21, so the check reads the study's
`intervals.parquet` (one row per interval the report prints). Where a report number is the
four-decimal print of an interval value, the check rounds that value's own digits. A report number
with no row in `intervals.parquet` is rounded from its printed digits, using `decimal` so that
binary floating-point error cannot move a half.

**A bare number is a weak check.** The report prints hundreds of numbers between 0 and 1, so a bare
page number that has drifted by 0.01 often still equals some other report number. The check cannot
say that a number sits against the right label, only that the number exists in the report at the
page's precision. Shifting each unsigned bare number of the ECMWF results section of the past-wind
page by 0.01, in each direction, is still accepted for 91 of 130 shifts (70%). Shifting one value
or one interval end of a triple by 0.01 is still accepted for 12 of 306 shifts (4%), because a whole
report row has to match.
"""

import argparse
import logging
import re
from collections.abc import Mapping
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Final

import polars as pl

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

MINUS_SIGNS: Final[str] = "−–"
"""Unicode minus and en dash, both of which a page may use for a minus sign."""

NUMBER_PATTERN: Final[str] = r"[+\-]?\d+\.\d+"
"""One signed decimal number, as `report.md` prints it."""

PAIR_PATTERN: Final[str] = rf"\[\s*({NUMBER_PATTERN})\s*,\s*({NUMBER_PATTERN})\s*\]"
"""A bracketed interval, `[a, b]`, of two decimal numbers, capturing the two ends."""

PAIR_REGEX: Final[re.Pattern[str]] = re.compile(PAIR_PATTERN)

TRIPLE_REGEX: Final[re.Pattern[str]] = re.compile(
    rf"({NUMBER_PATTERN})(?:\s*(?:%|points?\b|pp\b|percentage points?\b))?\s*\|?\s*{PAIR_PATTERN}"
)
"""A number followed by a bracketed interval: `0.20 points [0.06, 0.33]` or `0.2 | [0.06, 0.33]`."""

NUMBER_REGEX: Final[re.Pattern[str]] = re.compile(NUMBER_PATTERN)

EXCLUDED_REGEX: Final[re.Pattern[str]] = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:m(?![/\w])|degrees?\b|°)"
)
"""Heights (`100 m`) and the ENS grid spacing (`0.25 degrees`, `0.25°`)."""

INTERVAL_PRINT_DECIMALS: Final[int] = 4
"""The decimal places the study report prints an interval value at."""

PageTriple = tuple[str, str, str]
"""A page or report row's value, then the lower and upper ends of its interval."""

ExactValues = Mapping[str, set[Decimal]]
"""Each printed report token to the full-precision values that print as it."""


def _ascii_signs(text: str) -> str:
    """Return `text` with unicode minus signs replaced by an ASCII hyphen.

    Args:
        text: Text from the page or the report.

    Returns:
        The text with each character of `MINUS_SIGNS` replaced by `-`.
    """
    return text.translate({ord(sign): "-" for sign in MINUS_SIGNS})


def section_text(*, page_text: str, heading: str) -> str:
    """Return the body of a markdown section, from its heading to the next of equal rank.

    Args:
        page_text: The whole page.
        heading: The heading line as written, including its leading `#` characters.

    Returns:
        The lines after the heading, up to but not including the next heading with the same or fewer
        `#` characters.

    Raises:
        ValueError: If the heading does not appear exactly once.
    """
    lines = page_text.splitlines()
    starts = [index for index, line in enumerate(lines) if line.strip() == heading.strip()]
    if len(starts) != 1:
        msg = f"heading {heading!r} appears {len(starts)} times in the page; expected exactly one"
        raise ValueError(msg)
    rank = len(heading) - len(heading.lstrip("#"))
    body: list[str] = []
    for line in lines[starts[0] + 1 :]:
        stripped = line.lstrip("#")
        if line.startswith("#") and stripped.startswith(" ") and len(line) - len(stripped) <= rank:
            break
        body.append(line)
    return "\n".join(body)


def bullet_text(*, section: str, prefix: str) -> str:
    """Return the one list item of a section whose text starts with `prefix`.

    A list item starts at a line beginning with a hyphen and a space, and runs over its indented
    continuation lines.

    Args:
        section: A section's text, from `section_text`.
        prefix: The start of the item, including its hyphen marker.

    Returns:
        The item's lines, joined.

    Raises:
        ValueError: If no item, or more than one item, starts with `prefix`.
    """
    items: list[list[str]] = []
    for line in section.splitlines():
        if line.startswith("- "):
            items.append([line])
        elif items and line.startswith(" "):
            items[-1].append(line)
    matches = ["\n".join(item) for item in items if item[0].startswith(prefix)]
    if len(matches) != 1:
        msg = f"{len(matches)} list items start with {prefix!r}; expected exactly one"
        raise ValueError(msg)
    return matches[0]


def _decimals(*, printed: str) -> int:
    """Return how many digits follow the decimal point in a printed number.

    Args:
        printed: A number such as `+0.142`.

    Returns:
        The count of digits after the point.
    """
    return len(printed.split(".")[1])


def _rounded(*, value: Decimal, decimals: int) -> Decimal:
    """Round a number half-up to `decimals` places, in magnitude.

    Args:
        value: The number, with any sign.
        decimals: The precision to round to.

    Returns:
        The rounded magnitude, never negative.
    """
    return abs(value).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)


def full_precision_values(*, intervals_path: Path | None) -> ExactValues:
    """Map each four-decimal number the report prints for an interval to the values behind it.

    Args:
        intervals_path: The study's `intervals.parquet`, with `value`, `lower` and `upper`
            columns, or None to check against the printed digits alone.

    Returns:
        Each printed token, such as `+0.2047`, to the full-precision values that print as it. A
        token can stand for more than one value, because two intervals can agree to four decimals.
    """
    if intervals_path is None:
        return {}
    frame = pl.read_parquet(intervals_path)
    exact: dict[str, set[Decimal]] = {}
    for column in ("value", "lower", "upper"):
        for value in frame[column].to_list():
            for token in (
                f"{value:+.{INTERVAL_PRINT_DECIMALS}f}",
                f"{value:.{INTERVAL_PRINT_DECIMALS}f}",
            ):
                exact.setdefault(token, set()).add(Decimal(repr(float(value))))
    return exact


def _matches(*, page: str, report: str, exact: ExactValues, unsigned_is_positive: bool) -> bool:
    """Return whether one page number equals one report number, at the page's precision and sign.

    Args:
        page: A page number with any sign, in ASCII.
        report: A report number with any sign, in ASCII.
        exact: `full_precision_values`'s result.
        unsigned_is_positive: Whether a page number with no sign has to meet a report number that is
            not negative. False lets it meet either sign.

    Returns:
        Whether the page number is accounted for by the report number.
    """
    decimals = _decimals(printed=page)
    wanted = _rounded(value=Decimal(page), decimals=decimals)
    candidates = exact.get(report) or {Decimal(report)}
    if not any(_rounded(value=value, decimals=decimals) == wanted for value in candidates):
        return False
    if not wanted:
        return True
    if page[0] not in "+-" and not unsigned_is_positive:
        return True
    return page.startswith("-") == report.startswith("-")


def _report_numbers(*, report_text: str) -> list[str]:
    """Return every decimal number the report printed, as printed.

    Args:
        report_text: The report's text.

    Returns:
        The numbers in order of appearance, with any sign.
    """
    return NUMBER_REGEX.findall(_ascii_signs(report_text))


def _report_pairs(*, report_text: str) -> list[tuple[str, str]]:
    """Return every bracketed pair the report printed, as printed.

    Args:
        report_text: The report's text.

    Returns:
        Each pair's two numbers, with any signs.
    """
    return PAIR_REGEX.findall(_ascii_signs(report_text))


def _report_triples(*, report_text: str) -> list[PageTriple]:
    """Return every number the report printed directly before a bracketed pair, as printed.

    Args:
        report_text: The report's text.

    Returns:
        Each row's value, then the two ends of its interval, with any signs.
    """
    return TRIPLE_REGEX.findall(_ascii_signs(report_text))


def _pair_matches(
    *, page_pair: tuple[str, str], report_pair: tuple[str, str], exact: ExactValues
) -> bool:
    """Return whether one page pair equals one report pair, at the page's precision.

    Both ends must match in magnitude and in sign, in order. A page end with no sign or a plus sign
    has to meet a report end that is not negative, and a page end with a minus sign a negative one.
    An end that rounds to zero at the page's precision meets either sign.

    Args:
        page_pair: The page's two numbers, with any signs.
        report_pair: The report's two numbers, with any signs.
        exact: `full_precision_values`'s result.

    Returns:
        Whether the pair is accounted for.
    """
    return all(
        _matches(page=page_end, report=report_end, exact=exact, unsigned_is_positive=True)
        for page_end, report_end in zip(page_pair, report_pair, strict=True)
    )


def _triple_matches(
    *, page_triple: PageTriple, report_triple: PageTriple, exact: ExactValues
) -> bool:
    """Return whether one page triple equals one report row, value and interval together.

    Args:
        page_triple: The page's value, then the two ends of its interval, with any signs.
        report_triple: The report row's value and interval ends, with any signs.
        exact: `full_precision_values`'s result.

    Returns:
        Whether the triple is accounted for. The value meets either sign when the page gives none.
    """
    return _matches(
        page=page_triple[0], report=report_triple[0], exact=exact, unsigned_is_positive=False
    ) and _pair_matches(
        page_pair=(page_triple[1], page_triple[2]),
        report_pair=(report_triple[1], report_triple[2]),
        exact=exact,
    )


def _split_page_numbers(
    *, section: str
) -> tuple[list[PageTriple], list[tuple[str, str]], list[str]]:
    """Split a section's numbers into triples, pairs on their own, and bare numbers.

    Args:
        section: A page section's text.

    Returns:
        The triples, then the pairs that no triple holds, then the numbers outside both, all in
        ASCII with any sign. Heights and grid spacings are left out.
    """
    cleaned = _ascii_signs(EXCLUDED_REGEX.sub(" ", section))
    triples = TRIPLE_REGEX.findall(cleaned)
    without_triples = TRIPLE_REGEX.sub(" ", cleaned)
    pairs = PAIR_REGEX.findall(without_triples)
    bare = NUMBER_REGEX.findall(PAIR_REGEX.sub(" ", without_triples))
    return triples, pairs, bare


def _bare_matches(*, printed: str, report_numbers: list[str], exact: ExactValues) -> list[str]:
    """Return the report numbers a bare page number matches, at the page's precision.

    Args:
        printed: A page number outside any pair or triple, with any sign.
        report_numbers: Every number the report printed, with any sign.
        exact: `full_precision_values`'s result.

    Returns:
        The matching report numbers. A page number with an explicit sign matches only report numbers
        of that sign, unless it rounds to zero. Empty when the page number is unaccounted for.
    """
    return [
        candidate
        for candidate in report_numbers
        if _matches(page=printed, report=candidate, exact=exact, unsigned_is_positive=False)
    ]


def unaccounted_numbers(
    *, section: str, report_text: str, exact: ExactValues | None = None
) -> list[str]:
    """Return every triple, pair and bare number in `section` that the report does not hold.

    Args:
        section: A page section's text.
        report_text: The report's text.
        exact: `full_precision_values`'s result, or None to round from the printed digits.

    Returns:
        A description of each triple, pair or number with no match in the report; empty when every
        one is accounted for.
    """
    resolved = exact or {}
    triples, pairs, bare = _split_page_numbers(section=section)
    report_triples = _report_triples(report_text=report_text)
    report_pairs = _report_pairs(report_text=report_text)
    report_numbers = _report_numbers(report_text=report_text)
    missing = [
        f"{triple[0]} [{triple[1]}, {triple[2]}]"
        for triple in triples
        if not any(
            _triple_matches(page_triple=triple, report_triple=row, exact=resolved)
            for row in report_triples
        )
    ]
    missing += [
        f"[{pair[0]}, {pair[1]}]"
        for pair in pairs
        if not any(
            _pair_matches(page_pair=pair, report_pair=report_pair, exact=resolved)
            for report_pair in report_pairs
        )
    ]
    missing += [
        printed
        for printed in bare
        if not _bare_matches(printed=printed, report_numbers=report_numbers, exact=resolved)
    ]
    return missing


def bare_magnitude_audit(
    *, section: str, report_text: str, exact: ExactValues | None = None
) -> list[str]:
    """List every bare number in `section` with the signs of the report numbers it matched.

    A bare number with no sign matches on magnitude, so the sign of the report number behind it is
    the one thing the check does not verify. The list lets a reviewer check each direction claim in
    the prose ("lower", "higher") against the sign the report printed.

    Args:
        section: A page section's text.
        report_text: The report's text.
        exact: `full_precision_values`'s result, or None to round from the printed digits.

    Returns:
        One line per bare number, such as `0.27: report signs -` or `0.14: report signs +, -`;
        a number matching report numbers of both signs is the one to audit first.
    """
    report_numbers = _report_numbers(report_text=report_text)
    audit: list[str] = []
    for printed in _split_page_numbers(section=section)[2]:
        matched = _bare_matches(printed=printed, report_numbers=report_numbers, exact=exact or {})
        signs = sorted({"-" if candidate.startswith("-") else "+" for candidate in matched})
        audit.append(f"{printed}: report signs {', '.join(signs) if signs else 'none'}")
    return audit


def check_page_numbers(
    *,
    page_path: Path,
    report_path: Path,
    heading: str,
    intervals_path: Path | None = None,
    bullet_prefix: str | None = None,
    allow_empty: bool = False,
) -> int:
    """Raise unless every number in a page section is in the report.

    Args:
        page_path: The docs page.
        report_path: The study's `report.md`.
        heading: The section's heading line as written on the page, including its `#` characters.
        intervals_path: The study's `intervals.parquet`, read for full-precision values, or None.
        bullet_prefix: When given, only the one list item of the section that starts with this
            text is checked.
        allow_empty: Whether a section with no decimal numbers passes, for a section that is
            checked in case a number is added to it later.

    Returns:
        How many triples, pairs and bare numbers were checked.

    Raises:
        ValueError: Naming every number or pair the report does not hold, or if the checked text
            holds none at all and `allow_empty` is False, which would mean the heading matched the
            wrong section.
    """
    section = section_text(page_text=page_path.read_text(), heading=heading)
    label = heading
    if bullet_prefix is not None:
        section = bullet_text(section=section, prefix=bullet_prefix)
        label = f"{heading} / {bullet_prefix}"
    report_text = report_path.read_text()
    exact = full_precision_values(intervals_path=intervals_path)
    missing = unaccounted_numbers(section=section, report_text=report_text, exact=exact)
    triples, pairs, bare = _split_page_numbers(section=section)
    checked = len(triples) + len(pairs) + len(bare)
    if checked == 0 and not allow_empty:
        msg = f"section {label!r} holds no decimal numbers to check"
        raise ValueError(msg)
    if missing:
        msg = f"{len(missing)} numbers in {label!r} are not in {report_path.name}: {missing}"
        raise ValueError(msg)
    for line in bare_magnitude_audit(section=section, report_text=report_text, exact=exact):
        _LOG.info("bare magnitude, audit its direction by hand: %s", line)
    return checked


def main() -> int:
    """Check page sections against a report, and print each bare-magnitude audit list.

    Returns:
        0 when every number is accounted for; a `ValueError` is raised otherwise.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("page", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--intervals",
        type=Path,
        default=None,
        help="The study's intervals.parquet; defaults to the one beside the report, if present.",
    )
    parser.add_argument(
        "--section",
        action="append",
        default=[],
        help="A heading line, including its # characters, whose whole section is checked.",
    )
    parser.add_argument(
        "--bullet",
        action="append",
        nargs=2,
        default=[],
        metavar=("HEADING", "PREFIX"),
        help="A heading, and the start of the one list item in it that is checked.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Pass a section or list item that holds no decimal numbers.",
    )
    arguments = parser.parse_args()
    if not arguments.section and not arguments.bullet:
        parser.error("give at least one --section or --bullet")
    beside = arguments.report.parent / "intervals.parquet"
    intervals = arguments.intervals or (beside if beside.exists() else None)
    checks = [(heading, None) for heading in arguments.section]
    checks += [(heading, prefix) for heading, prefix in arguments.bullet]
    for heading, prefix in checks:
        checked = check_page_numbers(
            page_path=arguments.page,
            report_path=arguments.report,
            heading=heading,
            intervals_path=intervals,
            bullet_prefix=prefix,
            allow_empty=arguments.allow_empty,
        )
        _LOG.info("%s: %d triples, pairs and numbers checked", prefix or heading, checked)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
