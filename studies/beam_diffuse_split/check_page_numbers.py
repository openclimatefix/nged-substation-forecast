"""Check that every decimal number and `[a, b]` interval on a docs-page section is in a `report.md`.

A study script prints every table the page quotes into a `report.md`, and the page is written by
hand from it. This check closes the gap: it takes one section of the page by its heading, extracts
every decimal number and every bracketed pair from it, and requires each to equal a number the
report printed, rounded to the precision the page uses.

The report prints signed differences to three decimal places, such as `+0.142 [+0.031, +0.254]`,
where the page writes `0.14 points [0.03, 0.25]`. The convention for signs:

- **A bracketed pair `[a, b]` matches including sign.** A page end with a minus sign meets a
  negative report end, and a page end with no sign or a plus sign meets a report end that is not
  negative. A pair `[0.03, 0.25]` therefore fails against a report pair `[-0.03, -0.25]`. An end
  that rounds to zero at the page's precision meets either sign.
- **A bare number matches on its magnitude,** because the page states the direction in words ("0.27
  points lower"). A bare number with an explicit minus sign has to meet a negative report number.
  `bare_magnitude_audit` lists every bare number with the signs of the report numbers it matched,
  and `check_page_numbers` logs the list, so a reviewer can audit each direction claim by hand.

Rounding is half-up on the printed digits, using `decimal`, so a value such as 0.125 does not
round to a neighbour by binary floating-point error.

The check passes a section only when nothing in it is unaccounted for. It cannot say that a number
sits against the right label, only that the number exists in the report at the page's precision,
so a page number that coincides with an unrelated report number still passes.
"""

import argparse
import logging
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Final

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

MINUS_SIGNS: Final[str] = "−–"
"""Unicode minus and en dash, both of which a page may use for a minus sign."""

NUMBER_PATTERN: Final[str] = r"[+\-]?\d+\.\d+"
"""One signed decimal number, as `report.md` prints it."""

PAIR_REGEX: Final[re.Pattern[str]] = re.compile(
    rf"\[\s*({NUMBER_PATTERN})\s*,\s*({NUMBER_PATTERN})\s*\]"
)
"""A bracketed interval, `[a, b]`, of two decimal numbers."""

NUMBER_REGEX: Final[re.Pattern[str]] = re.compile(NUMBER_PATTERN)

EXCLUDED_REGEX: Final[re.Pattern[str]] = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:m(?![/\w])|degrees?\b|°|members\b)"
)
"""Heights (`100 m`), the ENS grid spacing (`0.25 degrees`) and the ensemble size (`51 members`)."""


def _ascii_signs(text: str) -> str:
    """Return `text` with unicode minus signs replaced by an ASCII hyphen.

    Args:
        text: Text from the page.

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


def _decimals(*, printed: str) -> int:
    """Return how many digits follow the decimal point in a printed number.

    Args:
        printed: A number such as `+0.142`.

    Returns:
        The count of digits after the point.
    """
    return len(printed.split(".")[1])


def _rounded(*, printed: str, decimals: int) -> Decimal:
    """Round a printed number half-up to `decimals` places, in magnitude.

    Args:
        printed: A number such as `+0.142`, or `-0.031`.
        decimals: The precision to round to.

    Returns:
        The rounded magnitude, never negative.
    """
    magnitude = abs(Decimal(printed))
    return magnitude.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)


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
    return [(low, high) for low, high in PAIR_REGEX.findall(_ascii_signs(report_text))]


def _pair_matches(*, page_pair: tuple[str, str], report_pair: tuple[str, str]) -> bool:
    """Return whether one page pair equals one report pair, at the page's precision.

    Both ends must match in magnitude and in sign. A page end carrying a minus sign must meet a
    report end that is negative, and a page end with no sign or a plus sign must meet a report end
    that is not negative. An end that rounds to zero at the page's precision meets either sign.

    Args:
        page_pair: The page's two numbers, with any signs.
        report_pair: The report's two numbers, with any signs.

    Returns:
        Whether the pair is accounted for.
    """
    for page_end, report_end in zip(page_pair, report_pair, strict=True):
        decimals = _decimals(printed=page_end)
        if _rounded(printed=page_end, decimals=decimals) != _rounded(
            printed=report_end, decimals=decimals
        ):
            return False
        if _rounded(printed=page_end, decimals=decimals) and (
            page_end.startswith("-") != report_end.startswith("-")
        ):
            return False
    return True


def unaccounted_numbers(*, section: str, report_text: str) -> list[str]:
    """Return every decimal number and bracketed pair in `section` that the report does not hold.

    Args:
        section: A page section's text.
        report_text: The report's text.

    Returns:
        A description of each number or pair with no match in the report; empty when every one is
        accounted for.
    """
    cleaned = _ascii_signs(EXCLUDED_REGEX.sub(" ", section))
    report_numbers = _report_numbers(report_text=report_text)
    report_pairs = _report_pairs(report_text=report_text)
    missing: list[str] = []
    for low, high in PAIR_REGEX.findall(cleaned):
        page_pair = (low, high)
        if not any(_pair_matches(page_pair=page_pair, report_pair=pair) for pair in report_pairs):
            missing.append(f"[{low}, {high}]")
    without_pairs = PAIR_REGEX.sub(" ", cleaned)
    missing.extend(
        printed
        for printed in NUMBER_REGEX.findall(without_pairs)
        if not _bare_matches(printed=printed, report_numbers=report_numbers)
    )
    return missing


def _bare_matches(*, printed: str, report_numbers: list[str]) -> list[str]:
    """Return the report numbers a bare page number matches, at the page's precision.

    Args:
        printed: A page number outside any pair, with any sign.
        report_numbers: Every number the report printed, with any sign.

    Returns:
        The matching report numbers. A page number with a minus sign matches only negative ones,
        unless it rounds to zero. Empty when the page number is unaccounted for.
    """
    decimals = _decimals(printed=printed)
    wanted = _rounded(printed=printed, decimals=decimals)
    return [
        candidate
        for candidate in report_numbers
        if _rounded(printed=candidate, decimals=decimals) == wanted
        and (not printed.startswith("-") or not wanted or candidate.startswith("-"))
    ]


def bare_magnitude_audit(*, section: str, report_text: str) -> list[str]:
    """List every bare number in `section` with the signs of the report numbers it matched.

    A bare number matches on magnitude, so the sign of the report number behind it is the one thing
    the check does not verify. The list lets a reviewer check each direction claim in the prose
    ("lower", "higher") against the sign the report printed.

    Args:
        section: A page section's text.
        report_text: The report's text.

    Returns:
        One line per bare number, such as `0.27: report signs -` or `0.14: report signs +, -`;
        a number matching report numbers of both signs is the one to audit first.
    """
    cleaned = _ascii_signs(EXCLUDED_REGEX.sub(" ", section))
    report_numbers = _report_numbers(report_text=report_text)
    audit: list[str] = []
    for printed in NUMBER_REGEX.findall(PAIR_REGEX.sub(" ", cleaned)):
        signs = sorted(
            {
                "-" if candidate.startswith("-") else "+"
                for candidate in _bare_matches(printed=printed, report_numbers=report_numbers)
            },
            reverse=True,
        )
        audit.append(f"{printed}: report signs {', '.join(signs) if signs else 'none'}")
    return audit


def check_page_numbers(*, page_path: Path, report_path: Path, heading: str) -> int:
    """Raise unless every decimal number and bracketed pair in a page section is in the report.

    Args:
        page_path: The docs page.
        report_path: The study's `report.md`.
        heading: The section's heading line as written on the page, including its `#` characters.

    Returns:
        How many numbers and pairs were checked.

    Raises:
        ValueError: Naming every number or pair the report does not hold, or if the section holds
            none at all, which would mean the heading matched the wrong section.
    """
    section = section_text(page_text=page_path.read_text(), heading=heading)
    report_text = report_path.read_text()
    missing = unaccounted_numbers(section=section, report_text=report_text)
    cleaned = _ascii_signs(EXCLUDED_REGEX.sub(" ", section))
    checked = len(PAIR_REGEX.findall(cleaned)) + len(
        NUMBER_REGEX.findall(PAIR_REGEX.sub(" ", cleaned))
    )
    if checked == 0:
        msg = f"section {heading!r} holds no decimal numbers to check"
        raise ValueError(msg)
    if missing:
        msg = f"{len(missing)} numbers in {heading!r} are not in {report_path.name}: {missing}"
        raise ValueError(msg)
    for line in bare_magnitude_audit(section=section, report_text=report_text):
        _LOG.info("bare magnitude, audit its direction by hand: %s", line)
    return checked


def main() -> int:
    """Check one page section against a report, and print the bare-magnitude audit list.

    Returns:
        0 when every number is accounted for; a `ValueError` is raised otherwise.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("page", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("heading", help="The section's heading line, including its # characters.")
    arguments = parser.parse_args()
    checked = check_page_numbers(
        page_path=arguments.page, report_path=arguments.report, heading=arguments.heading
    )
    _LOG.info("%d numbers and pairs checked", checked)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
