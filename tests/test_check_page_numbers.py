"""Tests for `studies/beam_diffuse_split/check_page_numbers.py`.

The check is the only unit-tested piece of the ENS and HRES wind study, so each test below is built
to fail on the bug it exists for: a page number that no longer matches the report by one digit, a
sign that disagrees inside a bracketed pair, a section taken from the wrong heading, and a rounding
that goes the wrong way at a half.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

import pytest

REPO_ROOT: Final[Path] = Path(__file__).parent.parent
MODULE_PATH: Final[Path] = REPO_ROOT / "studies" / "beam_diffuse_split" / "check_page_numbers.py"

REPORT: Final[str] = """\
### Report

| Scope | Contrast | Difference | 95% interval |
|---|---|---|---|
| all | hres_wind − ukv_wind | +0.142 | [+0.031, +0.254] |
| all | ens_wind − hres_wind | -0.125 | [-0.310, -0.040] |
| all | ukv_wind | 7.391 | [6.702, 8.115] |

Mean speed 7.85 m/s.
"""

PAGE: Final[str] = """\
# Page

## Earlier section

Nothing here is checked: 9.99 and [1.11, 2.22].

## ECMWF wind

HRES's error was 0.14 points [0.03, 0.25] worse than UKV's, at 100 m and 0.25 degrees, with 51
members. The mean speed was 7.85 m/s. UKV's own error was 7.39 [6.70, 8.12].

### A subsection

ENS beat HRES by 0.13 points [−0.31, −0.04].

## Later section

Not checked either: 5.55.
"""


def _load_module() -> ModuleType:
    """Import `check_page_numbers.py` from its path in `studies/beam_diffuse_split/`.

    Returns:
        The imported module.
    """
    spec = importlib.util.spec_from_file_location("check_page_numbers", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CHECK = _load_module()
HEADING: Final[str] = "## ECMWF wind"


def _write(*, directory: Path, page: str, report: str = REPORT) -> tuple[Path, Path]:
    """Write a page and a report into a directory.

    Args:
        directory: Where to write.
        page: The page text.
        report: The report text.

    Returns:
        The page path and the report path.
    """
    page_path = directory / "page.md"
    report_path = directory / "report.md"
    page_path.write_text(page)
    report_path.write_text(report)
    return page_path, report_path


def test_check_passes_when_every_number_is_in_the_report(tmp_path: Path) -> None:
    page_path, report_path = _write(directory=tmp_path, page=PAGE)
    checked = CHECK.check_page_numbers(
        page_path=page_path, report_path=report_path, heading=HEADING
    )
    assert checked == 7


def test_check_raises_when_one_number_is_perturbed(tmp_path: Path) -> None:
    page_path, report_path = _write(
        directory=tmp_path, page=PAGE.replace("0.14 points", "0.15 points")
    )
    with pytest.raises(ValueError, match=r"0\.15"):
        CHECK.check_page_numbers(page_path=page_path, report_path=report_path, heading=HEADING)


def test_check_raises_when_one_end_of_a_pair_is_perturbed(tmp_path: Path) -> None:
    page_path, report_path = _write(
        directory=tmp_path, page=PAGE.replace("[0.03, 0.25]", "[0.03, 0.26]")
    )
    with pytest.raises(ValueError, match=r"\[0\.03, 0\.26\]"):
        CHECK.check_page_numbers(page_path=page_path, report_path=report_path, heading=HEADING)


def test_check_raises_when_a_negative_page_pair_meets_a_positive_report_pair(
    tmp_path: Path,
) -> None:
    page = PAGE.replace("[0.03, 0.25]", "[-0.03, -0.25]")
    page_path, report_path = _write(directory=tmp_path, page=page)
    with pytest.raises(ValueError, match=r"\[-0\.03, -0\.25\]"):
        CHECK.check_page_numbers(page_path=page_path, report_path=report_path, heading=HEADING)


def test_check_reads_only_the_named_section(tmp_path: Path) -> None:
    """The earlier and later sections hold numbers the report lacks, and must not be read."""
    page_path, report_path = _write(directory=tmp_path, page=PAGE)
    section = CHECK.section_text(page_text=PAGE, heading=HEADING)
    assert "9.99" not in section
    assert "5.55" not in section
    assert "0.13 points" in section
    assert CHECK.check_page_numbers(page_path=page_path, report_path=report_path, heading=HEADING)


def test_check_rounds_a_half_up_from_the_printed_digits(tmp_path: Path) -> None:
    """The report prints -0.125, so a page rounding it to two places writes 0.13, not 0.12."""
    good_page = "## ECMWF wind\n\nThe gap was 0.13 points.\n"
    bad_page = "## ECMWF wind\n\nThe gap was 0.12 points.\n"
    page_path, report_path = _write(directory=tmp_path, page=good_page)
    assert CHECK.check_page_numbers(page_path=page_path, report_path=report_path, heading=HEADING)
    page_path.write_text(bad_page)
    with pytest.raises(ValueError, match=r"0\.12"):
        CHECK.check_page_numbers(page_path=page_path, report_path=report_path, heading=HEADING)


def test_check_raises_on_a_missing_or_repeated_heading(tmp_path: Path) -> None:
    page_path, report_path = _write(directory=tmp_path, page=PAGE)
    with pytest.raises(ValueError, match="appears 0 times"):
        CHECK.check_page_numbers(page_path=page_path, report_path=report_path, heading="## Absent")
    page_path.write_text(PAGE + "\n## ECMWF wind\n\nAgain 1.0.\n")
    with pytest.raises(ValueError, match="appears 2 times"):
        CHECK.check_page_numbers(page_path=page_path, report_path=report_path, heading=HEADING)


def test_check_raises_on_a_section_with_no_decimals(tmp_path: Path) -> None:
    page_path, report_path = _write(directory=tmp_path, page="## ECMWF wind\n\nNo numbers here.\n")
    with pytest.raises(ValueError, match="no decimal numbers"):
        CHECK.check_page_numbers(page_path=page_path, report_path=report_path, heading=HEADING)
