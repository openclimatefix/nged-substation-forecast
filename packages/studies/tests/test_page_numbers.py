"""Tests for `studies.page_numbers`.

The check is the only unit-tested piece of the ENS and HRES wind study, so each test below is built
to fail on the bug it exists for: a page number that no longer matches the report by one digit, a
sign that disagrees inside a bracketed pair or on a bare number, a value that meets an interval from
another row, a bare magnitude listed for audit, a section taken from the wrong heading, and a
rounding that goes the wrong way at a half or that rounds twice.
"""

import logging
from pathlib import Path
from typing import Final

import polars as pl
import pytest

from studies import page_numbers

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


def _write_intervals(*, directory: Path, values: list[float]) -> Path:
    """Write an `intervals.parquet` whose value, lower and upper columns all hold `values`.

    Args:
        directory: Where to write.
        values: The full-precision numbers the report prints to four decimals.

    Returns:
        The parquet path.
    """
    path = directory / "intervals.parquet"
    pl.DataFrame({"value": values, "lower": values, "upper": values}).write_parquet(path)
    return path


def test_check_passes_when_every_number_is_in_the_report(tmp_path: Path) -> None:
    page_path, report_path = _write(directory=tmp_path, page=PAGE)
    checked = page_numbers.check_page_numbers(
        page_path=page_path, report_path=report_path, heading=HEADING
    )
    assert checked == 4


def test_check_raises_when_one_number_is_perturbed(tmp_path: Path) -> None:
    page_path, report_path = _write(
        directory=tmp_path, page=PAGE.replace("0.14 points", "0.15 points")
    )
    with pytest.raises(ValueError, match=r"0\.15"):
        page_numbers.check_page_numbers(
            page_path=page_path, report_path=report_path, heading=HEADING
        )


def test_check_raises_when_one_end_of_a_pair_is_perturbed(tmp_path: Path) -> None:
    page_path, report_path = _write(
        directory=tmp_path, page=PAGE.replace("[0.03, 0.25]", "[0.03, 0.26]")
    )
    with pytest.raises(ValueError, match=r"\[0\.03, 0\.26\]"):
        page_numbers.check_page_numbers(
            page_path=page_path, report_path=report_path, heading=HEADING
        )


def test_check_raises_when_a_negative_page_pair_meets_a_positive_report_pair(
    tmp_path: Path,
) -> None:
    page = PAGE.replace("[0.03, 0.25]", "[-0.03, -0.25]")
    page_path, report_path = _write(directory=tmp_path, page=page)
    with pytest.raises(ValueError, match=r"\[-0\.03, -0\.25\]"):
        page_numbers.check_page_numbers(
            page_path=page_path, report_path=report_path, heading=HEADING
        )


def test_check_raises_when_a_positive_page_pair_meets_a_negative_report_pair(
    tmp_path: Path,
) -> None:
    """The report holds `[-0.310, -0.040]`; a page dropping both minus signs must not pass."""
    page = "## ECMWF wind\n\nENS beat HRES by 0.13 points [0.31, 0.04].\n"
    page_path, report_path = _write(directory=tmp_path, page=page)
    with pytest.raises(ValueError, match=r"\[0\.31, 0\.04\]"):
        page_numbers.check_page_numbers(
            page_path=page_path, report_path=report_path, heading=HEADING
        )


def test_check_accepts_a_plus_sign_and_signed_negative_pairs(tmp_path: Path) -> None:
    page = "## ECMWF wind\n\nHRES was [+0.03, +0.25] and ENS [-0.31, -0.04] against it.\n"
    page_path, report_path = _write(directory=tmp_path, page=page)
    assert page_numbers.check_page_numbers(
        page_path=page_path, report_path=report_path, heading=HEADING
    )


def test_check_raises_when_a_bare_minus_meets_only_positive_report_numbers(
    tmp_path: Path,
) -> None:
    page_path, report_path = _write(
        directory=tmp_path, page="## ECMWF wind\n\nHRES changed by -0.14 points.\n"
    )
    with pytest.raises(ValueError, match=r"-0\.14"):
        page_numbers.check_page_numbers(
            page_path=page_path, report_path=report_path, heading=HEADING
        )


def test_bare_magnitude_audit_lists_the_sign_of_the_report_number_matched() -> None:
    """A bare 0.13 matches the report's -0.125 only, so its listed sign is negative."""
    section = "The gap was 0.13 points, HRES's was 0.14 points, and UKV's error 7.39."
    audit = page_numbers.bare_magnitude_audit(section=section, report_text=REPORT)
    assert audit == [
        "0.13: report signs -",
        "0.14: report signs +",
        "7.39: report signs +",
    ]


def test_check_reads_only_the_named_section(tmp_path: Path) -> None:
    """The earlier and later sections hold numbers the report lacks, and must not be read."""
    page_path, report_path = _write(directory=tmp_path, page=PAGE)
    section = page_numbers.section_text(page_text=PAGE, heading=HEADING)
    assert "9.99" not in section
    assert "5.55" not in section
    assert "0.13 points" in section
    assert page_numbers.check_page_numbers(
        page_path=page_path, report_path=report_path, heading=HEADING
    )


def test_check_rounds_a_half_up_from_the_printed_digits(tmp_path: Path) -> None:
    """The report prints -0.125, so a page rounding it to two places writes 0.13, not 0.12."""
    good_page = "## ECMWF wind\n\nThe gap was 0.13 points.\n"
    bad_page = "## ECMWF wind\n\nThe gap was 0.12 points.\n"
    page_path, report_path = _write(directory=tmp_path, page=good_page)
    assert page_numbers.check_page_numbers(
        page_path=page_path, report_path=report_path, heading=HEADING
    )
    page_path.write_text(bad_page)
    with pytest.raises(ValueError, match=r"0\.12"):
        page_numbers.check_page_numbers(
            page_path=page_path, report_path=report_path, heading=HEADING
        )


def test_check_raises_on_a_missing_or_repeated_heading(tmp_path: Path) -> None:
    page_path, report_path = _write(directory=tmp_path, page=PAGE)
    with pytest.raises(ValueError, match="appears 0 times"):
        page_numbers.check_page_numbers(
            page_path=page_path, report_path=report_path, heading="## Absent"
        )
    page_path.write_text(PAGE + "\n## ECMWF wind\n\nAgain 1.0.\n")
    with pytest.raises(ValueError, match="appears 2 times"):
        page_numbers.check_page_numbers(
            page_path=page_path, report_path=report_path, heading=HEADING
        )


def test_check_raises_on_a_section_with_no_decimals(tmp_path: Path) -> None:
    page_path, report_path = _write(directory=tmp_path, page="## ECMWF wind\n\nNo numbers here.\n")
    with pytest.raises(ValueError, match="no decimal numbers"):
        page_numbers.check_page_numbers(
            page_path=page_path, report_path=report_path, heading=HEADING
        )


def test_check_raises_when_a_plus_sign_meets_only_negative_report_numbers(tmp_path: Path) -> None:
    """The report's only 0.13 is -0.125, so a page writing +0.13 has the direction wrong."""
    page_path, report_path = _write(
        directory=tmp_path, page="## ECMWF wind\n\nThe gap was +0.13 points.\n"
    )
    with pytest.raises(ValueError, match=r"\+0\.13"):
        page_numbers.check_page_numbers(
            page_path=page_path, report_path=report_path, heading=HEADING
        )


def test_check_raises_when_a_value_and_an_interval_come_from_different_report_rows(
    tmp_path: Path,
) -> None:
    """0.14 is the first row's value and [-0.31, -0.04] is the second row's interval."""
    page_path, report_path = _write(
        directory=tmp_path, page="## ECMWF wind\n\nHRES was 0.14 points [-0.31, -0.04] worse.\n"
    )
    with pytest.raises(ValueError, match=r"0\.14 \[-0\.31, -0\.04\]"):
        page_numbers.check_page_numbers(
            page_path=page_path, report_path=report_path, heading=HEADING
        )


def test_check_raises_when_the_sign_of_a_triples_value_disagrees(tmp_path: Path) -> None:
    page_path, report_path = _write(
        directory=tmp_path, page="## ECMWF wind\n\nHRES changed by -0.14 points [0.03, 0.25].\n"
    )
    with pytest.raises(ValueError, match=r"-0\.14 \[0\.03, 0\.25\]"):
        page_numbers.check_page_numbers(
            page_path=page_path, report_path=report_path, heading=HEADING
        )


@pytest.mark.parametrize(
    "triple",
    [
        "0.14 points [0.03, 0.25]",
        "+0.14 [0.03, 0.25]",
        "0.14 pp [0.03, 0.25]",
        "0.14% [0.03, 0.25]",
    ],
)
def test_a_triple_may_name_its_unit_or_carry_a_sign(tmp_path: Path, triple: str) -> None:
    page_path, report_path = _write(
        directory=tmp_path, page=f"## ECMWF wind\n\nHRES was {triple}.\n"
    )
    assert (
        page_numbers.check_page_numbers(
            page_path=page_path, report_path=report_path, heading=HEADING
        )
        == 1
    )


def test_a_table_row_value_and_interval_in_adjacent_cells_form_a_triple(tmp_path: Path) -> None:
    page = "## ECMWF wind\n\n| HRES | 0.14 | [0.03, 0.25] |\n"
    page_path, report_path = _write(directory=tmp_path, page=page)
    assert (
        page_numbers.check_page_numbers(
            page_path=page_path, report_path=report_path, heading=HEADING
        )
        == 1
    )


def test_a_page_number_rounds_the_full_precision_value_not_the_four_decimal_print(
    tmp_path: Path,
) -> None:
    """0.20496 prints as 0.2050, which a second half-up rounding would take to 0.21."""
    report = "| all | a − b | +0.2050 | [+0.2050, +0.2050] |\n"
    intervals = _write_intervals(directory=tmp_path, values=[0.20496])
    page_path, report_path = _write(
        directory=tmp_path, page="## ECMWF wind\n\nThe gap was 0.20 points.\n", report=report
    )
    assert page_numbers.check_page_numbers(
        page_path=page_path, report_path=report_path, heading=HEADING, intervals_path=intervals
    )
    page_path.write_text("## ECMWF wind\n\nThe gap was 0.21 points.\n")
    with pytest.raises(ValueError, match=r"0\.21"):
        page_numbers.check_page_numbers(
            page_path=page_path, report_path=report_path, heading=HEADING, intervals_path=intervals
        )


def test_a_report_number_with_no_interval_row_rounds_from_its_printed_digits(
    tmp_path: Path,
) -> None:
    report = "Ratio 0.9650 to ERA5.\n"
    intervals = _write_intervals(directory=tmp_path, values=[0.1234])
    page_path, report_path = _write(
        directory=tmp_path, page="## ECMWF wind\n\nThe ratio was 0.97.\n", report=report
    )
    assert page_numbers.check_page_numbers(
        page_path=page_path, report_path=report_path, heading=HEADING, intervals_path=intervals
    )


def test_bullet_text_returns_the_one_item_that_starts_with_the_prefix() -> None:
    section = "\n- **First.** One 1.11.\n  Second line 2.22.\n- **Other.** Three 3.33.\n"
    assert (
        page_numbers.bullet_text(section=section, prefix="- **First")
        == "- **First.** One 1.11.\n  Second line 2.22."
    )


@pytest.mark.parametrize("prefix", ["- **Absent", "- **"])
def test_bullet_text_raises_unless_exactly_one_item_matches(prefix: str) -> None:
    section = "- **First.** One.\n- **Other.** Two.\n"
    with pytest.raises(ValueError, match="expected exactly one"):
        page_numbers.bullet_text(section=section, prefix=prefix)


def test_check_reads_only_the_bullet_it_is_given(tmp_path: Path) -> None:
    page = "## ECMWF wind\n\n- **Yes.** HRES was 0.14 points.\n- **No.** Not 8.88.\n"
    page_path, report_path = _write(directory=tmp_path, page=page)
    assert (
        page_numbers.check_page_numbers(
            page_path=page_path,
            report_path=report_path,
            heading=HEADING,
            bullet_prefix="- **Yes",
        )
        == 1
    )


def test_a_half_is_rounded_from_decimal_digits_not_from_a_binary_float(tmp_path: Path) -> None:
    """0.145 is 0.1449999... as a float, so float rounding would give 0.14, not 0.15."""
    page_path, report_path = _write(
        directory=tmp_path,
        page="## ECMWF wind\n\nThe gap was 0.15 points.\n",
        report="| +0.145 |\n",
    )
    assert page_numbers.check_page_numbers(
        page_path=page_path, report_path=report_path, heading=HEADING
    )


def test_the_low_end_of_a_pair_is_checked(tmp_path: Path) -> None:
    page_path, report_path = _write(
        directory=tmp_path, page=PAGE.replace("[0.03, 0.25]", "[0.04, 0.25]")
    )
    with pytest.raises(ValueError, match=r"0\.14 \[0\.04, 0\.25\]"):
        page_numbers.check_page_numbers(
            page_path=page_path, report_path=report_path, heading=HEADING
        )


@pytest.mark.parametrize("pair", ["[0.03, 8.12]", "[0.25, 0.03]"])
def test_both_ends_of_a_pair_must_come_from_one_report_pair_in_order(
    tmp_path: Path, pair: str
) -> None:
    page_path, report_path = _write(directory=tmp_path, page=f"## ECMWF wind\n\nHRES was {pair}.\n")
    with pytest.raises(ValueError, match=r"not in report"):
        page_numbers.check_page_numbers(
            page_path=page_path, report_path=report_path, heading=HEADING
        )


@pytest.mark.parametrize("minus", ["−", "–"])
def test_a_unicode_minus_or_en_dash_on_the_page_is_a_minus_sign(tmp_path: Path, minus: str) -> None:
    page = f"## ECMWF wind\n\nHRES was [{minus}0.03, {minus}0.25].\n"
    page_path, report_path = _write(directory=tmp_path, page=page)
    with pytest.raises(ValueError, match=r"not in report"):
        page_numbers.check_page_numbers(
            page_path=page_path, report_path=report_path, heading=HEADING
        )


@pytest.mark.parametrize("minus", ["−", "–"])
def test_a_unicode_minus_or_en_dash_in_the_report_is_a_minus_sign(
    tmp_path: Path, minus: str
) -> None:
    page_path, report_path = _write(
        directory=tmp_path,
        page="## ECMWF wind\n\nENS moved by -0.13 points.\n",
        report=f"| {minus}0.125 |\n",
    )
    assert page_numbers.check_page_numbers(
        page_path=page_path, report_path=report_path, heading=HEADING
    )


def test_a_triple_whose_ends_round_to_zero_meets_either_sign(tmp_path: Path) -> None:
    page = "## ECMWF wind\n\nENS moved by -0.00 points [0.00, 0.04].\n"
    page_path, report_path = _write(
        directory=tmp_path, page=page, report="| +0.003 | [-0.004, +0.040] |\n"
    )
    assert (
        page_numbers.check_page_numbers(
            page_path=page_path, report_path=report_path, heading=HEADING
        )
        == 1
    )


def test_a_bare_minus_that_rounds_to_zero_meets_a_positive_report_number(tmp_path: Path) -> None:
    page_path, report_path = _write(
        directory=tmp_path,
        page="## ECMWF wind\n\nENS moved by -0.00 points.\n",
        report="| +0.003 |\n",
    )
    assert page_numbers.check_page_numbers(
        page_path=page_path, report_path=report_path, heading=HEADING
    )


def test_a_degree_sign_and_a_decimal_height_are_excluded(tmp_path: Path) -> None:
    page = "## ECMWF wind\n\nOn the 0.25° grid at 9.87 m, HRES was 0.14 points worse.\n"
    page_path, report_path = _write(directory=tmp_path, page=page)
    assert (
        page_numbers.check_page_numbers(
            page_path=page_path, report_path=report_path, heading=HEADING
        )
        == 1
    )


def test_a_higher_rank_heading_ends_the_section() -> None:
    page = "## Top\n\n### ECMWF wind\n\n0.14\n\n## Next\n\n5.55\n"
    assert "5.55" not in page_numbers.section_text(page_text=page, heading="### ECMWF wind")


def test_a_line_starting_with_a_hash_but_no_space_is_not_a_heading() -> None:
    page = "## ECMWF wind\n\nSee issue\n#841](https://example.com) for 0.14 points.\n"
    assert "0.14" in page_numbers.section_text(page_text=page, heading=HEADING)


def test_a_heading_that_extends_the_named_one_is_not_matched(tmp_path: Path) -> None:
    page_path, report_path = _write(
        directory=tmp_path, page=PAGE + "\n## ECMWF wind speeds\n\n9.99\n"
    )
    assert (
        page_numbers.check_page_numbers(
            page_path=page_path, report_path=report_path, heading=HEADING
        )
        == 4
    )


def test_check_logs_the_audit_list(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    page_path, report_path = _write(
        directory=tmp_path, page="## ECMWF wind\n\nHRES was 0.14 points worse.\n"
    )
    with caplog.at_level(logging.INFO):
        page_numbers.check_page_numbers(
            page_path=page_path, report_path=report_path, heading=HEADING
        )
    assert "0.14: report signs +" in caplog.text


def test_a_number_matching_both_signs_lists_both_signs() -> None:
    audit = page_numbers.bare_magnitude_audit(section="gap 0.14", report_text="+0.142 and -0.138")
    assert audit == ["0.14: report signs +, -"]


def test_the_audit_skips_exclusions_pairs_and_triples() -> None:
    audit = page_numbers.bare_magnitude_audit(
        section="At 0.25 degrees HRES was 0.14 worse, and 0.14 points [0.03, 0.25].",
        report_text=REPORT,
    )
    assert audit == ["0.14: report signs +"]


@pytest.mark.parametrize("printed", ["7.4", "0.142"])
def test_the_precision_follows_the_page(tmp_path: Path, printed: str) -> None:
    page_path, report_path = _write(
        directory=tmp_path, page=f"## ECMWF wind\n\nIt was {printed}.\n"
    )
    assert page_numbers.check_page_numbers(
        page_path=page_path, report_path=report_path, heading=HEADING
    )


def test_a_section_with_no_numbers_passes_only_when_empty_is_allowed(tmp_path: Path) -> None:
    page_path, report_path = _write(directory=tmp_path, page="## ECMWF wind\n\nNo numbers.\n")
    assert (
        page_numbers.check_page_numbers(
            page_path=page_path, report_path=report_path, heading=HEADING, allow_empty=True
        )
        == 0
    )


STATION_REPORT: Final[str] = "| `ukv_station_wind` | 7 | +0.995 | [+0.585, +1.452] | 34,156 |\n"
"""A station-report row: signed, three decimals."""

STATION_INTERVAL: Final[list[float]] = [0.994766, 0.58524, 1.45206]
"""The full-precision value, lower and upper end behind `STATION_REPORT`'s row."""


def _check_station_page(*, directory: Path, page_number: str) -> int:
    """Check a one-triple page against `STATION_REPORT` and its `intervals.parquet`.

    Args:
        directory: Where to write the files.
        page_number: The triple as the page quotes it, such as `0.99 [0.59, 1.45]`.

    Returns:
        How many numbers were checked.
    """
    value, lower, upper = STATION_INTERVAL
    intervals = _write_intervals(directory=directory, values=[value])
    pl.DataFrame({"value": [value], "lower": [lower], "upper": [upper]}).write_parquet(intervals)
    page_path, report_path = _write(
        directory=directory,
        page=f"## ECMWF wind\n\nThe gap was {page_number}.\n",
        report=STATION_REPORT,
    )
    return page_numbers.check_page_numbers(
        page_path=page_path, report_path=report_path, heading=HEADING, intervals_path=intervals
    )


def test_a_three_decimal_signed_report_row_matches_the_two_decimal_unsigned_page(
    tmp_path: Path,
) -> None:
    """0.995 half-up rounds to 1.00, so 0.99 matches only through the value 0.994766."""
    assert _check_station_page(directory=tmp_path, page_number="0.99 [0.59, 1.45]") == 1


@pytest.mark.parametrize(
    "page_number", ["1.00 [0.59, 1.45]", "1.00 [0.58, 1.45]", "0.98 [0.59, 1.45]"]
)
def test_a_three_decimal_report_row_rejects_a_page_number_that_is_0_01_off(
    tmp_path: Path, page_number: str
) -> None:
    with pytest.raises(ValueError, match=r"not in report\.md"):
        _check_station_page(directory=tmp_path, page_number=page_number)


def test_print_decimals_are_read_from_the_report_brackets() -> None:
    assert page_numbers.interval_print_decimals(report_text=STATION_REPORT) == {3}
    assert page_numbers.interval_print_decimals(report_text=REPORT) == {3}
    assert page_numbers.interval_print_decimals(report_text="| +0.2050 | [+0.2050, +0.2050] |") == {
        4
    }
    assert page_numbers.interval_print_decimals(report_text="no brackets 1.23") == {4}
