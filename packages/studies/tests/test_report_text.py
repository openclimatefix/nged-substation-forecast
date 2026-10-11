"""Tests for `studies.report_text`, each written to fail on the defect it names."""

from pathlib import Path

import pytest
from studies.report_text import (
    BOOT_PATTERN,
    T_PATTERN,
    Estimate,
    ReportText,
    estimate_in,
)

REPORT = """# Report

## First section

- primary: -0.081 points; month t-interval [-0.216, 0.015]; bootstrap 95% [-0.175, 0.011]
- sensitivity: 0.255 points; month t-interval [0.004, 0.563]; bootstrap 95% [0.040, 0.487]

| estimator | value | note |
|---|---|---|
| KF | 1.5000 | None |
| HM | 2.0000 | text |

| other | table |
|---|---|
| x | 1 |

## Second section

- primary: 9.000 points; month t-interval [9.0, 9.1]
### Nested heading

- inside: 1
"""


@pytest.fixture
def report(tmp_path: Path) -> ReportText:
    path = tmp_path / "report.md"
    path.write_text(REPORT)
    return ReportText(path=path)


def test_line_returns_only_the_line_under_its_own_heading(report: ReportText):
    first = report.line(heading="First section", prefix="- primary")
    second = report.line(heading="Second section", prefix="- primary")

    assert "-0.081" in first
    assert "9.000" in second


def test_section_stops_at_a_nested_heading(report: ReportText):
    assert not any("inside" in line for line in report.section(heading="Second section"))
    assert report.line(heading="Nested heading", prefix="- inside") == "- inside: 1"


def test_a_missing_or_repeated_heading_raises(report: ReportText):
    with pytest.raises(ValueError, match="found 0 times"):
        report.section(heading="No such section")
    report.lines += ["## First section"]
    with pytest.raises(ValueError, match="found 2 times"):
        report.section(heading="First section")


def test_a_line_matching_zero_or_two_prefixes_raises(report: ReportText):
    with pytest.raises(ValueError, match="0 lines"):
        report.line(heading="First section", prefix="- tertiary")
    with pytest.raises(ValueError, match="2 lines"):
        report.line(heading="First section", prefix="- ")


def test_table_reads_only_the_first_table_and_types_its_cells(report: ReportText):
    table = report.table(heading="First section")

    assert table.columns == ["estimator", "value", "note"]
    assert table["value"].to_list() == [1.5, 2.0]
    assert table["note"].to_list() == [None, "text"]
    assert table.height == 2


def test_table_raises_when_a_section_holds_none(report: ReportText):
    with pytest.raises(ValueError, match="no table"):
        report.table(heading="Second section")


def test_estimate_in_reads_the_t_interval_not_the_bootstrap_one(report: ReportText):
    line = report.line(heading="First section", prefix="- primary")

    assert estimate_in(line=line, pattern=T_PATTERN) == Estimate(-0.081, -0.216, 0.015)


def test_estimate_in_raises_when_the_pattern_is_absent():
    with pytest.raises(ValueError, match="does not match"):
        estimate_in(line="- primary: 0.1 points", pattern=BOOT_PATTERN)


def test_estimate_in_keeps_the_sign_of_each_end():
    found = estimate_in(
        line="- x minus y: -1.467 points, bootstrap [-2.084, -0.854]", pattern=BOOT_PATTERN
    )

    assert found.point < 0
    assert found.low < found.point < found.high < 0
