"""Check that every decimal number and `[a, b]` interval on a docs page is in a study's `report.md`.

A thin command-line caller of `studies.page_numbers`, which owns the rules: what is extracted from
the page, how signs and rounding are matched, and how the report's print precision is read.
"""

import argparse
import logging
from pathlib import Path

from studies.page_numbers import check_page_numbers

_LOG = logging.getLogger(__name__)


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
