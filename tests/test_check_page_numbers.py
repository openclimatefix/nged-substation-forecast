"""Tests for the command line of `studies/beam_diffuse_split/check_page_numbers.py`.

The rules themselves are tested in `packages/studies/tests/test_page_numbers.py`; these tests cover
what the script adds: reading its arguments, finding `intervals.parquet` beside the report, and
raising on a mismatch.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

import polars as pl
import pytest

MODULE_PATH: Final[Path] = (
    Path(__file__).parent.parent / "studies" / "beam_diffuse_split" / "check_page_numbers.py"
)
HEADING: Final[str] = "## ECMWF wind"
REPORT: Final[str] = "| all | a − b | +0.2050 | [+0.2050, +0.2050] |\n"
PAGE: Final[str] = "## ECMWF wind\n\n- **Yes.** The gap was 0.20 points.\n"


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


def _write(*, directory: Path, page: str) -> tuple[Path, Path]:
    """Write a page, a report and an `intervals.parquet` holding 0.20496 into a directory.

    Args:
        directory: Where to write.
        page: The page text.

    Returns:
        The page path and the report path.
    """
    page_path = directory / "page.md"
    report_path = directory / "report.md"
    page_path.write_text(page)
    report_path.write_text(REPORT)
    values = [0.20496]
    pl.DataFrame({"value": values, "lower": values, "upper": values}).write_parquet(
        directory / "intervals.parquet"
    )
    return page_path, report_path


def test_main_checks_a_section_and_reads_the_intervals_beside_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page_path, report_path = _write(directory=tmp_path, page=PAGE)
    monkeypatch.setattr(
        sys, "argv", ["check", str(page_path), str(report_path), "--section", HEADING]
    )
    assert CHECK.main() == 0


def test_main_checks_a_bullet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    page_path, report_path = _write(directory=tmp_path, page=PAGE)
    monkeypatch.setattr(
        sys, "argv", ["check", str(page_path), str(report_path), "--bullet", HEADING, "- **Yes"]
    )
    assert CHECK.main() == 0


def test_main_raises_when_a_section_number_is_not_in_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page_path, report_path = _write(directory=tmp_path, page=PAGE.replace("0.20", "0.22"))
    monkeypatch.setattr(
        sys, "argv", ["check", str(page_path), str(report_path), "--section", HEADING]
    )
    with pytest.raises(ValueError, match=r"0\.22"):
        CHECK.main()
