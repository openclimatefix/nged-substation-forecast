import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Final
from xml.sax.saxutils import escape

import polars as pl
import pytest

from studies import ecr_reader

SHEET_NAME: Final[str] = "Register Part 1 test"
SECOND_SHEET_NAME: Final[str] = "Register Part 2 test"
MAIN_NS: Final[str] = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
RELATIONSHIP_NS: Final[str] = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_RELATIONSHIP_NS: Final[str] = "http://schemas.openxmlformats.org/package/2006/relationships"

# The header has a run of spaces, a newline and a space after a colon, all of which are collapsed.
HEADER: list[str] = ["Customer   Name", "Location (X-coordinate):\nEastings", "Reference"]


def _inline(*, reference: str, text: str) -> str:
    return f'<c r="{reference}" t="inlineStr"><is><t>{escape(text)}</t></is></c>'


def _row(*, number: int, cells: Sequence[str | None], cell_type: str | None = None) -> str:
    rendered = []
    for column, text in zip("ABC", cells, strict=False):
        if text is None:
            continue
        if cell_type is None:
            rendered.append(_inline(reference=f"{column}{number}", text=text))
        else:
            rendered.append(f'<c r="{column}{number}" t="{cell_type}"><v>{text}</v></c>')
    return f'<row r="{number}">{"".join(rendered)}</row>'


def _sheet_xml(*, rows: list[str]) -> str:
    return f'<worksheet xmlns="{MAIN_NS}"><sheetData>{"".join(rows)}</sheetData></worksheet>'


def _write_workbook(
    *, path: Path, sheets: dict[str, str], target_prefix: str = "worksheets/"
) -> Path:
    """Write a workbook with one worksheet part per entry of `sheets` (name -> sheet XML)."""
    declarations = "".join(
        f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>'
        for i, name in enumerate(sheets, start=1)
    )
    relationships = "".join(
        f'<Relationship Id="rId{i}" Target="{target_prefix}sheet{i}.xml"/>'
        for i in range(1, len(sheets) + 1)
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{MAIN_NS}" xmlns:r="{RELATIONSHIP_NS}">'
            f"<sheets>{declarations}</sheets></workbook>",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            f'<Relationships xmlns="{PACKAGE_RELATIONSHIP_NS}">{relationships}</Relationships>',
        )
        for i, xml in enumerate(sheets.values(), start=1):
            archive.writestr(f"xl/worksheets/sheet{i}.xml", xml)
    return path


def _standard_rows(*, sheet_tag: str) -> list[str]:
    return [
        _row(number=1, cells=["Banner text"]),
        _row(number=2, cells=HEADER),
        _row(number=3, cells=[f"Alpha {sheet_tag}", "123.5", "R1"]),
        _row(number=4, cells=[None, None, None]),
        _row(number=5, cells=[f"Beta {sheet_tag}", "data not available", "R2"]),
    ]


@pytest.fixture
def workbook(tmp_path: Path) -> Path:
    return _write_workbook(
        path=tmp_path / "ecr.xlsx",
        sheets={
            SHEET_NAME: _sheet_xml(rows=_standard_rows(sheet_tag="one")),
            SECOND_SHEET_NAME: _sheet_xml(rows=_standard_rows(sheet_tag="two")),
        },
    )


def test_read_sheet_collapses_header_whitespace_and_drops_the_empty_row(workbook: Path):
    sheet = ecr_reader.read_sheet(path=workbook, sheet_name=SHEET_NAME)

    assert sheet.columns == ["Customer Name", "Location (X-coordinate):Eastings", "Reference"]
    assert sheet.to_dicts() == [
        {
            "Customer Name": "Alpha one",
            "Location (X-coordinate):Eastings": "123.5",
            "Reference": "R1",
        },
        {
            "Customer Name": "Beta one",
            "Location (X-coordinate):Eastings": "data not available",
            "Reference": "R2",
        },
    ]
    assert set(sheet.dtypes) == {pl.String}


def test_read_sheet_with_a_missing_sheet_raises(workbook: Path):
    with pytest.raises(KeyError, match="No such sheet"):
        ecr_reader.read_sheet(path=workbook, sheet_name="No such sheet")


def test_read_sheet_with_a_shared_string_cell_raises(tmp_path: Path):
    rows = [
        _row(number=1, cells=["Banner text"]),
        _row(number=2, cells=HEADER),
        _row(number=3, cells=["0", "1", "2"], cell_type="s"),
    ]
    path = _write_workbook(
        path=tmp_path / "shared.xlsx", sheets={SHEET_NAME: _sheet_xml(rows=rows)}
    )

    with pytest.raises(ValueError, match="shared strings"):
        ecr_reader.read_sheet(path=path, sheet_name=SHEET_NAME)


def test_read_ecr_types_the_numeric_columns_and_tags_each_row_with_its_sheet(
    workbook: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(ecr_reader, "REGISTER_SHEETS", (SHEET_NAME, SECOND_SHEET_NAME))
    monkeypatch.setattr(
        ecr_reader,
        "COLUMN_NAMES",
        {
            "Customer Name": "customer_name",
            "Location (X-coordinate):Eastings": "easting",
            "Reference": "reference",
        },
    )
    monkeypatch.setattr(ecr_reader, "NUMERIC_COLUMNS", ("easting",))

    register = ecr_reader.read_ecr(path=workbook)

    assert register.schema == {
        "customer_name": pl.String,
        "easting": pl.Float64,
        "reference": pl.String,
        "sheet": pl.String,
    }
    assert register["easting"].to_list() == [123.5, None, 123.5, None]
    assert register["sheet"].to_list() == [SHEET_NAME] * 2 + [SECOND_SHEET_NAME] * 2


def test_read_ecr_with_a_missing_sheet_raises(workbook: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ecr_reader, "REGISTER_SHEETS", (SHEET_NAME, "No such sheet"))
    monkeypatch.setattr(ecr_reader, "COLUMN_NAMES", {"Reference": "reference"})

    with pytest.raises(KeyError, match="No such sheet"):
        ecr_reader.read_ecr(path=workbook)


def _raw_row(*, number: int, cells: str) -> str:
    return f'<row r="{number}">{cells}</row>'


def _inline_cells(*, number: int, texts: dict[str, str]) -> str:
    return "".join(_inline(reference=f"{col}{number}", text=text) for col, text in texts.items())


def test_read_sheet_reads_numeric_cells_and_multi_letter_columns(tmp_path: Path):
    rows = [
        _raw_row(
            number=2, cells=_inline_cells(number=2, texts={"A": "Name", "B": "X", "AA": "Far"})
        ),
        _raw_row(
            number=3,
            cells=(
                _inline(reference="A3", text="Alpha")
                + '<c r="B3"><v>1.5</v></c>'
                + '<c r="AA3" t="n"><v>7</v></c>'
            ),
        ),
    ]
    path = _write_workbook(
        path=tmp_path / "numeric.xlsx", sheets={SHEET_NAME: _sheet_xml(rows=rows)}
    )

    sheet = ecr_reader.read_sheet(path=path, sheet_name=SHEET_NAME)

    assert sheet.to_dicts() == [{"Name": "Alpha", "X": "1.5", "Far": "7"}]


def test_read_sheet_leaves_a_gap_between_sparse_cells_null(tmp_path: Path):
    rows = [
        _raw_row(
            number=2, cells=_inline_cells(number=2, texts={"A": "One", "B": "Two", "C": "Three"})
        ),
        _raw_row(number=3, cells=_inline_cells(number=3, texts={"A": "a", "C": "c"})),
    ]
    path = _write_workbook(
        path=tmp_path / "sparse.xlsx", sheets={SHEET_NAME: _sheet_xml(rows=rows)}
    )

    sheet = ecr_reader.read_sheet(path=path, sheet_name=SHEET_NAME)

    assert sheet.to_dicts() == [{"One": "a", "Two": None, "Three": "c"}]


def test_read_sheet_drops_a_row_whose_cells_hold_only_empty_values(tmp_path: Path):
    rows = [
        _raw_row(number=2, cells=_inline_cells(number=2, texts={"A": "One", "B": "Two"})),
        _raw_row(number=3, cells='<c r="A3"><v/></c><c r="B3"><v></v></c>'),
        _raw_row(number=4, cells=_inline_cells(number=4, texts={"A": "kept"})),
    ]
    path = _write_workbook(path=tmp_path / "empty.xlsx", sheets={SHEET_NAME: _sheet_xml(rows=rows)})

    sheet = ecr_reader.read_sheet(path=path, sheet_name=SHEET_NAME)

    assert sheet.to_dicts() == [{"One": "kept", "Two": None}]


def test_read_sheet_follows_an_absolute_relationship_target(tmp_path: Path):
    rows = [
        _raw_row(number=2, cells=_inline_cells(number=2, texts={"A": "One"})),
        _raw_row(number=3, cells=_inline_cells(number=3, texts={"A": "a"})),
    ]
    path = _write_workbook(
        path=tmp_path / "absolute.xlsx",
        sheets={SHEET_NAME: _sheet_xml(rows=rows)},
        target_prefix="/xl/worksheets/",
    )

    sheet = ecr_reader.read_sheet(path=path, sheet_name=SHEET_NAME)

    assert sheet.to_dicts() == [{"One": "a"}]
