import zipfile
from collections.abc import Mapping
from pathlib import Path
from xml.sax.saxutils import escape

import ecr
import polars as pl
import pytest

SHEET_XML = (
    '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
    "<sheetData>{rows}</sheetData></worksheet>"
)
HEADINGS = dict(
    zip(
        ecr.COLUMN_NAMES,
        [chr(ord("A") + i) if i < 26 else "A" + chr(ord("A") + i - 26) for i in range(40)],
        strict=False,
    )
)


def _cell(*, reference: str, text: str | None) -> str:
    if text is None:
        return f'<c r="{reference}" t="n"></c>'
    return f'<c r="{reference}" t="inlineStr"><is><t>{text}</t></is></c>'


def _row(*, number: int, values: Mapping[str, str | None]) -> str:
    cells = "".join(_cell(reference=f"{col}{number}", text=text) for col, text in values.items())
    return f'<row r="{number}">{cells}</row>'


def _write_workbook(
    *,
    path: Path,
    sheets: dict[str, list[dict[str, str | None]]],
    heading_overrides: dict[str, str] | None = None,
) -> None:
    """Write a workbook whose sheets each have a banner row, a heading row, and data rows."""
    headings = {letter: heading for heading, letter in HEADINGS.items()}
    headings.update(heading_overrides or {})
    sheet_entries = ""
    relationships = ""
    with zipfile.ZipFile(path, "w") as archive:
        for index, (name, rows) in enumerate(sheets.items(), start=1):
            sheet_entries += f'<sheet name="{escape(name)}" sheetId="{index}" r:id="rId{index}"/>'
            relationships += (
                f'<Relationship Id="rId{index}" Target="/xl/worksheets/sheet{index}.xml"/>'
            )
            body = _row(number=1, values={"A": "General Data"})
            body += _row(number=2, values=headings)
            for offset, values in enumerate(rows, start=3):
                body += _row(number=offset, values=values)
            archive.writestr(f"xl/worksheets/sheet{index}.xml", SHEET_XML.format(rows=body))
        archive.writestr(
            "xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f"<sheets>{sheet_entries}</sheets></workbook>",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f"{relationships}</Relationships>",
        )


def _values(**by_name: str | None) -> dict[str, str | None]:
    return {HEADINGS[name]: value for name, value in by_name.items()}


HEADINGS_INDEX_OF_CUSTOMER_NAME = list(ecr.COLUMN_NAMES).index("Customer Name")


def test_a_cell_reference_maps_to_its_zero_based_column() -> None:
    assert ecr._column_index(reference="A3") == 0
    assert ecr._column_index(reference="Z1") == 25
    assert ecr._column_index(reference="AB12") == 27
    assert ecr._column_index(reference="BG8000") == 58


def test_read_sheet_collapses_a_heading_with_a_line_break_and_keeps_empty_cells_as_null(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ecr.xlsx"
    _write_workbook(
        path=path,
        sheets={"One": [_values(**{"Customer Name": "Acme", "Reference": None})]},
        heading_overrides={HEADINGS["Customer Name"]: "Customer\n  Name "},
    )

    frame = ecr.read_sheet(path=path, sheet_name="One")

    assert frame["Customer Name"].to_list() == ["Acme"]
    assert frame["Reference"].to_list() == [None]


def test_read_sheet_removes_the_space_after_a_colon_in_a_heading(tmp_path: Path) -> None:
    path = tmp_path / "ecr.xlsx"
    heading = "Location (X-coordinate):\nEastings (where data is held)"
    _write_workbook(
        path=path,
        sheets={"One": [_values(**{"Customer Name": "Acme"})]},
        heading_overrides={HEADINGS["Customer Name"]: heading},
    )

    frame = ecr.read_sheet(path=path, sheet_name="One")

    assert frame.columns[HEADINGS_INDEX_OF_CUSTOMER_NAME] == (
        "Location (X-coordinate):Eastings (where data is held)"
    )


def test_read_sheet_drops_rows_with_every_cell_empty(tmp_path: Path) -> None:
    path = tmp_path / "ecr.xlsx"
    _write_workbook(
        path=path,
        sheets={"One": [_values(**{"Customer Name": "Acme"}), _values(**{"Customer Name": None})]},
    )

    assert ecr.read_sheet(path=path, sheet_name="One").height == 1


def test_read_sheet_raises_for_a_missing_sheet(tmp_path: Path) -> None:
    path = tmp_path / "ecr.xlsx"
    _write_workbook(path=path, sheets={"One": []})

    with pytest.raises(KeyError):
        ecr.read_sheet(path=path, sheet_name="Two")


def test_read_ecr_joins_both_sheets_and_types_the_numbers(tmp_path: Path) -> None:
    path = tmp_path / "ecr.xlsx"
    small, large = ecr.REGISTER_SHEETS
    _write_workbook(
        path=path,
        sheets={
            small: [
                _values(
                    **{
                        "Maximum Export Capacity (MW)": "0.25",
                        "Maximum Import Capacity (MW)": "data not available",
                        "Connection Status": "Connected",
                    }
                )
            ],
            large: [_values(**{"Maximum Export Capacity (MW)": "12.5"})],
        },
    )

    frame = ecr.read_ecr(path=path)

    assert frame["sheet"].to_list() == [small, large]
    assert frame["maximum_export_capacity_mw"].to_list() == [0.25, 12.5]
    assert frame["maximum_import_capacity_mw"].to_list() == [None, None]
    assert frame.schema["maximum_export_capacity_mw"] == pl.Float64
