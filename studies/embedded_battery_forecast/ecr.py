"""Read NGED's Embedded Capacity Register (ECR) workbook without a spreadsheet library.

The workbook's generation and storage rows sit in two sheets, `Register Part 1 50kW - <1MW` and
`Register Part 1 - ≥1MW`, with a banner on the first row and the column headings on the second. The
file is written with inline strings, so the standard library's XML reader is enough and the project
needs no spreadsheet dependency. Every cell is read as text; `read_ecr` types the columns the census
uses.
"""

import re
import zipfile
from pathlib import Path
from typing import Final
from xml.etree import ElementTree

import polars as pl
from studies.sources import REPO_DATA_DIR

ECR_PATH: Final[Path] = REPO_DATA_DIR.parent / "literature" / "NGED" / "NGED ECR AUG 2026.xlsx"
"""The August 2026 release, kept outside the repository."""

REGISTER_SHEETS: Final[tuple[str, ...]] = (
    "Register Part 1 50kW - <1MW",
    "Register Part 1 - ≥1MW",
)
"""The two sheets that list generation and storage."""

HEADER_ROW: Final[int] = 2
"""The row holding the column headings, counting from 1."""

_SPREADSHEET_NS: Final[str] = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_RELATIONSHIP_NS: Final[str] = (
    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
)
_PACKAGE_RELATIONSHIP_NS: Final[str] = (
    "{http://schemas.openxmlformats.org/package/2006/relationships}"
)

COLUMN_NAMES: Final[dict[str, str]] = {
    "Export MPAN / MSID": "export_mpan",
    "Import MPAN / MSID": "import_mpan",
    "Customer Name": "customer_name",
    "Customer Site": "customer_site",
    "Address Line 1": "address_line_1",
    "Address Line 2": "address_line_2",
    "Town/ City": "town",
    "County": "county",
    "Postcode": "postcode",
    "Country": "country",
    "Location (X-coordinate):Eastings (where data is held)": "easting",
    "Location (y-coordinate):Northings (where data is held)": "northing",
    "Grid Supply Point": "grid_supply_point",
    "Bulk Supply Point": "bulk_supply_point",
    "Primary": "primary_substation",
    "Point of Connection (POC) Voltage (kV)": "connection_voltage_kv",
    "Licence Area": "licence_area",
    "Energy Source 1": "energy_source_1",
    "Energy Conversion Technology 1": "technology_1",
    "Energy Source 2": "energy_source_2",
    "Energy Conversion Technology 2": "technology_2",
    "Energy Source 3": "energy_source_3",
    "Energy Conversion Technology 3": "technology_3",
    "Connection Status": "connection_status",
    "Already connected Registered Capacity (MW)": "connected_registered_capacity_mw",
    "Maximum Export Capacity (MW)": "maximum_export_capacity_mw",
    "Maximum Import Capacity (MW)": "maximum_import_capacity_mw",
    "Accepted to Connect Registered Capacity (MW)": "accepted_registered_capacity_mw",
    "Reference": "reference",
    "Last Updated": "last_updated",
}
"""The ECR headings the census uses, after whitespace is collapsed, and the names it gives them."""

NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "easting",
    "northing",
    "connected_registered_capacity_mw",
    "maximum_export_capacity_mw",
    "maximum_import_capacity_mw",
    "accepted_registered_capacity_mw",
)
"""The columns `read_ecr` turns into numbers. A cell reading `data not available` becomes null."""


def _column_index(*, reference: str) -> int:
    """Return the zero-based column of a cell reference such as `AB12`."""
    letters = re.match(r"[A-Z]+", reference)
    assert letters is not None
    index = 0
    for letter in letters.group():
        index = index * 26 + (ord(letter) - ord("A") + 1)
    return index - 1


def _sheet_part(*, archive: zipfile.ZipFile, sheet_name: str) -> str:
    """Return the path inside the workbook archive of the sheet with the given name.

    Args:
        archive: The open workbook.
        sheet_name: The sheet's name.

    Returns:
        The path of the sheet's XML part, such as `xl/worksheets/sheet3.xml`.

    Raises:
        KeyError: If the workbook has no sheet of that name.
    """
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    relationship_id = None
    for sheet in workbook.iter(f"{_SPREADSHEET_NS}sheet"):
        if sheet.get("name") == sheet_name:
            relationship_id = sheet.get(f"{_RELATIONSHIP_NS}id")
    if relationship_id is None:
        raise KeyError(sheet_name)
    relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    for relationship in relationships.iter(f"{_PACKAGE_RELATIONSHIP_NS}Relationship"):
        if relationship.get("Id") == relationship_id:
            return "xl/" + str(relationship.get("Target")).removeprefix("/xl/")
    raise KeyError(relationship_id)


def _cell_text(*, cell: ElementTree.Element) -> str | None:
    """Return a cell's text, or None for an empty cell.

    Args:
        cell: A worksheet `c` element.

    Returns:
        The cell's inline string or stored value, or None if it holds neither.

    Raises:
        ValueError: If the cell holds a shared-string index. The reader supports only inline
            strings, and an index would otherwise be read as a number and match nothing.
    """
    if cell.get("t") == "s":
        msg = "The workbook uses shared strings, which this reader does not support."
        raise ValueError(msg)
    if cell.get("t") == "inlineStr":
        return "".join(node.text or "" for node in cell.iter(f"{_SPREADSHEET_NS}t"))
    value = cell.find(f"{_SPREADSHEET_NS}v")
    return None if value is None else value.text


def read_sheet(*, path: Path, sheet_name: str) -> pl.DataFrame:
    """Read one register sheet as text, with its collapsed headings as column names.

    Args:
        path: The workbook.
        sheet_name: The sheet's name.

    Returns:
        One String column per heading, one row per register row. A row with every cell empty is
        dropped. Headings have runs of whitespace collapsed to one space and no space after a colon.
    """
    header: dict[int, str] = {}
    rows: list[dict[int, str | None]] = []
    with zipfile.ZipFile(path) as archive:
        part = _sheet_part(archive=archive, sheet_name=sheet_name)
        with archive.open(part) as stream:
            for _, element in ElementTree.iterparse(stream):
                if element.tag != f"{_SPREADSHEET_NS}row":
                    continue
                row_number = int(element.get("r", "0"))
                cells = {
                    _column_index(reference=str(cell.get("r"))): _cell_text(cell=cell)
                    for cell in element.findall(f"{_SPREADSHEET_NS}c")
                }
                if row_number == HEADER_ROW:
                    header = {
                        i: " ".join(t.split()).replace(": ", ":") for i, t in cells.items() if t
                    }
                elif row_number > HEADER_ROW and any(t not in (None, "") for t in cells.values()):
                    rows.append(cells)
                element.clear()
    return pl.DataFrame(
        {name: [row.get(i) for row in rows] for i, name in header.items()},
        schema=dict.fromkeys(header.values(), pl.String),
    )


def read_ecr(*, path: Path = ECR_PATH) -> pl.DataFrame:
    """Read the two Part 1 sheets into one typed frame.

    Args:
        path: The workbook.

    Returns:
        One row per register row, with the columns in `COLUMN_NAMES` (as text, except
        `NUMERIC_COLUMNS`) and `sheet`, the name of the sheet the row came from.
    """
    frames = []
    for sheet_name in REGISTER_SHEETS:
        sheet = read_sheet(path=path, sheet_name=sheet_name)
        selected = sheet.select(list(COLUMN_NAMES)).rename(COLUMN_NAMES)
        frames.append(selected.with_columns(sheet=pl.lit(sheet_name)))
    combined = pl.concat(frames)
    return combined.with_columns(
        pl.col(column).cast(pl.Float64, strict=False) for column in NUMERIC_COLUMNS
    )
