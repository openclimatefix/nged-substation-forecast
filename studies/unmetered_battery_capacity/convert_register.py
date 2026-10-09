"""Convert the storage-relevant columns of NGED's Embedded Capacity Register to one parquet file.

Neither `openpyxl` nor `fastexcel` is installed, so this script streams the two register sheets
(`Register Part 1 50kW - <1MW` and `Register Part 1 - >=1MW`) out of the workbook's XML with the
standard library, keeping only the columns the unmetered-battery study reads. The register lists no
customer-identifying column in the output: only the primary substation's name, the energy source and
technology of each of the three possible units, the connection status, and the registered capacity.

Run: `uv run python studies/unmetered_battery_capacity/convert_register.py`
Writes `data/studies/_private/embedded_capacity_register_storage.parquet`.
"""

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Final

import polars as pl
from studies.sources import PRIVATE_DIR, REPO_DATA_DIR

WORKBOOK: Final[Path] = REPO_DATA_DIR.parent / "literature" / "NGED" / "NGED ECR AUG 2026.xlsx"
"""The register, filed under `literature/` in the main checkout."""
SHEETS: Final[dict[str, str]] = {"sheet3": "50kW_to_1MW", "sheet4": "1MW_and_above"}
"""The workbook's sheet files and what each holds."""
COLUMNS: Final[dict[str, str]] = {
    "O": "primary_substation",
    "R": "energy_source_1",
    "S": "technology_1",
    "X": "energy_source_2",
    "Y": "technology_2",
    "AD": "energy_source_3",
    "AE": "technology_3",
    "AK": "connection_status",
    "AL": "connected_capacity_mw",
}
"""The spreadsheet columns kept, by letter. Row 2 holds the headings."""
NS: Final[str] = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
OUTPUT: Final[Path] = PRIVATE_DIR / "embedded_capacity_register_storage.parquet"
HEADER_ROW: Final[int] = 2


def _cell_text(*, cell: ET.Element) -> str:
    """Return a cell's text, whether it is inline, numeric, or empty.

    Args:
        cell: A `c` element of a sheet.

    Returns:
        The text, or an empty string.
    """
    if cell.get("t") == "s":
        message = "A shared-string cell holds an index into the string table, not text."
        raise ValueError(message)
    if cell.get("t") == "inlineStr":
        return "".join(node.text or "" for node in cell.iter(NS + "t"))
    value = cell.find(NS + "v")
    return (value.text or "") if value is not None else ""


def read_sheet(*, archive: zipfile.ZipFile, sheet: str, label: str) -> pl.DataFrame:
    """Stream one sheet's kept columns into a frame.

    Args:
        archive: The open workbook.
        sheet: The sheet's file stem, such as `sheet3`.
        label: What the sheet holds, written to the `sheet` column.

    Returns:
        One row per register entry, with string columns.
    """
    rows: list[dict[str, str]] = []
    with archive.open(f"xl/worksheets/{sheet}.xml") as handle:
        for _, element in ET.iterparse(handle):
            if element.tag != NS + "row":
                continue
            if int(element.get("r", "0")) > HEADER_ROW:
                found = {}
                for cell in element.findall(NS + "c"):
                    letters = re.match(r"[A-Z]+", cell.get("r", ""))
                    if letters and letters.group() in COLUMNS:
                        found[COLUMNS[letters.group()]] = _cell_text(cell=cell).strip()
                rows.append({name: found.get(name, "") for name in COLUMNS.values()})
            element.clear()
    return pl.DataFrame(rows).with_columns(sheet=pl.lit(label))


def main() -> None:
    """Convert both register sheets and write the parquet file."""
    with zipfile.ZipFile(WORKBOOK) as archive:
        frame = pl.concat(
            [read_sheet(archive=archive, sheet=stem, label=label) for stem, label in SHEETS.items()]
        ).filter(pl.col("primary_substation") != "")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(OUTPUT)
    print(frame.height, "rows;", frame["connection_status"].value_counts().to_dicts())
    print(frame["energy_source_1"].value_counts().sort("count", descending=True).head(15))


if __name__ == "__main__":
    main()
