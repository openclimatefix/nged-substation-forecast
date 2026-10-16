"""Convert the NGED switching-log workbook to `logs_raw.parquet` (run once).

Run with `uv run --no-project --with openpyxl --with polars python e00037_read_workbook.py
<workbook.xlsx> <out.parquet>`, because the project environment has no `openpyxl`.
"""

import datetime as dt
import sys

import openpyxl  # ty: ignore[unresolved-import]
import polars as pl


def _timestamp(value: object) -> dt.datetime | None:
    return value if isinstance(value, dt.datetime) else None


def main() -> None:
    """Read every sheet but `Info` and write the log rows as parquet."""
    workbook = openpyxl.load_workbook(sys.argv[1], read_only=True, data_only=True)
    records: list[dict] = []
    for sheet in workbook.worksheets:
        if sheet.title == "Info":
            continue
        last: dict | None = None
        for row_number, raw in enumerate(sheet.iter_rows(values_only=True)):
            row = [*raw, *([None] * 8)]
            if isinstance(row[1], dt.datetime) or isinstance(row[3], dt.datetime):
                last = {
                    "sheet": sheet.title,
                    "row": row_number,
                    "job": str(row[0]).strip(),
                    "recorded": _timestamp(row[1]),
                    "operator": str(row[2]).strip() if row[2] else None,
                    "manual": _timestamp(row[3]),
                    "state": str(row[4]).strip() if row[4] else None,
                    "action": str(row[5]).strip() if row[5] else None,
                    "location": str(row[6]).strip() if row[6] else None,
                    "circuit": str(row[7]).strip() if row[7] else None,
                    "note": None,
                }
                records.append(last)
            elif last is not None and row[0] is not None and all(v is None for v in row[1:8]):
                last["note"] = (last["note"] or "") + " | " + str(row[0]).strip()
    schema = {
        "sheet": pl.String,
        "row": pl.Int64,
        "job": pl.String,
        "recorded": pl.Datetime("us"),
        "operator": pl.String,
        "manual": pl.Datetime("us"),
        "state": pl.String,
        "action": pl.String,
        "location": pl.String,
        "circuit": pl.String,
        "note": pl.String,
    }
    frame = pl.DataFrame(records, schema=schema)
    frame.write_parquet(sys.argv[2])
    print(frame.shape)


if __name__ == "__main__":
    main()
