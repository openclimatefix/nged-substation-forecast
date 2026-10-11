"""Render a Polars frame as a Markdown table for `report.md`."""

import polars as pl


def markdown_table(*, frame: pl.DataFrame, decimals: int = 4) -> str:
    """Return the frame as a pipe table, with floats rounded to `decimals` places.

    Args:
        frame: Any frame; nulls print as empty cells.
        decimals: Decimal places for float columns.

    Returns:
        The table, one line per row, with a header and a separator line.
    """

    def cell(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.{decimals}f}"
        return str(value)

    header = "| " + " | ".join(frame.columns) + " |"
    separator = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(cell(value) for value in row) + " |" for row in frame.iter_rows()]
    return "\n".join([header, separator, *rows])
