"""Show a model a weather product's value in the hours around each scored hour.

**A product's neighbouring hours carry information its own hour does not**: a front arriving an
hour early or late in the model, or a cloud edge that the hourly mean smooths away.

**The neighbouring values must come from the product's own download, not from the scored rows.**
The scored rows exclude every hour holding a zero half-hour of metered power, so a neighbour read
from them would be missing exactly where the target was zero next door, which tells the model about
the target. `with_neighbouring_hours` therefore takes the source separately and leaves a neighbour
missing only where the product itself served no value.
"""

from collections.abc import Mapping
from typing import Final

import polars as pl

JOIN_KEYS: Final[tuple[str, str]] = ("site", "time")
"""The columns identifying one hour at one generator, in the frame and in the source."""


def with_neighbouring_hours(
    *, frame: pl.DataFrame, source: pl.DataFrame, columns: Mapping[str, tuple[str, int]]
) -> pl.DataFrame:
    """Add each named source column's value a whole number of hours away from each row's hour.

    Args:
        frame: The rows to add columns to, carrying `site` and `time`.
        source: The product's own download, one row per (site, time), carrying every source column.
        columns: The new column's name, mapped to the source column and the offset in hours. An
            offset of -1 reads the hour before each row's hour, and +1 the hour after.

    Returns:
        `frame`, in its own row order and with its own rows, plus one column per entry in `columns`,
        null wherever the source holds no value at that site and hour.

    Raises:
        ValueError: If `source` holds more than one row for a (site, time), which would duplicate
            the frame's rows, or if `frame` already carries a column named in `columns`, which a
            join would otherwise rename rather than raise on.
    """
    if source.select(JOIN_KEYS).is_duplicated().any():
        msg = "the source holds more than one row for a (site, time)"
        raise ValueError(msg)
    clashes = sorted(set(columns) & set(frame.columns))
    if clashes:
        msg = f"frame already has {clashes}, which columns would add again"
        raise ValueError(msg)
    for name, (column, offset_hours) in columns.items():
        shifted = source.select(
            "site",
            time=pl.col("time").dt.offset_by(f"{-offset_hours}h"),
            **{name: pl.col(column)},
        )
        frame = frame.join(shifted, on=list(JOIN_KEYS), how="left", maintain_order="left")
    return frame
