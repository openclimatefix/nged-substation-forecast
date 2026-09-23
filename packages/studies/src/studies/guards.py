"""Two run-time guards a study script calls before it trusts its rows or writes its outputs.

Both raise rather than warn, because each catches a mistake that would otherwise publish a number
nobody could tell apart from a real one: an arm scored on rows where its input is missing, or a
merged page's evidence overwritten by a later run.
"""

from collections.abc import Iterable
from pathlib import Path

import polars as pl


def refuse_to_overwrite(*, paths: Iterable[Path]) -> None:
    """Raise if any output a run is about to write already exists.

    Args:
        paths: Every file the run writes.

    Raises:
        FileExistsError: Naming the first file that exists, which has to be moved to a
            `superseded/` subfolder first, because a merged page may quote it.
    """
    for path in paths:
        if path.exists():
            msg = f"{path} exists; move it to a superseded/ subfolder before re-running"
            raise FileExistsError(msg)


def check_no_missing(*, frame: pl.DataFrame, columns: Iterable[str]) -> None:
    """Raise if any named column holds a null or, for a float column, a not-a-number.

    Scoring every arm on the same rows rests on every arm's input being present on every row. An
    arm shown a missing value would still be fitted and scored, on rows where it had nothing to go
    on.

    Args:
        frame: The rows every arm is scored on.
        columns: Every column an arm is shown.

    Raises:
        ValueError: Naming each column that holds a missing value, and how many.
    """
    missing: dict[str, int] = {}
    for column in dict.fromkeys(columns):
        values = frame[column]
        count = values.null_count()
        if values.dtype.is_float():
            count += int(values.is_nan().sum())
        if count:
            missing[column] = count
    if missing:
        msg = f"missing values in columns an arm is shown: {missing}"
        raise ValueError(msg)
