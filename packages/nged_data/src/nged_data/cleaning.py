"""Flagging NGED's power telemetry that should not be trained or forecast on.

`flag_nged_power` is where cleaning rules live. The `clean_nged_power_data` Dagster asset calls it
with the whole raw `power_time_series` table and writes the result to the
`cleaned_power_time_series` Delta table.
"""

import ast
import hashlib
from pathlib import Path
from typing import Final

import patito as pt
import polars as pl
from contracts.power_schemas import CleanedPowerTimeSeries, PowerTimeSeries, TimeSeriesMetadata


def hash_code_ignoring_docs(source: str) -> str:
    """Return the SHA-256 of a Python module's syntax tree, with docstrings and comments removed.

    The syntax tree carries no comments, line numbers or formatting, and this function deletes every
    statement that is a bare string literal (a docstring), so editing any of those leaves the hash
    unchanged. Upgrading Python can change the hash, because each Python version may print its
    syntax tree differently; the cost is one extra rebuild.

    Args:
        source: The module's Python source code.

    Returns:
        The hex digest.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list):
            node.body = [  # ty: ignore[unresolved-attribute]
                statement
                for statement in body
                if not (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                )
            ]
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()


CLEANING_CODE_HASH: Final[str] = hash_code_ignoring_docs(Path(__file__).read_text())
"""The hash of this module's code, computed at import by `hash_code_ignoring_docs`.

`clean_nged_power_data` skips a rebuild only when this hash matches the one recorded in the
cleaned table's newest write commit, so any edit to a cleaning rule forces a rebuild on the next
run, whether or not the edit is committed. An edit to a docstring or a comment does not. The
cleaning rules must live in this module, or the hash must be extended to cover every file they
live in.
"""


def flag_nged_power(
    power: pt.LazyFrame[PowerTimeSeries],
    metadata: pt.DataFrame[TimeSeriesMetadata],
) -> pt.LazyFrame[CleanedPowerTimeSeries]:
    """Flag the rows of NGED's power telemetry that cleaning rejects, by adding `drop_reason`.

    **Where this sits.** `clean_nged_power_data` calls this function with the whole raw power
    table and the `TimeSeriesMetadata` table, about four times a day, and writes the result to the
    cleaned table. Every reader of power then sees only the rows whose `drop_reason` is null.

    **How to add a rule.** Write a boolean Polars expression that is true for a row to flag. Add
    the expression as one more `.when(...).then(pl.lit("<reason>"))` branch of the chain below, and
    add the reason to `contracts.power_schemas.DROP_REASONS`. The first matching branch wins. A
    rule needing another `TimeSeriesMetadata` column adds that column to the `select` below.

    **The contract a rule keeps.** Return every input row exactly once, flagged rather than
    deleted. Never change `time_series_id` or `time`. Do not call `collect` inside this function.

    **Logging.** `clean_nged_power_data` reports to Dagster, per reason, the rows and series
    flagged and the range of flagged power and time, so a rule needs no logging code. Python's
    `logging` output reaches only the step's captured stderr.

    The one rule today, `substation_zero`, flags a reading of exactly 0 from a `Primary`, `BSP`,
    or `GSP` series. A substation almost never truly reads zero, so a zero is almost always a
    telemetry fault. A series missing from the `TimeSeriesMetadata` table has no `substation_type`
    and is not flagged.

    Args:
        power: The whole raw power table.
        metadata: The `TimeSeriesMetadata` table.

    Returns:
        Every row of `power`, in no guaranteed order, with a `drop_reason` column: null for a row
        that passed, otherwise the name of the rule that flagged it.
    """
    # Join the `TimeSeriesMetadata` columns the cleaning rules need onto the power rows. Both frames
    # are stripped of their Patito models so Polars' cross-subclass join check does not reject the
    # join.
    metadata_columns = (
        pl.DataFrame._from_pydf(metadata._df).select("time_series_id", "substation_type").lazy()
    )
    joined = pl.LazyFrame._from_pyldf(power._ldf).join(
        metadata_columns, on="time_series_id", how="left"
    )
    # `joined` has one row per power reading, with the columns `time_series_id`, `time`, `power`,
    # and `substation_type` (null for a series missing from the `TimeSeriesMetadata` table).

    flagged = joined.with_columns(
        # `pl.when(condition).then(value)` sets `drop_reason` to `value` on every row where
        # `condition` is true. Add a rule as another `.when(...).then(...)` before `.otherwise`.
        drop_reason=pl.when(
            # The `substation_zero` rule: the reading is exactly zero, and the series is a
            # substation. `&` is a row-by-row "and"; `is_in` is true where `substation_type` is
            # one of the listed values, and null (so not flagged) where `substation_type` is null.
            (pl.col("power") == 0)
            & pl.col("substation_type").is_in(TimeSeriesMetadata.SUBSTATION_TYPES)
        )
        .then(pl.lit("substation_zero"))
        # Every row no rule flagged gets a null `drop_reason`. A bare `pl.lit(None)` has the `Null`
        # dtype, which would not match the contract's String.
        .otherwise(pl.lit(None, dtype=pl.String))
    ).select("time_series_id", "time", "power", "drop_reason")
    return pt.LazyFrame.from_existing(flagged).set_model(CleanedPowerTimeSeries)
