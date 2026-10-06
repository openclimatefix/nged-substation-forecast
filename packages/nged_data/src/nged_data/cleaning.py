"""Flagging NGED's power telemetry that should not be trained or forecast on.

`flag_nged_power` is where cleaning rules live. The `clean_nged_power_data` Dagster asset calls it
with the whole raw `power_time_series` table and writes the result to the
`cleaned_power_time_series` Delta table.
"""

import hashlib
from pathlib import Path
from typing import Final

import patito as pt
import polars as pl
from contracts.power_schemas import CleanedPowerTimeSeries, PowerTimeSeries, TimeSeriesMetadata

SUBSTATION_TYPES: Final[tuple[str, ...]] = ("Primary", "BSP", "GSP")
"""The `TimeSeriesMetadata.substation_type` values that mean "a substation", as opposed to a
generator or a battery at an `HV Customer` or `EHV Customer` connection. The `substation_zero`
rule applies only to these."""

CLEANING_CODE_HASH: Final[str] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
"""The SHA-256 of this module's own source file, computed at import.

`clean_nged_power_data` skips a rebuild only when this hash matches the one recorded in the
cleaned table's newest write commit, so any edit to a cleaning rule forces a rebuild on the next
run, whether or not the edit is committed. The rules must live in this module, or the hash must be
extended to cover every file they live in.
"""


def flag_nged_power(
    power: pt.LazyFrame[PowerTimeSeries],
    metadata: pt.DataFrame[TimeSeriesMetadata],
) -> pt.LazyFrame[CleanedPowerTimeSeries]:
    """Flag the rows of NGED's power telemetry that cleaning rejects, by adding `drop_reason`.

    **Where this sits.** `clean_nged_power_data` calls this function with the whole raw power
    table and the roster, about four times a day, and writes the result to the cleaned table.
    Every reader of power then sees only the rows whose `drop_reason` is null.

    **How to add a rule.** Write a boolean Polars expression that is true for a row to drop. Add it
    as one more `.when(...).then(pl.lit("<reason>"))` branch of the chain below, and add the reason
    to `contracts.power_schemas.DROP_REASONS`. The first matching branch wins. A rule that needs a
    `TimeSeriesMetadata` column other than `substation_type` adds that column to the roster
    `select` in this function.

    **The contract a rule keeps.** Return every input row exactly once. Never change
    `time_series_id` or `time`. Flag rows rather than deleting them. Do not call `collect` inside
    this function.

    **Logging.** The asset reports to Dagster, per reason, the rows and series flagged, the
    minimum and maximum flagged power, and the first and last flagged time, so there is no logging
    code to write. Output from Python's `logging` reaches only the step's captured stderr.

    The one rule today, `substation_zero`, flags a reading of exactly 0 from a `Primary`, `BSP`,
    or `GSP` series. A substation almost never truly reads zero, so a zero is almost always a
    telemetry fault. A series missing from the roster has no `substation_type` and is not flagged.

    Args:
        power: The whole raw power table.
        metadata: The `TimeSeriesMetadata` roster.

    Returns:
        Every row of `power`, in no guaranteed order, with a `drop_reason` column: null for a row
        that passed, otherwise the name of the rule that flagged it.
    """
    # Join the roster columns the rules need onto the power rows. The roster is stripped of its
    # Patito model so Polars' cross-subclass join check does not reject the join.
    roster = (
        pl.DataFrame._from_pydf(metadata._df).select("time_series_id", "substation_type").lazy()
    )
    joined = pl.LazyFrame._from_pyldf(power._ldf).join(roster, on="time_series_id", how="left")

    flagged = joined.with_columns(
        drop_reason=pl.when(
            (pl.col("power") == 0) & pl.col("substation_type").is_in(SUBSTATION_TYPES)
        )
        .then(pl.lit("substation_zero"))
        # A bare `pl.lit(None)` has the `Null` dtype, which would not match the contract's String.
        .otherwise(pl.lit(None, dtype=pl.String))
    ).select("time_series_id", "time", "power", "drop_reason")
    return pt.LazyFrame.from_existing(flagged).set_model(CleanedPowerTimeSeries)
