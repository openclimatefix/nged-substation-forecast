"""Flagging NGED's power telemetry that should not be trained or forecast on.

`flag_nged_power` is where cleaning rules live. The `clean_nged_power_data` Dagster asset calls it
with the whole raw `power_time_series` table and writes the result to the
`cleaned_power_time_series` Delta table.
"""

import hashlib
from pathlib import Path
from typing import Final

import contracts.power_schemas
import patito as pt
import polars as pl
from contracts.power_schemas import CleanedPowerTimeSeries, PowerTimeSeries, TimeSeriesMetadata

CLEANING_CODE_HASH: Final[str] = hashlib.sha256(
    Path(__file__).read_bytes() + Path(contracts.power_schemas.__file__).read_bytes()
).hexdigest()
"""The SHA-256 of this module's source file and of `contracts/power_schemas.py`, computed at import.

`power_schemas.py` is included because the `substation_zero` rule reads
`TimeSeriesMetadata.SUBSTATION_TYPES` from it.

`clean_nged_power_data` skips a rebuild only when this hash matches the one recorded in the
cleaned table's newest write commit, so any edit to a cleaning rule forces a rebuild on the next
run, whether or not the edit is committed. Every file a cleaning rule reads its code or constants
from must be in this hash.
"""

INITIAL_ANALOGUE_NOISE_READINGS: Final[int] = 30 * 3 * 48
"""How many of a series' earliest readings the `initial_analogue_noise` rule flags: 4,320, which
is 90 days' worth of half-hourly readings.

The count is of readings present, not of calendar time, so a gap early in a series pushes the
cut-off later than 3 months in time. Some series report only sporadically at first, and the cut-off
has to fall at least is designed to include at least 2 months after the data starts in earnest.
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

    The rules today, in the order they are tried:

    1. `initial_analogue_noise` flags each series' first `INITIAL_ANALOGUE_NOISE_READINGS`
       readings in time order, for every series whatever its `substation_type`. The first months
       of a series are often noisy, so they are not trained on. The rule counts readings present,
       so a gap does not count towards the cut-off. Two consequences follow, and both are
       intended. A new series is flagged in full until it has `INITIAL_ANALOGUE_NOISE_READINGS`
       readings, so the series gets no training data before then. A back-filled
       reading earlier than a series' current first reading moves that series' cut-off earlier in
       the series' history, and un-flags the readings that are no longer among the first
       `INITIAL_ANALOGUE_NOISE_READINGS`.
    2. `substation_zero` flags a reading of exactly 0 from a `Primary`, `BSP`, or `GSP` series. A
       substation almost never truly reads zero, so a zero is almost always a telemetry fault. A
       series missing from the `TimeSeriesMetadata` table has no `substation_type` and is not
       flagged.

    **Example.** The output holds every input power row, with a `drop_reason` column added. These
    seven rows are real readings from a `Primary` substation, MARSH LANE 33 11kV S STN, after
    cleaning:

    ```text
    time_series_id  time (UTC)        power (MVA)  drop_reason
    Int32           Datetime          Float32      String
    26              2026-03-05 12:30  0.043        null
    26              2026-03-05 13:00  0.094        null
    26              2026-03-05 13:30  0.045        null
    26              2026-03-05 14:00  0.0          substation_zero
    26              2026-03-05 14:30  0.392        null
    26              2026-03-05 15:00  1.492        null
    26              2026-03-05 15:30  2.585        null
    ```

    The 14:00 reading of exactly 0 MVA is flagged. The flagged row stays in the output. Every reader
    of power skips the flagged row, so the zero never reaches training, cross-validation, scoring,
    or live forecasts. A zero from an `EHV Customer` or `HV Customer` series would get a null
    `drop_reason` and stay in, because a customer site, such as a solar farm at night, can truly
    read zero. `CleanedPowerTimeSeries` in `contracts.power_schemas` defines these four columns and
    their dtypes.

    Args:
        power: The whole raw power table.
        metadata: The `TimeSeriesMetadata` table.

    Returns:
        Every row of `power`, in no guaranteed order, with a `drop_reason` column: null for a row
        that passed, otherwise the name of the rule that flagged it.
    """
    # Join the `TimeSeriesMetadata` columns the cleaning rules need onto the power rows. An eager
    # `select` returns a plain Polars frame. The power frame is stripped of its `PowerTimeSeries`
    # model, which would otherwise carry through the join to a frame with different columns.
    metadata_columns = metadata.select("time_series_id", "substation_type").lazy()
    joined = pl.LazyFrame._from_pyldf(power._ldf).join(
        metadata_columns, on="time_series_id", how="left"
    )
    # `joined` has one row per power reading, with the columns `time_series_id`, `time`, `power`,
    # and `substation_type` (null for a series missing from the `TimeSeriesMetadata` table).

    flagged = joined.with_columns(
        # `pl.when(condition).then(value)` sets `drop_reason` to `value` on every row where
        # `condition` is true. Add a rule as another `.when(...).then(...)` before `.otherwise`.
        drop_reason=pl.when(
            # The `initial_analogue_noise` rule: the reading's position in its series, counting
            # from 0 in time order, is below the cut-off. `order_by` is required because neither
            # the raw Delta table nor the lazy join above returns rows in time order.
            pl.int_range(pl.len()).over("time_series_id", order_by="time")
            < INITIAL_ANALOGUE_NOISE_READINGS
        )
        .then(pl.lit("initial_analogue_noise"))
        .when(
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
