"""Turn a product's native time steps into a mean over the hour ending at each label.

Every weather product a study compares has to reach the same temporal object before an XGBoost
model sees it: the mean irradiance over the hour ending at the label, which is what the metered
power of the hour labelled `T` averages over too. Products do not publish that object natively. A
reanalysis can publish a running mean since its last forecast start, and a satellite retrieval
publishes instantaneous snapshots. **Either conversion, done wrong by one step, shifts the whole
series by half an hour or more without any value looking wrong**, which is why each lives here with
tests rather than inline in a study script.
"""

from collections.abc import Sequence
from typing import Final

import polars as pl

KEY_COLUMN: Final[str] = "key"
"""The column naming which series a row belongs to: a site label, or a grid cell's identifier."""


def hourly_from_running_means(
    *, frame: pl.DataFrame, value_column: str, cycle_hours: int
) -> pl.DataFrame:
    """Recover hourly means from running means since each forecast start.

    ICON-DREAM-EU stores each hour's radiation as the mean since the most recent forecast start,
    and its forecasts start every `cycle_hours` hours from 00 UTC. The value labelled `h`
    therefore averages the `k` hours since the start at or before `h - 1`, where
    `k = ((h - 1) mod cycle) + 1`. With `A_k` the value at `h` and `A_{k-1}` the value an hour
    earlier, the mean over the hour ending at `h` is `k * A_k - (k - 1) * A_{k-1}`. At `k = 1` it
    is `A_1` itself.

    **Negative results are clipped at zero.** De-averaging subtracts two nearly equal numbers at
    sunrise and sunset, and the upstream averaging leaves them a few W m⁻² apart the wrong way; a
    negative flux is not a physical value.

    Args:
        frame: One row per (`key`, `time`), with `time` a UTC datetime on the hour and
            `value_column` the running mean. Not-a-number rows are dropped first, because the
            download pads each month with them.
        value_column: The running-mean column.
        cycle_hours: Hours between forecast starts, counted from 00 UTC.

    Returns:
        One row per (`key`, `time`) whose hourly mean could be recovered, with `value_column` now
        the mean over the hour ending at `time`. A row whose `k` exceeds 1 and whose previous hour
        is missing is dropped rather than guessed.

    Raises:
        ValueError: If a (`key`, `time`) appears twice once the padding is dropped, which would
            make the previous hour ambiguous.
    """
    real = frame.filter(pl.col(value_column).is_not_nan() & pl.col(value_column).is_not_null())
    if real.select(KEY_COLUMN, "time").is_duplicated().any():
        msg = f"{value_column} holds more than one row for a (key, time)"
        raise ValueError(msg)
    previous = real.select(
        KEY_COLUMN,
        time=pl.col("time").dt.offset_by("1h"),
        previous=pl.col(value_column),
    )
    hour = pl.col("time").dt.hour().cast(pl.Int32)
    step = (hour + cycle_hours - 1) % cycle_hours + 1
    return (
        real.join(previous, on=[KEY_COLUMN, "time"], how="left")
        .with_columns(step=step)
        .filter((pl.col("step") == 1) | pl.col("previous").is_not_null())
        .with_columns(
            pl.when(pl.col("step") == 1)
            .then(pl.col(value_column))
            .otherwise(
                pl.col("step") * pl.col(value_column) - (pl.col("step") - 1) * pl.col("previous")
            )
            .clip(lower_bound=0.0)
            .alias(value_column)
        )
        .select(KEY_COLUMN, "time", value_column)
        .sort(KEY_COLUMN, "time")
    )


def hourly_from_snapshots(
    *, frame: pl.DataFrame, value_columns: Sequence[str], slot_offsets_minutes: Sequence[int]
) -> pl.DataFrame:
    """Average instantaneous snapshots into one value per hour, labelled at the hour's end.

    The hour labelled `T` takes the mean of the snapshots stamped `T + offset` for each offset in
    `slot_offsets_minutes`. Which slots represent the hour ending at `T` depends on when the
    instrument actually looked, which is not always the stamp: a caller states the offsets and a
    clear-sky check (`studies.timestamp_checks`) confirms them.

    Args:
        frame: One row per (`key`, `time`), with `time` a UTC snapshot stamp and one column per name
            in `value_columns`. A not-a-number or null value marks a snapshot that is missing.
        value_columns: The snapshot columns to average.
        slot_offsets_minutes: The offsets from the label, in minutes, of the snapshots to average,
            such as `(-60, -30)`.

    Returns:
        One row per (`key`, `time`) on the hour at which every listed snapshot is present in every
        value column, with each value column the snapshots' mean. An hour missing any snapshot is
        dropped rather than averaged over fewer.

    Raises:
        ValueError: If `slot_offsets_minutes` is empty, or a (`key`, `time`) appears twice.
    """
    if not slot_offsets_minutes:
        msg = "at least one snapshot offset is needed"
        raise ValueError(msg)
    if frame.select(KEY_COLUMN, "time").is_duplicated().any():
        msg = "the snapshots hold more than one row for a (key, time)"
        raise ValueError(msg)
    present = frame.filter(
        pl.all_horizontal(
            pl.col(column).is_not_nan() & pl.col(column).is_not_null() for column in value_columns
        )
    )
    # A snapshot stamped `T + offset` is moved onto its label `T` by shifting it `-offset`.
    shifted = [
        present.select(
            KEY_COLUMN,
            time=pl.col("time").dt.offset_by(f"{-offset}m"),
            **{f"{column}__{index}": pl.col(column) for column in value_columns},
        )
        for index, offset in enumerate(slot_offsets_minutes)
    ]
    labels = shifted[0].filter(pl.col("time").dt.truncate("1h") == pl.col("time"))
    for other in shifted[1:]:
        labels = labels.join(other, on=[KEY_COLUMN, "time"], how="inner")
    return labels.select(
        KEY_COLUMN,
        "time",
        *(
            pl.mean_horizontal(
                pl.col(f"{column}__{index}") for index in range(len(slot_offsets_minutes))
            ).alias(column)
            for column in value_columns
        ),
    ).sort(KEY_COLUMN, "time")
