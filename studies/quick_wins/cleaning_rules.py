"""Rules that mark half-hours of a series' power history to leave out of model training.

Production trains on the unflagged rows of the cleaned power table, which does minimal cleaning.
Each rule here names further half-hours to drop from the *training* rows only. Validation rows and
observations are never touched, so every arm is scored on the same rows. A rule reads only the
series' own history up to the end of the training window, so no validation information reaches the
training set.

Every rule takes a frame of one or more series' readings with columns `time_series_id`, `time`, and
`power`, and returns the `(time_series_id, time)` pairs to drop.
"""

from datetime import datetime
from typing import Final, Literal

import polars as pl

CleaningRule = Literal[
    "none",
    "first_1_month",
    "first_2_months",
    "first_3_months",
    "first_6_months",
    "isolated_zeros",
    "spikes",
    "commissioning_ramp",
]
"""The training-data cleaning rules a round 0 arm can use."""

CLEANING_RULES: Final[tuple[CleaningRule, ...]] = (
    "none",
    "first_1_month",
    "first_2_months",
    "first_3_months",
    "first_6_months",
    "isolated_zeros",
    "spikes",
    "commissioning_ramp",
)

ZERO_NEIGHBOUR_FRACTION: Final[float] = 0.01
"""A zero is isolated only if both neighbours exceed this fraction of the series' capacity."""

SPIKE_FRACTION: Final[float] = 0.3
"""A reading is a spike if it differs from both neighbours, in the same direction, by more than
this fraction of the series' capacity, while the neighbours differ from each other by less than
half as much."""

RAMP_DAYS_MINIMUM: Final[int] = 60
"""A series needs this many days of history before the commissioning-ramp rule judges it."""

RAMP_SETTLED_DAYS: Final[int] = 90
"""The settled level of a generator is its median relative output over this many final days."""

RAMP_SETTLED_FRACTION: Final[float] = 0.9
"""The ramp ends once a 14-day median of relative output reaches this fraction of the settled
level."""

RAMP_ROLLING_DAYS: Final[int] = 14
"""Width of the rolling median that finds the end of a commissioning ramp."""

RAMP_MIN_PEERS: Final[int] = 2
"""The ramp rule needs at least this many other series of the same type to form a reference."""

HALF_HOURS_PER_DAY: Final[int] = 48
"""Half-hourly readings in a day."""

MIN_HALF_HOURS_FOR_A_DAY: Final[int] = 40
"""A day counts toward the ramp rule only if it has this many half-hourly readings."""


def training_capacity(*, power: pl.DataFrame) -> pl.DataFrame:
    """Return each series' 99th percentile of absolute power in the given window, as capacity."""
    return power.group_by("time_series_id").agg(
        capacity=pl.col("power").abs().quantile(0.99).cast(pl.Float64)
    )


def first_months_after_first_nonzero(*, power: pl.DataFrame, months: int) -> pl.DataFrame:
    """Drop everything before a series' first non-zero reading, and `months` months after it.

    Args:
        power: Readings of every series, up to the end of the training window.
        months: Calendar months to drop after the first non-zero reading.

    Returns:
        The `(time_series_id, time)` pairs to drop. A series with no non-zero reading is dropped
        whole.
    """
    first_nonzero = (
        power.filter(pl.col("power").is_not_null() & (pl.col("power") != 0))
        .group_by("time_series_id")
        .agg(first=pl.col("time").min())
        .with_columns(cutoff=pl.col("first").dt.offset_by(f"{months}mo"))
    )
    return (
        power.join(first_nonzero, on="time_series_id", how="left")
        .filter(pl.col("cutoff").is_null() | (pl.col("time") < pl.col("cutoff")))
        .select("time_series_id", "time")
    )


def isolated_zeros(*, power: pl.DataFrame) -> pl.DataFrame:
    """Drop exact zeros whose two half-hour neighbours are both clearly non-zero.

    Args:
        power: Readings of every series, up to the end of the training window.

    Returns:
        The `(time_series_id, time)` pairs to drop.
    """
    capacity = training_capacity(power=power)
    ordered = power.sort("time_series_id", "time").join(capacity, on="time_series_id")
    step = pl.duration(minutes=30)
    return (
        ordered.with_columns(
            before=pl.col("power").shift(1).over("time_series_id"),
            after=pl.col("power").shift(-1).over("time_series_id"),
            gap_before=pl.col("time") - pl.col("time").shift(1).over("time_series_id"),
            gap_after=pl.col("time").shift(-1).over("time_series_id") - pl.col("time"),
        )
        .filter(
            (pl.col("power") == 0)
            & (pl.col("gap_before") == step)
            & (pl.col("gap_after") == step)
            & (pl.col("before").abs() > ZERO_NEIGHBOUR_FRACTION * pl.col("capacity"))
            & (pl.col("after").abs() > ZERO_NEIGHBOUR_FRACTION * pl.col("capacity"))
        )
        .select("time_series_id", "time")
    )


def spikes(*, power: pl.DataFrame) -> pl.DataFrame:
    """Drop single readings that jump away from both neighbours while the neighbours agree.

    Args:
        power: Readings of every series, up to the end of the training window.

    Returns:
        The `(time_series_id, time)` pairs to drop.
    """
    capacity = training_capacity(power=power)
    ordered = power.sort("time_series_id", "time").join(capacity, on="time_series_id")
    step = pl.duration(minutes=30)
    jump_before = pl.col("power") - pl.col("before")
    jump_after = pl.col("power") - pl.col("after")
    threshold = SPIKE_FRACTION * pl.col("capacity")
    return (
        ordered.with_columns(
            before=pl.col("power").shift(1).over("time_series_id"),
            after=pl.col("power").shift(-1).over("time_series_id"),
            gap_before=pl.col("time") - pl.col("time").shift(1).over("time_series_id"),
            gap_after=pl.col("time").shift(-1).over("time_series_id") - pl.col("time"),
        )
        .filter(
            (pl.col("gap_before") == step)
            & (pl.col("gap_after") == step)
            & (
                ((jump_before > threshold) & (jump_after > threshold))
                | ((jump_before < -threshold) & (jump_after < -threshold))
            )
            & ((pl.col("before") - pl.col("after")).abs() < threshold / 2)
        )
        .select("time_series_id", "time")
    )


def commissioning_ramp(*, power: pl.DataFrame, types: pl.DataFrame) -> pl.DataFrame:
    """Drop the days before a generator reaches its settled output relative to its peers.

    The reference for a series is the median daily mean output of the other series of its own type
    (solar farms against solar farms), which cancels the weather. The settled level is the median
    ratio over the last `RAMP_SETTLED_DAYS` days of the window. The ramp ends on the first day on
    which a `RAMP_ROLLING_DAYS`-day rolling median of the ratio reaches `RAMP_SETTLED_FRACTION` of
    the settled level, and every day before that is dropped.

    Args:
        power: Readings of every series, up to the end of the training window.
        types: Columns `time_series_id` and `time_series_type`. Only series of the generator types
            PV and Wind are judged.

    Returns:
        The `(time_series_id, time)` pairs to drop.
    """
    daily = (
        power.join(types, on="time_series_id")
        .filter(pl.col("time_series_type").is_in(["PV", "Wind"]))
        .with_columns(day=pl.col("time").dt.offset_by("-1m").dt.truncate("1d"))
        .group_by("time_series_id", "time_series_type", "day")
        .agg(mean=pl.col("power").mean(), n=pl.len())
        .filter(pl.col("n") >= MIN_HALF_HOURS_FOR_A_DAY)
    )
    peers = (
        daily.join(daily, on=["time_series_type", "day"], suffix="_peer")
        .filter(pl.col("time_series_id") != pl.col("time_series_id_peer"))
        .group_by("time_series_id", "time_series_type", "day")
        .agg(reference=pl.col("mean_peer").median(), n_peers=pl.len(), mean=pl.col("mean").first())
        .filter((pl.col("n_peers") >= RAMP_MIN_PEERS) & (pl.col("reference") > 0))
        .with_columns(ratio=pl.col("mean") / pl.col("reference"))
        .sort("time_series_id", "day")
    )
    ends = []
    for (series_id,), group in peers.group_by("time_series_id"):
        if group.height < RAMP_DAYS_MINIMUM:
            continue
        settled = group.tail(RAMP_SETTLED_DAYS).select(pl.col("ratio").median()).item()
        if not isinstance(settled, float) or settled <= 0:
            continue
        rolling = group.with_columns(
            smooth=pl.col("ratio").rolling_median(
                window_size=RAMP_ROLLING_DAYS, min_samples=RAMP_ROLLING_DAYS
            )
        )
        reached = rolling.filter(pl.col("smooth") >= RAMP_SETTLED_FRACTION * settled)
        if reached.height == 0:
            continue
        ends.append(
            {
                "time_series_id": series_id,
                "first_day": group["day"].min(),
                "ramp_end": reached["day"].min(),
            }
        )
    if not ends:
        return power.head(0).select("time_series_id", "time")
    ramp_ends = pl.DataFrame(ends).cast({"time_series_id": power.schema["time_series_id"]})
    return (
        power.join(ramp_ends, on="time_series_id")
        .with_columns(day=pl.col("time").dt.offset_by("-1m").dt.truncate("1d"))
        .filter((pl.col("day") >= pl.col("first_day")) & (pl.col("day") < pl.col("ramp_end")))
        .select("time_series_id", "time")
    )


def rows_to_drop(
    *,
    rule: CleaningRule,
    power: pl.DataFrame,
    types: pl.DataFrame,
    train_start: datetime,
) -> pl.DataFrame:
    """Return the `(time_series_id, time)` pairs a named rule drops from training.

    Every rule except the first-months rules judges only readings inside the training window
    (`time >= train_start`), so a fold whose validation period precedes its training window cannot
    leak that period into a threshold. The first-months rules need the whole history before the
    window end, because a series is new when it first reads above zero, not when training starts.

    Args:
        rule: One of `CLEANING_RULES`.
        power: Readings of every series, from the start of the record to the end of the training
            window.
        types: Columns `time_series_id` and `time_series_type`.
        train_start: The first instant of the training window.

    Returns:
        The pairs to drop, with no duplicates.
    """
    in_window = power.filter(pl.col("time") >= train_start)
    if rule == "none":
        dropped = power.head(0).select("time_series_id", "time")
    elif rule.startswith("first_"):
        months = int(rule.split("_")[1])
        dropped = first_months_after_first_nonzero(power=power, months=months)
        dropped = dropped.filter(pl.col("time") >= train_start)
    elif rule == "isolated_zeros":
        dropped = isolated_zeros(power=in_window)
    elif rule == "spikes":
        dropped = spikes(power=in_window)
    else:
        dropped = commissioning_ramp(power=in_window, types=types)
    return dropped.unique()
