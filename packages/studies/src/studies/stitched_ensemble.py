"""Turn a table of ensemble members into one hourly series of ensemble means, stitched across runs.

Open-Meteo's ensemble-mean archive serves, for each hour, a mean from the newest run that covers
the hour, with no run time attached. The functions here build the same kind of series from this
repository's own ensemble table (`data/studies/weather/ENS/`), so the two can sit in one
comparison: average the members, keep the newest run for each valid time, and spread the
three-hourly steps onto hours. The two ways of spreading follow the two kinds of variable.
"""

from collections.abc import Sequence
from typing import Final

import polars as pl

STEP_HOURS: Final[int] = 3
"""The spacing of the ensemble table's valid times, in hours, at the leads this module reads."""

MIN_EXTRATERRESTRIAL_W_M2: Final[float] = 1.0
"""Below this mean extraterrestrial irradiance over a step, the step has no clearness index."""


def newest_run_member_means(
    *,
    members: pl.DataFrame,
    value_columns: Sequence[str],
    expected_members: int,
    max_lead_hours: int,
) -> pl.DataFrame:
    """Average the members of each run, then keep the newest run for each valid time.

    Args:
        members: One row per (site, run, valid time, member), carrying `site`, `init_time`,
            `valid_time`, `lead_hours`, `ensemble_member` and every column in `value_columns`.
        value_columns: The columns to average.
        expected_members: How many members every run should hold at every valid time.
        max_lead_hours: The longest lead to read. A longer lead is dropped before any run is chosen,
            so a valid time never takes its value from a run further ahead than this.

    Returns:
        One row per (site, valid time), sorted, carrying `init_time` of the newest run that covers
        it and the member mean of each value column.

    Raises:
        ValueError: If any (site, run, valid time) holds a member count other than
            `expected_members`, which would bias its mean.
    """
    means = (
        members.filter(pl.col("lead_hours") <= max_lead_hours)
        .group_by("site", "init_time", "valid_time")
        .agg(
            *(pl.col(column).mean() for column in value_columns),
            n_members=pl.col("ensemble_member").n_unique(),
        )
    )
    short = means.filter(pl.col("n_members") != expected_members)
    if not short.is_empty():
        msg = (
            f"{short.height} (site, run, valid time) groups hold other than "
            f"{expected_members} members"
        )
        raise ValueError(msg)
    return (
        means.drop("n_members")
        .sort("site", "valid_time", "init_time")
        .unique(subset=["site", "valid_time"], keep="last", maintain_order=True)
        .sort("site", "valid_time")
    )


def hold_backward_mean_hourly(
    *, steps: pl.DataFrame, value_columns: Sequence[str], step_hours: int = STEP_HOURS
) -> pl.DataFrame:
    """Give every hour of a step the step's value, for a variable that is a mean over the step.

    A radiation value labelled `V` is a mean over the `step_hours` hours ending at `V`, so each of
    those hours gets that value. The hour labelled `V - 1 h` is the mean over the hour ending
    there, and its value is the step's mean, not an interpolation to a neighbouring step.

    Args:
        steps: One row per (site, valid time), carrying `site`, `valid_time` and the value columns.
        value_columns: The columns to spread.
        step_hours: The width of one step, in hours.

    Returns:
        One row per (site, hour label) named `time`, carrying the value columns.
    """
    offsets = pl.DataFrame({"offset_hours": list(range(step_hours))})
    return (
        steps.join(offsets, how="cross")
        .with_columns(time=pl.col("valid_time").dt.offset_by(pl.format("-{}h", "offset_hours")))
        .select("site", "time", *value_columns)
        .sort("site", "time")
    )


def interpolate_instants_hourly(
    *, steps: pl.DataFrame, value_columns: Sequence[str], step_hours: int = STEP_HOURS
) -> pl.DataFrame:
    """Interpolate linearly between steps, for a variable that is an instantaneous value.

    Args:
        steps: One row per (site, valid time), carrying `site`, `valid_time` and the value columns,
            with every valid time `step_hours` after the one before it.
        value_columns: The columns to interpolate.
        step_hours: The spacing of the steps, in hours.

    Returns:
        One row per (site, hour) named `time`, from each site's first valid time to its last,
        carrying the value columns.

    Raises:
        ValueError: If any site's valid times are not evenly `step_hours` apart, which linear
            interpolation across a gap would hide.
    """
    ordered = steps.sort("site", "valid_time")
    gaps = ordered.select(
        gap_hours=pl.col("valid_time").diff().over("site").dt.total_hours()
    ).drop_nulls()
    if (gaps["gap_hours"] != step_hours).any():
        msg = f"some valid times are not {step_hours} hours after the one before"
        raise ValueError(msg)
    sites = []
    for site, site_steps in ordered.group_by("site", maintain_order=True):
        hourly = (
            site_steps.select(pl.col("valid_time").alias("time"), *value_columns)
            .upsample(time_column="time", every="1h")
            .with_columns(pl.col(*value_columns).interpolate())
            .with_columns(site=pl.lit(site[0]))
            .select("site", "time", *value_columns)
        )
        sites.append(hourly)
    return pl.concat(sites).sort("site", "time")


def interpolate_clearness_hourly(
    *,
    steps: pl.DataFrame,
    extraterrestrial_hourly: pl.DataFrame,
    value_column: str,
    step_hours: int = STEP_HOURS,
    min_extraterrestrial_w_m2: float = MIN_EXTRATERRESTRIAL_W_M2,
) -> pl.DataFrame:
    """Spread a step-mean irradiance onto hours by interpolating its clearness index.

    `hold_backward_mean_hourly` gives every hour of a step the step's mean, so the series steps
    sharply at each boundary and ignores the sun's motion within the step. This function divides
    each step's mean by the mean extraterrestrial irradiance over the step (the clearness index),
    interpolates the clearness index linearly between step centres, and multiplies each hour's
    interpolated clearness index by that hour's extraterrestrial irradiance. A step with no
    daylight has no clearness index, and an hour beside such a step takes its own step's value.

    Args:
        steps: One row per (site, valid time), carrying `site`, `valid_time` and `value_column`,
            the mean over the `step_hours` hours ending at `valid_time`.
        extraterrestrial_hourly: One row per (site, hour label), carrying `site`, `time` and
            `extraterrestrial_horizontal_w_m2`, for every hour of every step.
        value_column: The step-mean irradiance column to spread.
        step_hours: The width of one step, in hours. Must be odd, so a step has a central hour.
        min_extraterrestrial_w_m2: A step whose mean extraterrestrial irradiance is below this has
            no clearness index.

    Returns:
        One row per (site, hour label) named `time`, carrying `value_column`.

    Raises:
        ValueError: If `step_hours` is even, or an hour of a step has no extraterrestrial value.
    """
    if step_hours % 2 == 0:
        msg = f"step_hours must be odd, got {step_hours}"
        raise ValueError(msg)
    centre = (step_hours - 1) // 2
    hours = (
        steps.select("site", "valid_time", value_column)
        .join(pl.DataFrame({"offset_hours": list(range(step_hours))}), how="cross")
        .with_columns(time=pl.col("valid_time").dt.offset_by(pl.format("-{}h", "offset_hours")))
        .join(extraterrestrial_hourly, on=["site", "time"], how="left")
    )
    if hours["extraterrestrial_horizontal_w_m2"].null_count():
        msg = "some hours of a step have no extraterrestrial irradiance"
        raise ValueError(msg)
    per_step = (
        hours.group_by("site", "valid_time")
        .agg(
            pl.col(value_column).first(),
            step_extraterrestrial=pl.col("extraterrestrial_horizontal_w_m2").mean(),
        )
        .with_columns(
            clearness=pl.when(pl.col("step_extraterrestrial") >= min_extraterrestrial_w_m2)
            .then(pl.col(value_column) / pl.col("step_extraterrestrial"))
            .otherwise(None)
        )
    )
    step_width = f"{step_hours}h"
    previous = per_step.select(
        "site",
        pl.col("valid_time").dt.offset_by(step_width),
        clearness_previous=pl.col("clearness"),
    )
    following = per_step.select(
        "site",
        pl.col("valid_time").dt.offset_by(f"-{step_width}"),
        clearness_next=pl.col("clearness"),
    )
    distance = (pl.col("offset_hours") - centre).abs()
    neighbour = (
        pl.when(pl.col("offset_hours") > centre)
        .then(pl.col("clearness_previous"))
        .otherwise(pl.col("clearness_next"))
    )
    interpolated = (
        hours.drop(value_column)
        .join(per_step.select("site", "valid_time", "clearness"), on=["site", "valid_time"])
        .join(previous, on=["site", "valid_time"], how="left")
        .join(following, on=["site", "valid_time"], how="left")
        .with_columns(
            clearness_hour=pl.col("clearness")
            + (pl.coalesce(neighbour, pl.col("clearness")) - pl.col("clearness"))
            * distance
            / step_hours
        )
    )
    return interpolated.select(
        "site",
        "time",
        pl.when(pl.col("clearness_hour").is_null())
        .then(0.0)
        .otherwise(pl.col("clearness_hour") * pl.col("extraterrestrial_horizontal_w_m2"))
        .alias(value_column),
    ).sort("site", "time")
