"""Read an ECMWF ENS forecast's members into the columns the ENS forecast study fits.

One-off throwaway module for the ENS forecast study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/810>. It names the weather fields
and the columns of each ENS arm, reduces the members of a run to the arm's statistic, and builds the
shared feature list. The past-weather studies score ENS against other products, so they read these
columns too.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl

from studies.arm_runner import SHARED_FEATURES as SOLAR_SHARED_FEATURES
from studies.ensemble import check_one_run_per_hour
from studies.product_frames import (
    Domain,
    DomainType,
)
from studies.resample import (
    step_means,
)

ENSEMBLE_SIZE: Final[int] = 51
"""The control member and 50 perturbed members."""


CONTROL_MEMBER: Final[int] = 0
"""The ensemble member ECMWF runs from the unperturbed analysis."""


def fields(*, domain: DomainType) -> tuple[str, ...]:
    """Return the weather fields an ENS arm is shown, in the order it is shown them.

    Args:
        domain: `solar` or `wind`.

    Returns:
        The field names every ENS arm's columns end in.
    """
    if domain == "solar":
        return ("ghi", "temp")
    return ("speed_100m", "sin_100m", "cos_100m", "speed_10m")


def ens_columns(*, arm: str, domain: DomainType) -> tuple[str, ...]:
    """Return one ENS arm's weather columns.

    Args:
        arm: The arm's name.
        domain: `solar` or `wind`.

    Returns:
        `<arm>_<field>` for each of `fields`.
    """
    return tuple(f"{arm}_{field}" for field in fields(domain=domain))


@dataclass(frozen=True)
class Steps:
    """One band's ENS members on their native steps, one row per (site, run, member)."""

    keys: pl.DataFrame
    """`site`, `init_time`, and `ensemble_member`, sorted, every (site, run) holding all members."""
    leads: np.ndarray
    """Each step's lead in hours."""
    widths: np.ndarray
    """How many hours each step's radiation averages over."""
    values: dict[str, np.ndarray]
    """Each field's values, shape (n_series, n_steps)."""
    ensemble_size: int = ENSEMBLE_SIZE
    """How many members each kept run holds. ENS's 51 by default; a caller building a different
    ensemble's steps (GEFS's 31, say) passes its own count through `band_steps`."""


def clear_sky_arrays(
    *, steps: Steps, targets: np.ndarray, clear_sky: pl.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Return the clear-sky mean over each step and over each target hour, per series.

    Args:
        steps: The band's steps.
        targets: The target hours' leads (each hour's end).
        clear_sky: The hourly clear-sky table from `hourly_clear_sky`.

    Returns:
        Shapes (n_series, n_steps) and (n_series, n_targets).

    Raises:
        ValueError: If a run's clear-sky hours are missing from the table.
    """
    runs = steps.keys.select("site", "init_time").unique(maintain_order=True)
    first_hour = int(min(steps.leads[0] - steps.widths[0] + 1, targets[0]))
    last_hour = int(max(steps.leads[-1], targets[-1]))
    hours = np.arange(first_hour, last_hour + 1)
    table = (
        runs.with_row_index("run")
        .join(pl.DataFrame({"lead": hours}), how="cross")
        .with_columns(time=pl.col("init_time") + pl.duration(hours=pl.col("lead")))
        .join(clear_sky, on=["site", "time"], how="left")
        .sort("run", "lead")
    )
    if table["clear_sky_w_m2"].null_count():
        msg = "a run's clear-sky hours are missing from the table"
        raise ValueError(msg)
    per_hour = table["clear_sky_w_m2"].to_numpy().reshape(runs.height, len(hours))
    run_of_series = np.repeat(np.arange(runs.height), steps.ensemble_size)
    return (
        step_means(
            hourly=per_hour, first_hour=first_hour, step_leads=steps.leads, step_widths=steps.widths
        )[run_of_series],
        per_hour[:, (targets - first_hour).astype(int)][run_of_series],
    )


def long_frame(*, steps: Steps, targets: np.ndarray, values: dict[str, np.ndarray]) -> pl.DataFrame:
    """Turn arrays over (series, target) into rows keyed by site, run, time, and member.

    Args:
        steps: The band's steps, whose keys name each series.
        targets: Each column's lead.
        values: Each field's array, shape (n_series, n_targets).

    Returns:
        One row per (site, time, member), with the run's `init_time`.
    """
    repeat = len(targets)
    keys = steps.keys.select(
        pl.col("site", "init_time", "ensemble_member").repeat_by(repeat).explode()
    )
    return keys.with_columns(
        lead=pl.Series(np.tile(targets, steps.keys.height)),
        **{name: pl.Series(array.reshape(-1).astype(np.float64)) for name, array in values.items()},
    ).select(
        "site",
        *values,
        init_time=pl.col("init_time").dt.replace_time_zone("UTC", non_existent="raise"),
        time=pl.col("init_time").dt.replace_time_zone("UTC", non_existent="raise")
        + pl.duration(hours=pl.col("lead").cast(pl.Int64)),
        member=pl.col("ensemble_member").cast(pl.Int32),
    )


def reduce_members(
    *, hourly: pl.DataFrame, domain: DomainType, way: str, ensemble_size: int = ENSEMBLE_SIZE
) -> pl.DataFrame:
    """Reduce every member's hourly fields to one value per (site, time).

    Args:
        hourly: One row per (site, time, member), with `init_time`.
        domain: `solar` or `wind`.
        way: `control` for the control member's own values, `mean` for the ensemble mean.
        ensemble_size: How many members each (site, time) must hold. ENS's 51 by default.

    Returns:
        One row per (site, time), with `fields(domain=domain)`.

    Raises:
        ValueError: Unless every (site, time) holds one run and all its members.
    """
    check_one_run_per_hour(hourly=hourly, members=ensemble_size)
    if way == "control":
        return hourly.filter(pl.col("member") == CONTROL_MEMBER).drop("member", "init_time")
    if domain == "solar":
        return hourly.group_by("site", "time").agg(pl.col("ghi", "temp").mean())
    # The mean direction is the mean wind vector's: each member's direction weighted by its speed.
    return (
        hourly.group_by("site", "time")
        .agg(
            pl.col("speed_100m", "speed_10m").mean(),
            east=(pl.col("speed_100m") * pl.col("sin_100m")).mean(),
            north=(pl.col("speed_100m") * pl.col("cos_100m")).mean(),
        )
        .with_columns(norm=(pl.col("east") ** 2 + pl.col("north") ** 2).sqrt())
        .select(
            "site",
            "time",
            "speed_100m",
            sin_100m=pl.col("east") / pl.col("norm"),
            cos_100m=pl.col("north") / pl.col("norm"),
            speed_10m="speed_10m",
        )
    )


def prefixed(*, frame: pl.DataFrame, arm: str, domain: DomainType) -> pl.DataFrame:
    """Rename a reduced frame's fields to one arm's column names.

    Args:
        frame: One row per (site, time), with `fields(domain=domain)`.
        arm: The arm.
        domain: `solar` or `wind`.

    Returns:
        `site`, `time`, and the arm's columns.
    """
    return frame.select(
        "site",
        "time",
        *(
            pl.col(field).alias(column)
            for field, column in zip(
                fields(domain=domain), ens_columns(arm=arm, domain=domain), strict=True
            )
        ),
    )


def shared_features(*, domain: Domain) -> tuple[str, ...]:
    """Return the columns every ENS arm is shown besides its weather.

    Args:
        domain: `product_frames.SOLAR` or `product_frames.WIND`.

    Returns:
        For solar, the past-weather studies' shared columns without ERA5's temperature, which each
        ENS arm replaces with its own; for wind, the wind study's shared columns.
    """
    if domain.name == "solar":
        return (*(c for c in SOLAR_SHARED_FEATURES if c != "temp_c"), "era_code")
    return domain.shared_features
