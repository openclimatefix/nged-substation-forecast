"""Join past IFS forecasts onto observed rows at a matched lead day, and cut folds by model era.

Written for the IFS solar-variables study planned in
<https://github.com/openclimatefix/nged-substation-forecast/pull/1137>. The study asks which IFS
forecast variables help predict solar farm output at forecast lead days 1 to 9, using one 00 UTC
run a day from Open-Meteo's Single Runs archive.

**A lead day is a whole valid day.** Lead day `L` of a valid hour at hour `h` of its UTC day comes
from the run that began at 00 UTC `L` days before that day, at lead `24 L + h` hours. Every valid
hour of one day therefore comes from one run, and a day-ahead decision made in the morning of the
day before (lead day 1) is the case the production service meets.

**The IFS cycle changed twice in the archive.** Cycle 49r1 went live on 2024-11-12 and cycle 50r1
on 2026-05-12, and Open-Meteo labels the runs before 2024-11-12 as 49r1 hindcasts. The study
therefore drops the two months that straddle the changes, labels the two eras that hold whole
months, drops the months after the second change (which are too few to cut into folds), and cuts
the folds inside each era.
"""

from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import Final

import polars as pl

from studies.cross_validation import cut_eras, search_fold_offsets

LEAD_DAYS: Final[tuple[int, ...]] = (1, 2, 3, 5, 7, 9)
"""The lead days the study scores. Lead day 10 holds one hour of the run, so it is not used."""

HOURS_PER_DAY: Final[int] = 24

DROPPED_MONTHS: Final[frozenset[str]] = frozenset({"2024-11", "2026-05"})
"""The months that straddle a change of IFS cycle, as `%Y-%m`."""

LAST_ERA_MONTH_EXCLUSIVE: Final[str] = "2026-05"
"""Rows from this month on follow the second cycle change and are dropped."""

FIRST_MONTHS_AFTER_FIRST_ERA: Final[tuple[str, ...]] = ("2024-12",)
"""The first month of every era after the first, as `%Y-%m`."""


def join_forecast_at_lead_day(
    *, rows: pl.DataFrame, forecasts: pl.DataFrame, lead_day: int
) -> pl.DataFrame:
    """Join the forecast issued `lead_day` days before each row's valid day onto the row.

    Args:
        rows: Observed rows carrying `site` and `time`, the valid hour as a UTC datetime.
        forecasts: One row per (`site`, `init_time`, `lead_hours`), with `init_time` a naive UTC
            datetime at 00 UTC, plus the forecast variables.
        lead_day: The whole number of days between the run's day and the valid day.

    Returns:
        The rows that have a forecast, with the forecast variables and `lead_hours` joined on.
        A row whose run is missing from `forecasts` is dropped.

    Raises:
        ValueError: If `lead_day` is not positive, or a run in the forecasts does not start at
            00 UTC.
    """
    if lead_day < 1:
        msg = f"lead_day must be at least 1, got {lead_day}"
        raise ValueError(msg)
    run_hours = forecasts["init_time"].dt.hour().unique().to_list()
    if run_hours != [0]:
        msg = f"the forecasts must all start at 00 UTC, got hours {sorted(run_hours)}"
        raise ValueError(msg)
    keyed = rows.with_columns(
        valid_time=pl.col("time").dt.replace_time_zone(None),
        lead_hours=(HOURS_PER_DAY * lead_day + pl.col("time").dt.hour()).cast(pl.Int32),
    ).with_columns(
        init_time=(pl.col("valid_time").dt.truncate("1d") - timedelta(days=lead_day)).cast(
            forecasts.schema["init_time"]
        )
    )
    joined = keyed.join(
        forecasts.drop("valid_time"),
        on=["site", "init_time", "lead_hours"],
        how="inner",
    )
    return joined.with_columns(lead_day=pl.lit(lead_day, dtype=pl.Int8))


def with_ifs_eras(*, rows: pl.DataFrame, fold_offsets: Mapping[int, int]) -> pl.DataFrame:
    """Drop the months that straddle a cycle change or follow the last one, label eras, cut folds.

    Args:
        rows: Rows carrying `site` and `month` (`%Y-%m`).
        fold_offsets: The fold rotation of each era, from `find_fold_offsets`.

    Returns:
        The kept rows with `era_code`, `era`, and `fold`.
    """
    kept = rows.filter(
        ~pl.col("month").is_in(sorted(DROPPED_MONTHS))
        & (pl.col("month") < LAST_ERA_MONTH_EXCLUSIVE)
    )
    return cut_eras(
        frame=kept, first_months=FIRST_MONTHS_AFTER_FIRST_ERA, fold_offsets=fold_offsets
    )


def find_fold_offsets(*, rows: pl.DataFrame) -> Mapping[int, int]:
    """Find the fold rotation of each era that leaves no calendar month untrained.

    Args:
        rows: Rows carrying `site`, `month`, and `time`.

    Returns:
        The first design `search_fold_offsets` finds.

    Raises:
        ValueError: If no rotation covers every calendar month.
    """
    kept = rows.filter(
        ~pl.col("month").is_in(sorted(DROPPED_MONTHS))
        & (pl.col("month") < LAST_ERA_MONTH_EXCLUSIVE)
    )
    designs = search_fold_offsets(frame=kept, first_months=FIRST_MONTHS_AFTER_FIRST_ERA)
    if not designs:
        msg = "no fold rotation leaves every calendar month in the training rows of a fold"
        raise ValueError(msg)
    return designs[0]


def raise_unless_same_folds(*, frames: Sequence[pl.DataFrame]) -> None:
    """Raise unless every lead-day frame gives each (site, valid hour) the same fold.

    A valid hour that falls in a held-out fold at one lead day must fall in the same fold at every
    lead day that holds it, or the lead-day models would train on each other's test hours.

    Args:
        frames: One frame per lead day, carrying `site`, `time`, and `fold`.

    Raises:
        ValueError: If any (site, time) has two different folds across the frames.
    """
    stacked = pl.concat([frame.select("site", "time", "fold") for frame in frames]).unique()
    clashes = (
        stacked.group_by("site", "time").agg(n=pl.col("fold").n_unique()).filter(pl.col("n") > 1)
    )
    if not clashes.is_empty():
        msg = f"{clashes.height} (site, time) rows carry different folds at different lead days"
        raise ValueError(msg)
