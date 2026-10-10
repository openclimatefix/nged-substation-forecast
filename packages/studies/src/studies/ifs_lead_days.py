"""Join past IFS forecasts onto observed rows at a matched lead day, and cut folds by model era.

Written for the IFS solar-variables study planned in
<https://github.com/openclimatefix/nged-substation-forecast/pull/1137>. The study asks which IFS
forecast variables help predict solar farm output at forecast lead days 1 to 9, using one 00 UTC
run a day from Open-Meteo's Single Runs archive.

**A lead day is a whole valid day.** Lead day `L` of a valid hour at hour `h` of its UTC day comes
from the run that began at 00 UTC `L` days before that day, at lead `24 L + h` hours. Every valid
hour of one day therefore comes from one run, and a day-ahead decision made in the morning of the
day before (lead day 1) is the case the production service meets.

**A value at the end of the hour is averaged with the value an hour earlier.** A row's `time` ends
its hour. Radiation, precipitation, and snowfall are means or totals over the hour ending at the
valid time, and gusts are the maximum over that hour, so those columns are used as they are. Every
other variable is a value at one instant, so the join replaces it by the mean of its values at
lead hours `24 L + h - 1` and `24 L + h` of the same run, as the ERA5 study averaged ERA5's
snapshots over the hour's two ends. Hour 0 therefore reads the last lead hour of the previous day
of the same run. A missing neighbour makes the averaged value missing. Night labels are absent from
the rows, so the hour-ending convention of `studies.ifs_single_runs` names the same run as the
label's own day.

**The first era starts in the middle of March 2024.** The archive's first run is on 2024-03-14, so
the first era runs from 2024-03-15 to 2024-10: 8 months, the first of them a part-month of 17 days
at lead day 1 whose length falls with the lead day. The two eras hold 25 months. Dropping the part
month leaves no fold rotation that covers every calendar month for every farm.

**The IFS cycle changed twice in the archive.** Cycle 49r1 went live on 2024-11-12 and cycle 50r1
on 2026-05-12, and Open-Meteo labels the runs before 2024-11-12 as 49r1 hindcasts. The study
therefore drops the two months that straddle the changes, labels the two eras that hold whole
months, drops the months after the second change (which are too few to cut into folds), and cuts
the folds inside each era.
"""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Final

import polars as pl

from studies.cross_validation import N_FOLDS, cut_eras, search_fold_offsets

LEAD_DAYS: Final[tuple[int, ...]] = (1, 2, 3, 5, 7, 9)
"""The lead days the study scores. Lead day 10 holds one hour of the run, so it is not used."""

HOURS_PER_DAY: Final[int] = 24

INSTANTANEOUS_VARIABLES: Final[frozenset[str]] = frozenset(
    {
        "cloud_cover",
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
        "dew_point_2m",
        "temperature_2m",
        "surface_pressure",
        "boundary_layer_height",
        "total_column_integrated_water_vapour",
        "cape",
        "convective_inhibition",
        "visibility",
        "surface_temperature",
        "snow_depth",
        "wind_speed_10m",
    }
)
"""The variables that are values at one instant, averaged over the hour's two ends by the join.

Radiation, precipitation, snowfall, and gusts are means, totals, or maxima over the hour ending at
the valid time, so they are not listed.
"""

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
        A row whose run is missing from `forecasts` is dropped. Each variable of
        `INSTANTANEOUS_VARIABLES` is the mean of its values at lead hours `24 * lead_day + h - 1`
        and `24 * lead_day + h`, and is missing if either value is.

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
    forecasts = forecasts.drop("valid_time")
    instantaneous = [name for name in forecasts.columns if name in INSTANTANEOUS_VARIABLES]
    earlier = forecasts.select(
        "site",
        "init_time",
        (pl.col("lead_hours") + 1).alias("lead_hours"),
        *(pl.col(name).alias(f"{name}__earlier") for name in instantaneous),
    )
    forecasts = (
        forecasts.join(earlier, on=["site", "init_time", "lead_hours"], how="left")
        .with_columns(
            ((pl.col(name) + pl.col(f"{name}__earlier")) / 2.0).alias(name)
            for name in instantaneous
        )
        .drop(f"{name}__earlier" for name in instantaneous)
    )
    keyed = rows.with_columns(
        valid_time=pl.col("time").dt.replace_time_zone(None),
        # The hour is Int8, so it is widened before the sum: 120 + 13 would wrap to -123 in Int8.
        lead_hours=HOURS_PER_DAY * lead_day + pl.col("time").dt.hour().cast(pl.Int32),
    ).with_columns(
        init_time=(pl.col("valid_time").dt.truncate("1d") - timedelta(days=lead_day)).cast(
            forecasts.schema["init_time"]
        )
    )
    joined = keyed.join(
        forecasts,
        on=["site", "init_time", "lead_hours"],
        how="inner",
    )
    return joined.with_columns(lead_day=pl.lit(lead_day, dtype=pl.Int8))


def _kept_months(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Drop the months that straddle a cycle change or follow the last one."""
    return rows.filter(
        ~pl.col("month").is_in(sorted(DROPPED_MONTHS))
        & (pl.col("month") < LAST_ERA_MONTH_EXCLUSIVE)
    )


def with_ifs_eras(*, rows: pl.DataFrame, fold_offsets: Mapping[int, int]) -> pl.DataFrame:
    """Drop the months that straddle a cycle change or follow the last one, label eras, cut folds.

    Args:
        rows: Rows carrying `site` and `month` (`%Y-%m`).
        fold_offsets: The fold rotation of each era, from `find_fold_offsets`.

    Returns:
        The kept rows with `era_code`, `era`, and `fold`.
    """
    kept = _kept_months(rows=rows)
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
    kept = _kept_months(rows=rows)
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


AIFS_FIRST_OPERATIONAL_INIT: Final[datetime] = datetime(2025, 3, 1, tzinfo=UTC)
"""The first run of AIFS Single that the blend rows keep.

AIFS Single became operational on 2025-02-25, and the Dynamical.org store's earlier runs are
experimental versions, so the blend rows start with the runs of the first whole month after.
"""

PARTNER_COLUMNS: Final[tuple[str, str]] = ("partner_shortwave_radiation", "partner_temperature_2m")
"""The second forecast's radiation and temperature, under names that do not carry the lead day."""


def join_aifs_partner(*, rows: pl.DataFrame, partner: pl.DataFrame, lead_day: int) -> pl.DataFrame:
    """Join AIFS Single's radiation and temperature at one lead day onto the lead day's rows.

    A row is kept only if AIFS Single has both values for it and its run is on or after
    `AIFS_FIRST_OPERATIONAL_INIT`.

    Args:
        rows: The lead day's rows, carrying `site` and `time` (UTC).
        partner: One row per (`site`, `time`) with `aifs_single_day{lead_day}_ghi`, `_temp`, and
            `_init_time`, the run that supplied the values.
        lead_day: The whole number of days between the run's day and the valid day.

    Returns:
        The kept rows with the two `PARTNER_COLUMNS` joined on.

    Raises:
        ValueError: If a run is not the 00 UTC run `lead_day` days before the row's valid day, or
            `partner` holds a (site, time) twice.
    """
    prefix = f"aifs_single_day{lead_day}"
    columns = {f"{prefix}_ghi": PARTNER_COLUMNS[0], f"{prefix}_temp": PARTNER_COLUMNS[1]}
    if partner.select("site", "time").is_duplicated().any():
        msg = "the partner frame holds a (site, time) twice"
        raise ValueError(msg)
    kept = partner.select("site", "time", f"{prefix}_init_time", *columns).drop_nulls(
        subset=[f"{prefix}_init_time", *columns]
    )
    expected = pl.col("time").dt.truncate("1d") - timedelta(days=lead_day)
    wrong = kept.filter(
        pl.col(f"{prefix}_init_time").dt.cast_time_unit("us") != expected.dt.cast_time_unit("us")
    )
    if not wrong.is_empty():
        msg = (
            f"{wrong.height} rows carry an AIFS Single run that is not the 00 UTC run "
            f"{lead_day} days before the valid day"
        )
        raise ValueError(msg)
    recent = kept.filter(pl.col(f"{prefix}_init_time") >= AIFS_FIRST_OPERATIONAL_INIT)
    return rows.join(
        recent.drop(f"{prefix}_init_time").rename(columns), on=["site", "time"], how="inner"
    )


def blend_month_folds(*, months: Sequence[str]) -> dict[str, int]:
    """Cut the blend rows' months into `N_FOLDS` contiguous blocks, the same for every farm.

    The folds depend on the months alone, so every lead day that holds a month gives it the same
    fold, whatever other months its rows lack.

    Args:
        months: The `%Y-%m` months of the blend rows at any lead day.

    Returns:
        Each month's fold. The first months fall in fold 0.

    Raises:
        ValueError: If there are fewer months than folds.
    """
    ordered = sorted(set(months))
    if len(ordered) < N_FOLDS:
        msg = f"{len(ordered)} months cannot fill {N_FOLDS} folds"
        raise ValueError(msg)
    return {month: index * N_FOLDS // len(ordered) for index, month in enumerate(ordered)}


def with_blend_folds(*, rows: pl.DataFrame, month_folds: Mapping[str, int]) -> pl.DataFrame:
    """Give each row the fold of its month.

    Args:
        rows: Rows carrying `month`.
        month_folds: `blend_month_folds`'s result.

    Returns:
        The rows with an Int32 `fold`.

    Raises:
        ValueError: If a row's month has no fold.
    """
    unknown = sorted(set(rows["month"].unique().to_list()) - set(month_folds))
    if unknown:
        msg = f"no fold for the months {unknown}"
        raise ValueError(msg)
    return rows.with_columns(
        fold=pl.col("month").replace_strict(dict(month_folds), return_dtype=pl.Int32)
    )
