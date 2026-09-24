"""Power forecasts that read no weather forecast, as baselines a weather-driven forecast must beat.

**Every baseline sees only the telemetry up to the forecast's issue time.** A study compares them
with a weather-driven forecast issued at the same instant, so both see the same information. The
telemetry is taken as available up to the issue time, with no delivery delay, which is the best case
for a persistence baseline. It is NGED's hourly power as published, with no cleaning: the hours a
live service would have seen.

**A forecast for the run's own day is cut off at the run's 00 UTC init time, not at 09:00.** An
issue at 09:00 on the target day would hand the baseline the target's own morning hours, which the
weather run it is compared with never saw; the run's init time is that run's own information
cut-off.

**An hour counts as observed at the issue time only once it has ended.** A solar hour is labelled by
its end, so the last observed solar hour is the one ending at the issue time. A wind hour is centred
on its label, so the last observed wind hour at 09:00 is the one labelled 08:00, covering 07:30 to
08:30; `persistence` takes that half-hour as its `observed_lag`.

Four baselines:

- **Persistence**: the last observed hour's power, held for every target hour. Where that hour is
  missing, the latest observed hour before it, up to 24 hours earlier.
- **Diurnal persistence**: the power at the same time of day on the last whole UTC day before the
  issue day. Where that hour is missing, the same time of day on the latest earlier day, up to 7
  days before.
- **Smart persistence.** For solar, persistence of the clear-sky index: the observed energy over the
  24 hours before the issue time divided by the clear-sky energy over the hours observed, times the
  clear-sky irradiance at the target hour. Clear-sky irradiance is the Haurwitz model's mean over
  each hour (`hourly_clear_sky`). The 24-hour window is what keeps the denominator away from zero: a
  09:00 UTC issue in December falls barely an hour after sunrise, where a one-hour clear-sky index
  divides noise by almost nothing, and 24 hours always hold a whole day's sunlight. A window whose
  observed hours hold less than half its clear-sky energy gives no forecast. For wind, persistence
  shrunk towards climatology: a weight on persistence and one minus that weight on climatology, the
  weight chosen per generator, fold, and forecast horizon to minimise the mean absolute error on the
  training folds, from 0 to 1 in steps of 0.01.
- **Climatology**: the median power at each generator in each calendar month and hour of day, over
  the training folds only, curtailed hours left out as the study's XGBoost models leave them out.
  The median rather than the mean, because the score is the mean absolute error, which the median
  minimises. **Fallback rule, chosen before looking at any result it would affect:** where a month
  and hour have no training rows — a fold cut within a UKV era can leave a calendar month's rows all
  on the scored side, which happens for June and July at the wind farms and for the shortest-history
  solar farm — the median of the two neighbouring calendar months at that hour, over the training
  folds, is used instead. Where even that is empty, the generator's median at that hour over every
  training month is used.
"""

from datetime import datetime, timedelta
from typing import Final

import numpy as np
import polars as pl

from studies.cross_validation import N_FOLDS
from studies.solar import zenith

ISSUE_DELAY: Final[timedelta] = timedelta(hours=9)
"""How long after a run's 00 UTC init time the forecast is issued.

`ml_core.features._nwp.NWP_PUBLICATION_DELAY_HOURS` models the live service reading a 00 UTC ECMWF
ENS run from 09:00 UTC, and the baselines are issued at the same instant.
"""

PERSISTENCE_TOLERANCE: Final[timedelta] = timedelta(hours=24)
"""How far before the issue time persistence looks for an observed hour."""

DIURNAL_TOLERANCE: Final[timedelta] = timedelta(days=7)
"""How many days back diurnal persistence looks for the same time of day."""

CLEAR_SKY_WINDOW: Final[timedelta] = timedelta(hours=24)
"""The window before the issue time the clear-sky index is taken over."""

MIN_OBSERVED_CLEAR_SKY_SHARE: Final[float] = 0.5
"""The least share of the window's clear-sky energy its observed hours must hold."""

CLEAR_SKY_SAMPLE_MINUTES: Final[tuple[int, ...]] = (5, 15, 25, 35, 45, 55)
"""How many minutes before each hour's end the clear-sky model is sampled at."""

SHRINKAGE_WEIGHTS: Final[tuple[float, ...]] = tuple(step / 100.0 for step in range(101))
"""The persistence weights the wind smart persistence chooses among."""


def issue_time(*, day_start: pl.Expr, day: int) -> pl.Expr:
    """Return the issue time of band `day`'s forecast for the target day starting at `day_start`.

    Args:
        day_start: Midnight UTC at the start of the target day.
        day: The band's day: how many days after the run's own day the target day falls.

    Returns:
        09:00 UTC on the run's own day, or for the run's own day itself, the run's 00 UTC init
        time.
    """
    delay = ISSUE_DELAY if day > 0 else timedelta(0)
    return day_start - pl.duration(days=day) + pl.lit(delay)


def persistence(*, keys: pl.DataFrame, hourly: pl.DataFrame, observed_lag: timedelta) -> pl.Series:
    """Return the last observed hour's power at each key's issue time.

    Args:
        keys: One row per scored row, with `site` and `issue_time`, in the order to return.
        hourly: NGED's hourly power, with `site`, `time`, and `power_mw`.
        observed_lag: How long after its label an hour ends: 0 for solar, 30 minutes for wind.

    Returns:
        The power, null where no hour is observed within `PERSISTENCE_TOLERANCE`.
    """
    observed = hourly.select("site", "time", "power_mw").sort("site", "time")
    looked_up = (
        keys.with_row_index("row")
        .with_columns(time=pl.col("issue_time") - pl.lit(observed_lag))
        .sort("site", "time")
        .join_asof(
            observed, on="time", by="site", strategy="backward", tolerance=PERSISTENCE_TOLERANCE
        )
        .sort("row")
    )
    return looked_up["power_mw"]


def diurnal_persistence(*, keys: pl.DataFrame, hourly: pl.DataFrame, day: int) -> pl.Series:
    """Return the power at each row's time of day on the last whole day before the issue day.

    Args:
        keys: One row per scored row, with `site` and `time`, in the order to return.
        hourly: NGED's hourly power, with `site`, `time`, and `power_mw`.
        day: The band's day, which puts the last whole day `day + 1` days before the target day.

    Returns:
        The power, null where no day within `DIURNAL_TOLERANCE` has that hour.
    """
    observed = (
        hourly.select("site", "time", "power_mw")
        .with_columns(clock=pl.col("time").dt.time())
        .sort("site", "clock", "time")
    )
    looked_up = (
        keys.select("site", "time")
        .with_row_index("row")
        .with_columns(time=pl.col("time") - pl.duration(days=day + 1))
        .with_columns(clock=pl.col("time").dt.time())
        .sort("site", "clock", "time")
        .join_asof(
            observed,
            on="time",
            by=["site", "clock"],
            strategy="backward",
            tolerance=DIURNAL_TOLERANCE,
        )
        .sort("row")
    )
    return looked_up["power_mw"]


def haurwitz_w_m2(*, apparent_zenith_deg: np.ndarray) -> np.ndarray:
    """Return the Haurwitz model's clear-sky global horizontal irradiance.

    The formula `pvlib.clearsky.haurwitz` implements, written out because that function takes a
    pandas object: 1098 W m⁻² times the cosine of the zenith angle times `exp(-0.059 / cos)`, and
    zero with the sun at or below the horizon.

    Args:
        apparent_zenith_deg: The apparent solar zenith angle in degrees.

    Returns:
        The irradiance in W m⁻².
    """
    cosine = np.cos(np.radians(apparent_zenith_deg))
    safe = np.where(cosine > 0.0, cosine, 1.0)
    return np.where(cosine > 0.0, 1098.0 * safe * np.exp(-0.059 / safe), 0.0)


def hourly_clear_sky(*, sites: pl.DataFrame, first: datetime, last: datetime) -> pl.DataFrame:
    """Return the Haurwitz clear-sky global irradiance averaged over every hour, at each generator.

    Each hour's value is the mean of six samples, at 5, 15, 25, 35, 45 and 55 minutes into the hour,
    of the Haurwitz model at the solar zenith angle `studies.solar.zenith` computes, so it is a
    period mean over the hour ending at its label, like the power and like ENS's radiation.

    Args:
        sites: The roster, with `site`, `latitude`, and `longitude`. Coordinates are read here and
            never written.
        first: The first hour's end.
        last: The last hour's end.

    Returns:
        One row per generator and hour, with `site`, `time` (the hour's end), and
        `clear_sky_w_m2`.
    """
    hours = pl.datetime_range(first, last, interval="1h", eager=True, time_zone="UTC")
    samples = pl.concat(
        [hours.dt.offset_by(f"-{minutes}m") for minutes in CLEAR_SKY_SAMPLE_MINUTES]
    )
    parts = []
    for site, latitude, longitude in sites.select("site", "latitude", "longitude").iter_rows():
        apparent_zenith = zenith(stamps=samples, latitude=latitude, longitude=longitude)
        irradiance = haurwitz_w_m2(apparent_zenith_deg=apparent_zenith)
        per_sample = irradiance.reshape(len(CLEAR_SKY_SAMPLE_MINUTES), len(hours))
        parts.append(
            pl.DataFrame(
                {
                    "site": [site] * len(hours),
                    "time": hours,
                    "clear_sky_w_m2": per_sample.mean(axis=0),
                }
            )
        )
    return pl.concat(parts)


def clear_sky_index(*, keys: pl.DataFrame, hourly: pl.DataFrame) -> pl.Series:
    """Return the clear-sky index over the 24 hours before each key's issue time.

    Args:
        keys: One row per scored row, with `site` and `issue_time`, in the order to return.
        hourly: NGED's hourly power with `clear_sky_w_m2`, for every hour of a whole-hour grid,
            `power_mw` null where no reading exists.

    Returns:
        Megawatts per W m⁻² of clear-sky irradiance, null where the observed hours hold less than
        `MIN_OBSERVED_CLEAR_SKY_SHARE` of the window's clear-sky energy.
    """
    issues = keys.select("site", "issue_time").unique()
    windows = (
        issues.join(hourly, on="site")
        .filter(
            (pl.col("time") > pl.col("issue_time") - pl.lit(CLEAR_SKY_WINDOW))
            & (pl.col("time") <= pl.col("issue_time"))
        )
        .group_by("site", "issue_time")
        .agg(
            energy=pl.col("power_mw").sum(),
            observed_clear_sky=pl.col("clear_sky_w_m2")
            .filter(pl.col("power_mw").is_not_null())
            .sum(),
            clear_sky=pl.col("clear_sky_w_m2").sum(),
        )
        .with_columns(
            index=pl.when(
                pl.col("observed_clear_sky") >= MIN_OBSERVED_CLEAR_SKY_SHARE * pl.col("clear_sky")
            ).then(pl.col("energy") / pl.col("observed_clear_sky"))
        )
    )
    return keys.join(windows, on=["site", "issue_time"], how="left", maintain_order="left")["index"]


def hourly_grid(*, hourly: pl.DataFrame) -> pl.DataFrame:
    """Put the hourly power on a whole-hour grid per generator, null where no reading exists.

    Args:
        hourly: NGED's hourly power, with `site`, `time`, and `power_mw`.

    Returns:
        One row per generator and hour from its first reading to its last.
    """
    spans = hourly.group_by("site").agg(first=pl.col("time").min(), last=pl.col("time").max())
    grid = spans.select(
        "site", time=pl.datetime_ranges(pl.col("first"), pl.col("last"), interval="1h")
    ).explode("time")
    return grid.join(hourly.select("site", "time", "power_mw"), on=["site", "time"], how="left")


def _mod_month(expr: pl.Expr) -> pl.Expr:
    """Return `expr` wrapped back into the 1-to-12 calendar-month range."""
    return ((expr - 1) % 12) + 1


def _climatology_from(*, train: pl.DataFrame, rows: pl.DataFrame) -> pl.Series:
    """Return each of `rows`' median training power at its generator, calendar month, and hour.

    Args:
        train: The training rows, with `site`, `time`, `constrained`, and `power_mw`.
        rows: The rows to forecast, with `site` and `time`.

    Returns:
        The median, one per row of `rows` in its order: the month-and-hour median; or, where the
        training rows hold no such month and hour, the median of the two neighbouring calendar
        months at that hour; or, where even that is empty, the generator's median at that hour over
        every training month.
    """
    keys = {"calendar_month": pl.col("time").dt.month(), "hour": pl.col("time").dt.hour()}
    kept = train.filter(~pl.col("constrained")).with_columns(**keys)
    by_month = kept.group_by("site", "calendar_month", "hour").agg(
        month_median=pl.col("power_mw").median()
    )
    neighbour_pool = pl.concat(
        [
            kept.with_columns(calendar_month=_mod_month(pl.col("calendar_month") + 1)),
            kept.with_columns(calendar_month=_mod_month(pl.col("calendar_month") - 1)),
        ]
    )
    by_neighbour = neighbour_pool.group_by("site", "calendar_month", "hour").agg(
        neighbour_median=pl.col("power_mw").median()
    )
    by_hour = kept.group_by("site", "hour").agg(hour_median=pl.col("power_mw").median())
    return (
        rows.select("site", "time")
        .with_columns(**keys)
        .join(by_month, on=["site", "calendar_month", "hour"], how="left", maintain_order="left")
        .join(
            by_neighbour, on=["site", "calendar_month", "hour"], how="left", maintain_order="left"
        )
        .join(by_hour, on=["site", "hour"], how="left", maintain_order="left")
        .select(climatology=pl.coalesce("month_median", "neighbour_median", "hour_median"))[
            "climatology"
        ]
    )


def climatology(*, frame: pl.DataFrame) -> pl.Series:
    """Return each row's out-of-fold median power for its generator, calendar month, and hour.

    Args:
        frame: The scored rows, with `site`, `time`, `fold`, `constrained`, and `power_mw`.

    Returns:
        The median, one per row in `frame`'s order.
    """
    keyed = frame.with_row_index("row")
    parts = []
    for fold in range(N_FOLDS):
        test = keyed.filter(pl.col("fold") == fold)
        parts.append(
            test.select("row").with_columns(
                climatology=_climatology_from(train=keyed.filter(pl.col("fold") != fold), rows=test)
            )
        )
    return pl.concat(parts).sort("row")["climatology"]


def shrunk_persistence(*, frame: pl.DataFrame, persisted: str) -> pl.DataFrame:
    """Blend persistence with climatology, the weight fitted per generator on the training folds.

    For each fold, climatology is taken from the other folds alone, and that one climatology is
    both what the weight is fitted against on the training folds and what the scored fold is
    forecast with, so nothing from the scored fold reaches the weight. Curtailed hours are left out
    of the fit.

    Args:
        frame: The scored rows, with `site`, `time`, `fold`, `constrained`, `power_mw`, and the
            persistence forecast named below.
        persisted: The persistence forecast's column.

    Returns:
        One row per row of `frame`, in its order, with the blended forecast `shrunk` and the
        persistence `weight` it used.
    """
    weights = np.asarray(SHRINKAGE_WEIGHTS)
    keyed = frame.with_row_index("row")
    parts = []
    for _site, rows in keyed.group_by(["site"], maintain_order=True):
        for fold in range(N_FOLDS):
            train = rows.filter((pl.col("fold") != fold) & ~pl.col("constrained"))
            test = rows.filter(pl.col("fold") == fold)
            if test.is_empty() or train.is_empty():
                continue
            train_climate = _climatology_from(train=train, rows=train).to_numpy()
            blended = (
                weights[None, :] * train[persisted].to_numpy()[:, None]
                + (1.0 - weights[None, :]) * train_climate[:, None]
            )
            errors = np.abs(blended - train["power_mw"].to_numpy()[:, None]).mean(axis=0)
            weight = float(weights[errors.argmin()])
            test_climate = _climatology_from(train=train, rows=test)
            parts.append(
                test.select("row")
                .with_columns(climate=test_climate)
                .with_columns(
                    shrunk=weight * test[persisted] + (1.0 - weight) * pl.col("climate"),
                    weight=pl.lit(weight),
                )
                .drop("climate")
            )
    return pl.concat(parts).sort("row").drop("row")
