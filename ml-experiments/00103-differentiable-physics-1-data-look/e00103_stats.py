"""Statistics behind experiment 103's figures and decisions: tables built from the panel.

Each function takes the panel (`e00103_build.py`) and returns a small Polars frame. Power is in
points of the series' effective capacity (percent), so sites of different sizes share one axis.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Final

import numpy as np
import polars as pl
from e00103_common import (
    HOURS_BANDS,
    PANEL_END,
    PANEL_PATH,
    STAMP_FIX_UTC,
    load_series,
)

LAST_SCORED_DAY: Final[date] = date(2026, 6, 30)
"""The last local date of any counted or scored statistic."""
LOCKDOWN_FIRST_DAY: Final[date] = date(2020, 3, 23)
LOCKDOWN_LAST_DAY: Final[date] = date(2021, 7, 19)
"""The fixed lockdown window, experiment 6's COVID indicator, fixed by the series brief."""
LOCKDOWN_FIRST_MONTH: Final[tuple[int, int]] = (2020, 3)
LOCKDOWN_LAST_MONTH: Final[tuple[int, int]] = (2021, 7)
"""The calendar months the window touches, for statistics that work in whole months."""


def load_panel(*, extended: bool = False) -> pl.DataFrame:
    """The panel with `points` (power as percent of effective capacity) and calendar columns.

    Rows stop at 2026-06-30 unless `extended`, which keeps the panel's rows to 2026-09-10 for the
    descriptive timestamp and early-calibration checks.
    """
    capacity = load_series().select("time_series_id", "units", "effective_capacity_mw")
    return (
        pl.read_parquet(PANEL_PATH)
        .join(capacity, on="time_series_id")
        .filter(True if extended else pl.col("local_date") <= LAST_SCORED_DAY)
        .with_columns(
            clean=pl.col("present") & ~pl.col("flagged"),
            points=100.0 * pl.col("power") / pl.col("effective_capacity_mw"),
            month=pl.col("local_date").dt.truncate("1mo"),
        )
        .with_columns(
            power_clean=pl.when(pl.col("clean")).then(pl.col("points")),
            weekday=pl.col("local_date").dt.weekday(),
        )
    )


def band_expression(*, band: str) -> pl.Expr:
    """True for the half-hours (by local start hour) of `night`, `evening`, or `midday`."""
    start, end = HOURS_BANDS[band]
    return (pl.col("local_hour_start") >= start) & (pl.col("local_hour_start") < end)


def monthly_levels(*, panel: pl.DataFrame) -> pl.DataFrame:
    """Mean points per series and local month for the night, evening, and midday bands.

    Midday is split into summer (April to September) and winter (October to March). `weekday_day`
    is the weekday 09:00 to 16:59 mean used by the early-calibration and lockdown rules.
    """
    clean = panel.filter(pl.col("clean"))
    month = pl.col("month").dt.month()
    summer = (month >= 4) & (month <= 9)
    specs = {
        "night": band_expression(band="night"),
        "evening": band_expression(band="evening"),
        "summer_midday": band_expression(band="midday") & summer,
        "winter_midday": band_expression(band="midday") & ~summer,
        "weekday_day": (pl.col("local_hour_start") >= 9)
        & (pl.col("local_hour_start") < 17)
        & (pl.col("weekday") <= 5),
    }
    out = None
    for name, condition in specs.items():
        part = (
            clean.filter(condition)
            .group_by("time_series_id", "month")
            .agg(pl.col("points").mean().alias(name), pl.len().alias(f"{name}_n"))
        )
        out = (
            part
            if out is None
            else out.join(part, on=["time_series_id", "month"], how="full", coalesce=True)
        )
    assert out is not None
    return out.sort("time_series_id", "month")


def availability_by_month(*, panel: pl.DataFrame) -> pl.DataFrame:
    """Share of half-hours present and unflagged, per series and calendar month (UTC)."""
    return (
        panel.with_columns(m=pl.col("time").dt.truncate("1mo"))
        .group_by("time_series_id", "m")
        .agg(share=pl.col("clean").mean(), flagged=pl.col("flagged").sum(), rows=pl.len())
        .rename({"m": "month"})
        .sort("time_series_id", "month")
    )


def is_lockdown(*, month: pl.Expr) -> pl.Expr:
    """True for the calendar months the fixed lockdown window touches (2020-03 to 2021-07)."""
    index = month.dt.year() * 12 + month.dt.month()
    return (index >= LOCKDOWN_FIRST_MONTH[0] * 12 + LOCKDOWN_FIRST_MONTH[1]) & (
        index <= LOCKDOWN_LAST_MONTH[0] * 12 + LOCKDOWN_LAST_MONTH[1]
    )


def in_lockdown_window(*, local_date: pl.Expr) -> pl.Expr:
    """True for the local dates from 2020-03-23 to 2021-07-19 inclusive."""
    return local_date.is_between(LOCKDOWN_FIRST_DAY, LOCKDOWN_LAST_DAY)


def first_clean_month(*, panel: pl.DataFrame) -> pl.DataFrame:
    """The first unflagged half-hour per series."""
    return (
        panel.filter(pl.col("clean"))
        .group_by("time_series_id")
        .agg(first_clean=pl.col("time").min())
        .sort("time_series_id")
    )


def early_calibration(*, levels: pl.DataFrame, first_clean: pl.DataFrame) -> pl.DataFrame:
    """Each of the first 12 months' night and weekday-day levels minus a later month's.

    The comparison month is a year later, unless that month falls in the fixed lockdown window, in
    which case it is the first later year outside the lockdown window. `differs` is true when either
    band differs by more than 15 points.
    """
    keyed = levels.select(
        "time_series_id", "month", "night", "weekday_day", "night_n", "weekday_day_n"
    )
    # A series' first month may be a part-month; the comparison month must hold at least 10 days.
    rows = []
    for series_id, first in first_clean.iter_rows():
        start = first.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        own = keyed.filter(pl.col("time_series_id") == series_id)
        for k in range(12):
            month = _add_months(start, k)
            later = None
            for years in range(1, 7):
                candidate = _add_months(month, 12 * years)
                if candidate >= PANEL_END:
                    break
                in_lockdown = bool(
                    pl.DataFrame({"m": [candidate]}).select(
                        is_lockdown(month=pl.col("m")).alias("x")
                    )["x"][0]
                )
                if in_lockdown:
                    continue
                if own.filter(
                    pl.col("month").dt.date() == candidate.date(), pl.col("night_n") >= 100
                ).height:
                    later = candidate
                    break
            this = own.filter(pl.col("month").dt.date() == month.date())
            other = (
                own.filter(pl.col("month").dt.date() == later.date(), pl.col("night_n") >= 100)
                if later
                else own.clear()
            )
            if this.is_empty() or other.is_empty():
                continue
            night_diff = _difference(this["night"][0], other["night"][0])
            day_diff = _difference(this["weekday_day"][0], other["weekday_day"][0])
            if night_diff is None and day_diff is None:
                continue
            rows.append(
                {
                    "time_series_id": series_id,
                    "month_index": k,
                    "month": month,
                    "compared_with": later,
                    "night_diff": night_diff,
                    "weekday_day_diff": day_diff,
                }
            )
    return pl.DataFrame(
        rows, schema_overrides={"night_diff": pl.Float64, "weekday_day_diff": pl.Float64}
    ).with_columns(
        differs=(
            (pl.col("night_diff").abs() > 15) | (pl.col("weekday_day_diff").abs() > 15)
        ).fill_null(False)
    )


def _difference(first: float | None, second: float | None) -> float | None:
    """`first - second`, or None when either band has no data that month."""
    return None if first is None or second is None else first - second


def _add_months(stamp: datetime, months: int) -> datetime:
    """`stamp` (first of a month) moved by `months` calendar months."""
    total = stamp.year * 12 + (stamp.month - 1) + months
    return stamp.replace(year=total // 12, month=total % 12 + 1)


def timestamp_correlation(*, panel: pl.DataFrame, summer_middays: bool = False) -> pl.DataFrame:
    """Correlation of hourly mean power with CAMS GHI on clear-sky daylight hours, by shift.

    Callers restrict the panel to the months they compare; the like-for-like check uses April to
    September on both sides of the correction.

    The power hour is shifted by -90 to +90 minutes in 30-minute steps. Clear-sky hours: clear-sky
    index above 0.8 with clear-sky GHI above 100 W/m2, any month, MW Primaries only. With
    `summer_middays`, the plan's stricter test: May to August, CAMS hours ending 11:00 to 14:00 UTC,
    and clear-sky GHI above 400 W/m2 (net power falls
    as GHI rises, so the correlation is negative and the right shift makes it most negative). The
    split is at the ingest correction instant. Returns series, side, shift, correlation.
    """
    mw = (
        load_series()
        .filter(pl.col("units") == "MW", pl.col("group") == "primary demand")["time_series_id"]
        .to_list()
    )
    rows = []
    for shift in range(-90, 91, 30):
        frame = (
            panel.filter(pl.col("time_series_id").is_in(mw), pl.col("clean"))
            .select(
                "time_series_id",
                "time",
                "points",
                "ghi_w_m2",
                "clear_sky_ghi_w_m2",
                "cams_hour_end",
            )
            .with_columns(shifted=pl.col("time") + pl.duration(minutes=shift))
        )
        # the power value whose stamp is `time + shift` is read as the value at `time`
        hourly = (
            frame.with_columns(
                hour_end=(pl.col("time") - pl.duration(minutes=shift)).dt.truncate("1h")
                + pl.duration(
                    hours=pl.when((pl.col("time") - pl.duration(minutes=shift)).dt.minute() == 0)
                    .then(0)
                    .otherwise(1)
                )
            )
            .group_by("time_series_id", "hour_end")
            .agg(power=pl.col("points").mean(), n=pl.len())
            .filter(pl.col("n") == 2)
        )
        cams = frame.select(
            "time_series_id", "cams_hour_end", "ghi_w_m2", "clear_sky_ghi_w_m2"
        ).unique(subset=["time_series_id", "cams_hour_end"])
        joined = hourly.join(
            cams,
            left_on=["time_series_id", "hour_end"],
            right_on=["time_series_id", "cams_hour_end"],
        ).filter(
            pl.col("clear_sky_ghi_w_m2") > (400 if summer_middays else 100),
            pl.col("ghi_w_m2") / pl.col("clear_sky_ghi_w_m2") > 0.8,
        )
        if summer_middays:
            joined = joined.filter(
                pl.col("hour_end").dt.month().is_between(5, 8),
                pl.col("hour_end").dt.hour().is_between(11, 14),
            )
        for (series_id, after), part in joined.with_columns(
            after=pl.col("hour_end") >= STAMP_FIX_UTC
        ).group_by("time_series_id", "after"):
            if part.height < 30:
                continue
            # remove each series' own mean by day-of-year-agnostic demeaning of both variables
            corr = float(np.corrcoef(part["power"].to_numpy(), part["ghi_w_m2"].to_numpy())[0, 1])
            rows.append(
                {
                    "time_series_id": series_id,
                    "after_fix": after,
                    "shift_minutes": shift,
                    "correlation": corr,
                    "n": part.height,
                }
            )
    return pl.DataFrame(rows).sort("time_series_id", "after_fix", "shift_minutes")


HALF_HOURS_PER_SITE_YEAR: Final[float] = 365.25 * 48


def season_year(*, date: pl.Expr) -> pl.Expr:
    """The July-to-June season-year, named by its start year (2024 is July 2024 to June 2025)."""
    return pl.when(date.dt.month() >= 7).then(date.dt.year()).otherwise(date.dt.year() - 1)


def dark_site_years(*, panel: pl.DataFrame) -> pl.DataFrame:
    """Dark clean half-hours per series and season-year, in site-years, and the evening share.

    A site-year is 17,532 half-hours. The evening share is the share of dark clean half-hours that
    start between 16:00 and 21:59 local time.
    """
    return (
        panel.filter(pl.col("clean"), pl.col("dark"))
        .with_columns(season=season_year(date=pl.col("local_date")))
        .group_by("time_series_id", "season")
        .agg(
            dark_half_hours=pl.len(),
            evening=((pl.col("local_hour_start") >= 16) & (pl.col("local_hour_start") < 22)).sum(),
        )
        .with_columns(
            site_years=pl.col("dark_half_hours") / HALF_HOURS_PER_SITE_YEAR,
            evening_share=pl.col("evening") / pl.col("dark_half_hours"),
        )
        .sort("time_series_id", "season")
    )


def daily_table(*, panel: pl.DataFrame) -> pl.DataFrame:
    """Per series and local day: night mean, daily range, mean, daily-mean temperature, counts."""
    night = band_expression(band="night")
    return (
        panel.filter(pl.col("clean"))
        .group_by("time_series_id", "local_date")
        .agg(
            night_mean=pl.col("points").filter(night).mean(),
            night_n=night.sum(),
            day_n=pl.len(),
            day_mean=pl.col("points").mean(),
            day_range=pl.col("points").max() - pl.col("points").min(),
            temperature_c=pl.col("temperature_c").mean(),
        )
        .with_columns(month=pl.col("local_date").dt.truncate("1mo"))
        .sort("time_series_id", "local_date")
    )


def month_block_slopes(
    *, x: np.ndarray, y: np.ndarray, months: np.ndarray, n_resamples: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """Least-squares slope of y on x, and a 95% interval from resampling whole calendar months."""
    if x.size < 10:
        return float("nan"), float("nan"), float("nan")
    slope = float(np.polyfit(x, y, 1)[0])
    unique = np.unique(months)
    if unique.size < 3:
        return slope, float("nan"), float("nan")
    index = {m: np.flatnonzero(months == m) for m in unique}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples)
    for k in range(n_resamples):
        picked = np.concatenate([index[m] for m in rng.choice(unique, unique.size)])
        draws[k] = np.polyfit(x[picked], y[picked], 1)[0] if np.ptp(x[picked]) > 0 else np.nan
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return slope, float(low), float(high)


def winter_year(*, date: pl.Expr) -> pl.Expr:
    """Winter (November to February) named by its November's year; null outside those months."""
    month = date.dt.month()
    return pl.when(month >= 11).then(date.dt.year()).when(month <= 2).then(date.dt.year() - 1)


def heating_slopes(*, daily: pl.DataFrame, below_c: float = 12.0) -> pl.DataFrame:
    """Per series and winter: slope of daily night mean on daily-mean temperature below 12 degC.

    Points of effective capacity per degC, with a month-block 95% interval. Needs 20 days.
    """
    rows = []
    winter = daily.with_columns(winter=winter_year(date=pl.col("local_date"))).filter(
        pl.col("winter").is_not_null(),
        pl.col("temperature_c") < below_c,
        pl.col("night_n") >= 8,
        pl.col("night_mean").is_not_null(),
    )
    for (series_id, w), part in winter.group_by("time_series_id", "winter"):
        if part.height < 20:
            continue
        slope, low, high = month_block_slopes(
            x=part["temperature_c"].to_numpy().astype(float),
            y=part["night_mean"].to_numpy().astype(float),
            months=part["month"].dt.month().to_numpy(),
        )
        rows.append(
            {
                "time_series_id": series_id,
                "winter": w,
                "days": part.height,
                "slope": slope,
                "low": low,
                "high": high,
            }
        )
    return pl.DataFrame(rows).sort("time_series_id", "winter")


def clip_slopes(*, panel: pl.DataFrame) -> pl.DataFrame:
    """Per series and season-year: summer weekday midday power against CAMS GHI.

    Hourly mean net power (two half-hours per CAMS hour) for May to August weekdays whose CAMS hour
    ends between 11:00 and 14:00 UTC (so the hour starts 10:00 to 13:00). Slopes in points of
    effective capacity per 100 W/m2.
    """
    hourly = (
        panel.filter(
            pl.col("clean"),
            pl.col("local_date").dt.month().is_between(5, 8),
            pl.col("weekday") <= 5,
            pl.col("cams_hour_end").dt.hour().is_between(11, 14),
        )
        .group_by("time_series_id", "cams_hour_end")
        .agg(power=pl.col("points").mean(), ghi=pl.col("ghi_w_m2").first(), n=pl.len())
        .filter(pl.col("n") == 2)
        .with_columns(season=season_year(date=pl.col("cams_hour_end")))
    )
    rows = []
    for (series_id, season), part in hourly.group_by("time_series_id", "season"):
        record = {"time_series_id": series_id, "season": season, "hours": part.height}
        for name, condition in (
            ("below_500", pl.col("ghi") < 500),
            ("above_600", pl.col("ghi") > 600),
        ):
            sub = part.filter(condition)
            if sub.height >= 30 and np.ptp(sub["ghi"].to_numpy()) > 0:
                record[name] = 100 * float(
                    np.polyfit(sub["ghi"].to_numpy(), sub["power"].to_numpy(), 1)[0]
                )
            else:
                record[name] = None
            record[f"{name}_n"] = sub.height
        rows.append(record)
    return pl.DataFrame(
        rows, schema_overrides={"below_500": pl.Float64, "above_600": pl.Float64}
    ).sort("time_series_id", "season")


def level_correlations(*, levels: pl.DataFrame, series_ids: list[int], band: str) -> pl.DataFrame:
    """Pairwise correlation of monthly band levels after removing each series' mean seasonal cycle.

    Uses the months where all the listed series have data and outside the fixed lockdown window.
    """
    wide = (
        levels.filter(
            pl.col("time_series_id").is_in(series_ids), ~is_lockdown(month=pl.col("month"))
        )
        .select("time_series_id", "month", band)
        .drop_nulls()
        .pivot(on="time_series_id", index="month", values=band)
        .drop_nulls()
        .sort("month")
    )
    value_columns = [c for c in wide.columns if c != "month"]
    anomalies = wide.with_columns(
        [
            (pl.col(c) - pl.col(c).mean().over(pl.col("month").dt.month())).alias(c)
            for c in value_columns
        ]
    )
    matrix = np.corrcoef(anomalies.select(value_columns).to_numpy().T)
    return pl.DataFrame(matrix, schema=value_columns).with_columns(
        series=pl.Series(value_columns), months=pl.lit(wide.height)
    )


def scarce_census(*, panel: pl.DataFrame) -> pl.DataFrame:
    """Per series and window: cold days, bank holidays, and Christmas-run days that hold clean data.

    The windows are the scoring years 2023-24, 2024-25, 2025-26 (July to June) and the cold block
    (November 2022 to February 2023). A cold day has daily-mean ERA5 temperature below 0 degC. The
    Christmas run is 24 December to 1 January. Bank holidays are England and Wales.
    """
    import holidays

    england = holidays.UnitedKingdom(subdiv="ENG", years=range(2019, 2027))
    bank_days = pl.Series(sorted(england.keys())).cast(pl.Date)
    days = daily_table(panel=panel).filter(pl.col("day_n") >= 44)
    days = days.with_columns(
        cold=pl.col("temperature_c") < 0.0,
        bank_holiday=pl.col("local_date").is_in(bank_days.implode()),
        christmas=((pl.col("local_date").dt.month() == 12) & (pl.col("local_date").dt.day() >= 24))
        | ((pl.col("local_date").dt.month() == 1) & (pl.col("local_date").dt.day() == 1)),
        season=season_year(date=pl.col("local_date")),
    )
    windows = {
        "2023-24": (pl.col("season") == 2023),
        "2024-25": (pl.col("season") == 2024),
        "2025-26": (pl.col("season") == 2025),
        "cold block Nov 2022-Feb 2023": pl.col("local_date").is_between(
            pl.date(2022, 11, 1), pl.date(2023, 2, 28)
        ),
    }
    parts = []
    for name, condition in windows.items():
        parts.append(
            days.filter(condition)
            .group_by("time_series_id")
            .agg(
                cold_days=pl.col("cold").sum(),
                bank_holidays=pl.col("bank_holiday").sum(),
                christmas_days=pl.col("christmas").sum(),
                days=pl.len(),
            )
            .with_columns(window=pl.lit(name))
        )
    return pl.concat(parts).sort("time_series_id", "window")


def leading_cut(*, calibration: pl.DataFrame, first_clean: pl.DataFrame) -> pl.DataFrame:
    """First usable time per series: after the leading run of months that differ by over 15 points.

    The run must start at the series' first month and stay unbroken, so the rule only acts at the
    start of a series. Months with no comparison month do not count as differing.
    """
    rows = []
    for series_id, first in first_clean.iter_rows():
        own = calibration.filter(pl.col("time_series_id") == series_id).sort("month_index")
        run = 0
        for differs in own["differs"].to_list():
            if not differs:
                break
            run += 1
        start = first.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        usable = _add_months(start, run) if run else first
        rows.append(
            {
                "time_series_id": series_id,
                "first_clean": first,
                "cut_months": run,
                "first_usable": usable,
            }
        )
    return pl.DataFrame(rows).sort("time_series_id")


def lockdown_departures(
    *, levels: pl.DataFrame, series_ids: list[int], first_usable: dict
) -> pl.DataFrame:
    """Weekday 09:00-16:59 level minus a no-lockdown fit, per site, for 2020-01 to 2021-12.

    Per site: least squares of the monthly level on 11 calendar-month dummies and a linear trend,
    fitted on the months after the site's first usable time and outside the fixed lockdown window.
    """
    rows = []
    for series_id in series_ids:
        own = (
            levels.filter(pl.col("time_series_id") == series_id)
            .select("month", "weekday_day", "weekday_day_n")
            .drop_nulls()
            .filter(pl.col("month") >= first_usable[series_id], pl.col("weekday_day_n") >= 8 * 16)
            .sort("month")
        )
        months = own["month"].dt.month().to_numpy()
        t = (own["month"].dt.year().to_numpy() * 12 + months).astype(float)
        design = np.column_stack(
            [
                t - t.mean(),
                *[(months == m).astype(float) for m in range(2, 13)],
                np.ones(own.height),
            ]
        )
        outside = ~own.select(is_lockdown(month=pl.col("month")).alias("x"))["x"].to_numpy()
        coefficients, *_ = np.linalg.lstsq(
            design[outside], own["weekday_day"].to_numpy()[outside], rcond=None
        )
        departure = own["weekday_day"].to_numpy() - design @ coefficients
        for month, value in zip(own["month"].to_list(), departure, strict=True):
            if (2020, 1) <= (month.year, month.month) <= (2021, 12):
                rows.append(
                    {"time_series_id": series_id, "month": month, "departure": float(value)}
                )
    return pl.DataFrame(rows).sort("time_series_id", "month")


def heating_slope_drift(
    *, daily: pl.DataFrame, series_id: int, first_usable: datetime, n_resamples: int = 2000
) -> dict:
    """Difference of the last and first winters' slopes (winter 2020-21 left out).

    Each winter's days (night mean against daily-mean temperature below 12 degC) are resampled by
    whole calendar months, and the difference of the two resampled slopes is the draw.
    """
    winter = (
        daily.filter(
            pl.col("time_series_id") == series_id, pl.col("local_date") >= first_usable.date()
        )
        .with_columns(winter=winter_year(date=pl.col("local_date")))
        .filter(
            pl.col("winter").is_not_null(),
            pl.col("winter") != 2020,
            pl.col("temperature_c") < 12.0,
            pl.col("night_n") >= 8,
            pl.col("night_mean").is_not_null(),
        )
    )
    counts = winter.group_by("winter").agg(pl.len()).filter(pl.col("len") >= 60).sort("winter")
    if counts.height < 2:
        return {"series_id": series_id, "first": None, "last": None}
    winters = counts["winter"].to_list()
    first, last = min(winters), max(winters)
    rng = np.random.default_rng(0)
    parts = {}
    for w in (first, last):
        part = winter.filter(pl.col("winter") == w)
        months = part["month"].dt.month().to_numpy()
        parts[w] = (
            part["temperature_c"].to_numpy().astype(float),
            part["night_mean"].to_numpy().astype(float),
            months,
        )

    def one(w: int, pick: bool) -> float:
        x, y, months = parts[w]
        if pick:
            unique = np.unique(months)
            chosen = rng.choice(unique, unique.size)
            index = np.concatenate([np.flatnonzero(months == m) for m in chosen])
            x, y = x[index], y[index]
        return float(np.polyfit(x, y, 1)[0]) if np.ptp(x) > 0 else float("nan")

    estimate = one(last, False) - one(first, False)
    draws = np.array([one(last, True) - one(first, True) for _ in range(n_resamples)])
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return {
        "series_id": series_id,
        "first": int(first),
        "last": int(last),
        "first_slope": one(first, False),
        "last_slope": one(last, False),
        "difference": estimate,
        "low": float(low),
        "high": float(high),
    }
