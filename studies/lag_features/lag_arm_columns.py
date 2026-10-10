"""Build the columns of the calibration and fingerprint arms of the lagged-power-features study.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1138>;
`build_lag_frame.py` calls these functions and owns the frame, the row filter and the report.

**Every window is anchored at the issue day and reads nothing after the issue time.** A forecast for
lead-day `N` is issued at 09:00 UTC on the run's own day (the run's 00 UTC init time at lead-day
0), so the latest whole day it has seen is `N + 1` days before the target day. Three kinds of
column read the record, and each has its own guard in `window_anchor_lines`:

- **Same-clock-hour windows** (`studies.baselines.same_clock_hour_window`) start `first_days_back`
  whole days back, counted from that latest whole day, and never read the issue day.
- **Daily features** (`daily_features`) are looked up at the latest whole day, which holds only
  hours that ended at or before midnight at the start of the issue day plus `N` days.
- **The issue morning** (`issue_morning`) reads the hours of the issue day that end at or before
  09:00 UTC, which is the issue time itself.

**The CAMS satellite irradiance arm (PC) is study-only unless it wins.** The live service does not
ingest CAMS. The study assumes CAMS publishes two days late, so PC's windows start three whole days
back (the CAMS documentation says up to 2 days; see the README).
"""

from typing import Final, NamedTuple

import numpy as np
import polars as pl
from studies.baselines import (
    MIN_OBSERVED_CLEAR_SKY_SHARE,
    hourly_clear_sky,
    hourly_grid,
    issue_time,
    same_clock_hour_window,
)

MORNING_CUTOFF_HOURS: Final[int] = 9
"""The issue morning ends at 09:00 UTC on the issue day, the issue time."""

TRANSFER_MIN_IRRADIANCE_W_M2: Final[float] = 50.0
"""TF and PC read only hours whose irradiance exceeds this, where a ratio is well conditioned."""

TRANSFER_DAYS: Final[int] = 30
TRANSFER_MIN_DAYS: Final[int] = 15
"""TF's window, and the fewest of its days that must hold the hour."""

SATELLITE_LAG_DAYS: Final[int] = 3
"""CAMS is assumed published two days late, so the latest whole day it has seen at the issue time
ends two days before the issue day, and the satellite windows start three whole days back."""

SATELLITE_POWER_DAYS: Final[int] = 7
SATELLITE_POWER_MIN_DAYS: Final[int] = 4
"""PC's power-to-satellite window, and the fewest of its days that must hold the hour."""

ANALOGUE_DAYS: Final[int] = 30
ANALOGUE_COUNT: Final[int] = 5
ANALOGUE_MIN: Final[int] = 3
"""AN looks at the last 30 days, keeps the 5 whose forecast clear-sky index is closest, and needs
at least 3 hours that can be rescaled."""

CEILING_QUANTILE: Final[float] = 0.995
CEILING_MIN_HOURS: Final[int] = 1000
CEILING_MAX_DAYS: Final[int] = 60
CEILING_MAX_MIN_DAYS: Final[int] = 30
"""CK's expanding quantile of hourly power and its fewest hours; the maximum's window and fewest
days."""

CLEAR_FORECAST_INDEX: Final[float] = 0.8
CLEAR_MIN_CLEAR_SKY_W_M2: Final[float] = 300.0
NEAR_CEILING_SHARE: Final[float] = 0.95
NEAR_WINDOW_DAYS: Final[int] = 30
NEAR_MIN_DAYS: Final[int] = 10
"""A clear hour has a forecast clear-sky index of at least 0.8 and a clear-sky irradiance of at
least 300 W m⁻². It is near the ceiling at 95% of the ceiling. The share is taken over 30 days."""

FINGERPRINT_DAYS: Final[int] = 30
FINGERPRINT_MIN_DAYS: Final[int] = 15
FINGERPRINT_MIN_HOURS: Final[int] = 6
SHOULDER_CLEAR_SKY_SHARE: Final[float] = 0.3
"""DT's window, the fewest of its days that must be valid, the fewest hours a day needs, and the
share of a day's peak clear-sky irradiance below which an hour is on the shoulder."""

RELATIVE_ENERGY_WINDOWS: Final[tuple[tuple[str, int, int], ...]] = (
    ("rp_7d", 7, 5),
    ("rp_1d", 1, 1),
)
"""RP's columns: the number of whole days and the fewest that must hold a valid day."""

CONTEXT_DAYS: Final[int] = 7
"""CTX7 reads the lag and the lag-hour forecast irradiance for days 1 to 7."""

DAILY_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "day_energy",
    "rp_7d",
    "rp_1d",
    "ck_expanding_p995",
    "ck_max_60d",
    "dt_centroid_shift_30d",
    "dt_shoulder_share_30d",
)
"""The columns of `daily_features`, looked up at the latest whole day."""


class LagInputs(NamedTuple):
    """The lead-independent inputs every arm's columns read."""

    hourly: pl.DataFrame
    clear_sky: pl.DataFrame
    cams: pl.DataFrame
    daily: pl.DataFrame


def clear_sky_table(*, hourly: pl.DataFrame, sites: pl.DataFrame) -> pl.DataFrame:
    """Return the Haurwitz clear-sky irradiance for every hour of the record, at every plant.

    Args:
        hourly: The lag source, with `site` and `time`.
        sites: The site list with `site`, `latitude` and `longitude`, read and never written.

    Returns:
        `site`, `time` and `clear_sky_w_m2`.
    """
    span = hourly.select(
        first=pl.col("time").min() - pl.duration(days=1),
        last=pl.col("time").max() + pl.duration(days=1),
    ).row(0, named=True)
    return hourly_clear_sky(sites=sites, first=span["first"], last=span["last"])


def _date_grid(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return every date from each plant's first date to the last date of any plant.

    A plant's grid runs to the common last date, so a feature looked up at a date after the plant's
    own last reading is null by its window's rule, whether or not later data exists.

    Args:
        frame: Rows with `site` and `date`.

    Returns:
        `site` and `date`, one row per plant and day.
    """
    last = frame["date"].max()
    return (
        frame.group_by("site")
        .agg(first=pl.col("date").min())
        .select("site", date=pl.date_ranges(pl.col("first"), pl.lit(last), interval="1d"))
        .explode("date")
    )


def _expanding_ceiling(*, hourly: pl.DataFrame, dates: pl.DataFrame) -> pl.DataFrame:
    """Return each plant's expanding 99.5th percentile of hourly power at the end of each date.

    Args:
        hourly: The lag source.
        dates: `_date_grid`'s result.

    Returns:
        `site`, `date` and `ck_expanding_p995`, null until the plant has `CEILING_MIN_HOURS` hours.
    """
    parts = []
    for (site,), power in (
        hourly.drop_nulls("power_mw").sort("time").group_by("site", maintain_order=True)
    ):
        values = power["power_mw"].to_numpy()
        own_dates = power["time"].dt.offset_by("-30m").dt.date().to_numpy()
        grid = dates.filter(pl.col("site") == site).sort("date")["date"].to_numpy()
        ends = np.searchsorted(own_dates, grid, side="right")
        ceiling = np.full(len(grid), np.nan)
        for index, end in enumerate(ends):
            if end >= CEILING_MIN_HOURS:
                ceiling[index] = np.quantile(values[:end], CEILING_QUANTILE)
        parts.append(
            pl.DataFrame({"date": grid, "ck_expanding_p995": ceiling}).with_columns(
                site=pl.lit(site), ck_expanding_p995=pl.col("ck_expanding_p995").fill_nan(None)
            )
        )
    return pl.concat(parts)


def daily_features(
    *, hourly: pl.DataFrame, clear_sky: pl.DataFrame, capacity: pl.DataFrame
) -> pl.DataFrame:
    """Summarise each plant's days into the lead-independent features of the daily arms.

    A day is valid only if its observed hours hold at least `MIN_OBSERVED_CLEAR_SKY_SHARE` of its
    clear-sky energy. The features at date `d` read hours that ended by midnight at the end of `d`.

    Args:
        hourly: The lag source, with `site`, `time` and `power_mw`.
        clear_sky: `clear_sky_table`'s result.
        capacity: `site` and `effective_capacity_mw`.

    Returns:
        One row per `(site, date)` of the complete date grid, with `DAILY_FEATURE_COLUMNS`.
    """
    grid = (
        hourly_grid(hourly=hourly)
        .join(clear_sky, on=["site", "time"], how="inner")
        .with_columns(
            date=(pl.col("time") - pl.duration(minutes=30)).dt.date(),
            midpoint=(pl.col("time") - pl.duration(minutes=30)).dt.hour().cast(pl.Float64) + 0.5,
        )
        .with_columns(day_cs_max=pl.col("clear_sky_w_m2").max().over("site", "date"))
    )
    present = pl.col("power_mw").is_not_null()
    daily = (
        grid.group_by("site", "date")
        .agg(
            peak=pl.col("power_mw").max(),
            energy=pl.col("power_mw").sum(),
            n_hours=present.sum(),
            observed_clear_sky=pl.col("clear_sky_w_m2").filter(present).sum(),
            clear_sky=pl.col("clear_sky_w_m2").sum(),
            moment_power=(pl.col("midpoint") * pl.col("power_mw")).sum(),
            moment_clear_sky=(pl.col("midpoint") * pl.col("clear_sky_w_m2")).filter(present).sum(),
            shoulder_energy=pl.col("power_mw")
            .filter(pl.col("clear_sky_w_m2") < SHOULDER_CLEAR_SKY_SHARE * pl.col("day_cs_max"))
            .sum(),
            shoulder_hours=pl.col("power_mw")
            .filter(
                (pl.col("clear_sky_w_m2") < SHOULDER_CLEAR_SKY_SHARE * pl.col("day_cs_max"))
                & present
            )
            .len(),
        )
        .with_columns(
            valid=(pl.col("clear_sky") > 0)
            & (pl.col("observed_clear_sky") >= MIN_OBSERVED_CLEAR_SKY_SHARE * pl.col("clear_sky"))
        )
        .with_columns(
            day_energy=pl.when("valid").then(pl.col("energy")),
            day_peak=pl.when("valid").then(pl.col("peak")),
            shape_valid=pl.col("valid")
            & (pl.col("n_hours") >= FINGERPRINT_MIN_HOURS)
            & (pl.col("energy") > 0),
        )
        .with_columns(
            centroid_shift=pl.when("shape_valid").then(
                pl.col("moment_power") / pl.col("energy")
                - pl.col("moment_clear_sky") / pl.col("observed_clear_sky")
            ),
            shoulder_share=pl.when(pl.col("shape_valid") & (pl.col("shoulder_hours") > 0)).then(
                pl.col("shoulder_energy") / pl.col("energy")
            ),
        )
        .join(capacity, on="site")
        .with_columns(relative_energy=pl.col("day_energy") / pl.col("effective_capacity_mw"))
    )
    dates = _date_grid(frame=daily)
    full = dates.join(daily, on=["site", "date"], how="left").sort("site", "date")
    rolled = full.with_columns(
        *(
            pl.col("relative_energy")
            .rolling_mean(window_size=window, min_samples=least)
            .over("site")
            .alias(name)
            for name, window, least in RELATIVE_ENERGY_WINDOWS
        ),
        ck_max_60d=pl.col("day_peak")
        .rolling_max(window_size=CEILING_MAX_DAYS, min_samples=CEILING_MAX_MIN_DAYS)
        .over("site"),
        dt_centroid_shift_30d=pl.col("centroid_shift")
        .rolling_mean(window_size=FINGERPRINT_DAYS, min_samples=FINGERPRINT_MIN_DAYS)
        .over("site"),
        dt_shoulder_share_30d=pl.col("shoulder_share")
        .rolling_mean(window_size=FINGERPRINT_DAYS, min_samples=FINGERPRINT_MIN_DAYS)
        .over("site"),
    )
    others = rolled.group_by("date").agg(
        **{f"{name}_total": pl.col(name).sum() for name, _, _ in RELATIVE_ENERGY_WINDOWS},
        **{
            f"{name}_count": pl.col(name).is_not_null().sum()
            for name, _, _ in RELATIVE_ENERGY_WINDOWS
        },
    )
    relative = rolled.join(others, on="date", how="left")
    for name, _, _ in RELATIVE_ENERGY_WINDOWS:
        others_count = pl.col(f"{name}_count") - pl.col(name).is_not_null().cast(pl.UInt32)
        others_mean = (pl.col(f"{name}_total") - pl.col(name).fill_null(0.0)) / others_count
        relative = relative.with_columns(
            pl.when((others_count > 0) & (others_mean > 0))
            .then(pl.col(name) / others_mean)
            .alias(name)
        )
    ceilings = _expanding_ceiling(hourly=hourly, dates=dates)
    return relative.join(ceilings, on=["site", "date"], how="left").select(
        "site", "date", *DAILY_FEATURE_COLUMNS
    )


def latest_whole_day(*, lead_day: int) -> pl.Expr:
    """Return the date of the latest whole day a forecast for a target hour has seen.

    Args:
        lead_day: The forecast's lead-day.

    Returns:
        The date `lead_day + 1` days before the target hour's own date, where a solar hour's own
        date is that of its midpoint.
    """
    return (pl.col("time") - pl.duration(minutes=30)).dt.date() - pl.duration(days=lead_day + 1)


def lookup_daily(
    *, rows: pl.DataFrame, daily: pl.DataFrame, lead_day: int, columns: list[str]
) -> pl.DataFrame:
    """Look the daily features up at each row's latest whole day.

    Args:
        rows: Rows with `site` and `time`.
        daily: `daily_features`' result, or any frame keyed `(site, date)`.
        lead_day: The forecast's lead-day.
        columns: The columns of `daily` to return.

    Returns:
        One row per row of `rows`, in its order, with `columns`.
    """
    keyed = rows.select("site", asof_date=latest_whole_day(lead_day=lead_day))
    return keyed.join(
        daily.select("site", pl.col("date").alias("asof_date"), *columns),
        on=["site", "asof_date"],
        how="left",
        maintain_order="left",
    ).select(columns)


def _window(
    *,
    rows: pl.DataFrame,
    table: pl.DataFrame,
    lead_day: int,
    first: int,
    last: int,
    min_count: int,
) -> pl.Series:
    """Return the mean of a table's `power_mw` over a same-clock-hour window.

    Args:
        rows: Rows with `site` and `time`.
        table: A frame with `site`, `time` and `power_mw` (any quantity, named so).
        lead_day: The forecast's lead-day.
        first: The nearest day of the window, counted from the latest whole day.
        last: The furthest day.
        min_count: The fewest days that must hold the hour.

    Returns:
        The mean, null where fewer than `min_count` days hold the hour.
    """
    return same_clock_hour_window(
        keys=rows,
        hourly=table,
        day=lead_day,
        first_days_back=first,
        last_days_back=last,
        statistic="mean",
        min_count=min_count,
    )


def issue_morning(
    *, rows: pl.DataFrame, hourly: pl.DataFrame, weather_lead0: pl.DataFrame, lead_day: int
) -> tuple[pl.DataFrame, pl.Series]:
    """Return the issue morning's observed energy, its ratio to forecast irradiance, and its hours.

    The morning is the hours of the issue day that end at or before 09:00 UTC. Their forecast
    irradiance is the same run's lead-day 0 value, which the 09:00 issue has.

    Args:
        rows: Rows with `site` and `time`.
        hourly: The lag source.
        weather_lead0: `nwp_ghi` at lead-day 0, with `site` and `time`.
        lead_day: The forecast's lead-day; at lead-day 0 the issue is at 00 UTC and the columns are
            null.

    Returns:
        The three columns `im_energy`, `im_ratio` and `im_hours`, and each row's latest morning hour
        read (null where none), for the anchor assertion.
    """
    empty = pl.DataFrame(
        {
            "im_energy": pl.Series([None] * rows.height, dtype=pl.Float64),
            "im_ratio": pl.Series([None] * rows.height, dtype=pl.Float64),
            "im_hours": pl.Series([None] * rows.height, dtype=pl.Float64),
        }
    )
    never = pl.Series([None] * rows.height, dtype=rows["time"].dtype)
    if lead_day < 1:
        return empty, never
    morning = (
        hourly.join(weather_lead0, on=["site", "time"])
        .drop_nulls(["power_mw", "nwp_ghi"])
        .with_columns(date=(pl.col("time") - pl.duration(minutes=30)).dt.date())
        .filter(
            pl.col("time")
            <= pl.col("date").cast(pl.Datetime("us", "UTC"))
            + pl.duration(hours=MORNING_CUTOFF_HOURS)
        )
        .group_by("site", "date")
        .agg(
            im_energy=pl.col("power_mw").sum(),
            irradiance=pl.col("nwp_ghi").sum(),
            im_hours=pl.len().cast(pl.Float64),
            latest=pl.col("time").max(),
        )
        .with_columns(
            im_ratio=pl.when(pl.col("irradiance") > 0).then(
                pl.col("im_energy") / pl.col("irradiance")
            )
        )
    )
    keyed = rows.select(
        "site",
        date=(pl.col("time") - pl.duration(minutes=30)).dt.date() - pl.duration(days=lead_day),
    )
    joined = keyed.join(morning, on=["site", "date"], how="left", maintain_order="left")
    return joined.select("im_energy", "im_ratio", "im_hours"), joined["latest"]


def transfer_function(
    *, rows: pl.DataFrame, hourly: pl.DataFrame, weather: pl.DataFrame, lead_day: int
) -> pl.DataFrame:
    """Return TF: the 30-day power-to-irradiance ratio, and that ratio times the forecast.

    Only hours whose forecast irradiance exceeds `TRANSFER_MIN_IRRADIANCE_W_M2` count, and at least
    `TRANSFER_MIN_DAYS` of the 30 days must hold the hour. The ratio is the window's mean power over
    its mean irradiance on the same days.

    Args:
        rows: Rows with `site`, `time` and `nwp_ghi`.
        hourly: The lag source.
        weather: The lead-day's `nwp_ghi`, with `site` and `time`, for every hour.
        lead_day: The forecast's lead-day.

    Returns:
        `tf_ratio` and `tf_scaled_forecast`.
    """
    both = (
        hourly.join(weather.select("site", "time", "nwp_ghi"), on=["site", "time"])
        .drop_nulls(["power_mw", "nwp_ghi"])
        .filter(pl.col("nwp_ghi") > TRANSFER_MIN_IRRADIANCE_W_M2)
    )
    numerator = _window(
        rows=rows,
        table=both.select("site", "time", "power_mw"),
        lead_day=lead_day,
        first=1,
        last=TRANSFER_DAYS,
        min_count=TRANSFER_MIN_DAYS,
    )
    denominator = _window(
        rows=rows,
        table=both.select("site", "time", power_mw=pl.col("nwp_ghi")),
        lead_day=lead_day,
        first=1,
        last=TRANSFER_DAYS,
        min_count=TRANSFER_MIN_DAYS,
    )
    ratio = numerator / denominator
    return pl.DataFrame({"tf_ratio": ratio}).with_columns(
        tf_scaled_forecast=pl.col("tf_ratio") * rows["nwp_ghi"]
    )


def satellite_ratios(
    *,
    rows: pl.DataFrame,
    hourly: pl.DataFrame,
    weather: pl.DataFrame,
    cams: pl.DataFrame,
    lead_day: int,
) -> pl.DataFrame:
    """Return PC: power to satellite irradiance over 7 days, and satellite to forecast over 30.

    The satellite irradiance is assumed published two days late, so both windows start
    `SATELLITE_LAG_DAYS` (3) whole days back: days 3 to 9 and days 3 to 32.

    Args:
        rows: Rows with `site` and `time`.
        hourly: The lag source.
        weather: The lead-day's `nwp_ghi`.
        cams: `ghi_cams`, with `site` and `time`.
        lead_day: The forecast's lead-day.

    Returns:
        `pc_power_to_cams_7d` and `pc_cams_to_forecast_30d`.
    """
    power_cams = (
        hourly.join(cams, on=["site", "time"])
        .drop_nulls(["power_mw", "ghi_cams"])
        .filter(pl.col("ghi_cams") > TRANSFER_MIN_IRRADIANCE_W_M2)
    )
    first = SATELLITE_LAG_DAYS
    power = _window(
        rows=rows,
        table=power_cams.select("site", "time", "power_mw"),
        lead_day=lead_day,
        first=first,
        last=first + SATELLITE_POWER_DAYS - 1,
        min_count=SATELLITE_POWER_MIN_DAYS,
    )
    satellite = _window(
        rows=rows,
        table=power_cams.select("site", "time", power_mw=pl.col("ghi_cams")),
        lead_day=lead_day,
        first=first,
        last=first + SATELLITE_POWER_DAYS - 1,
        min_count=SATELLITE_POWER_MIN_DAYS,
    )
    cams_forecast = (
        cams.join(weather.select("site", "time", "nwp_ghi"), on=["site", "time"])
        .drop_nulls(["ghi_cams", "nwp_ghi"])
        .filter(
            (pl.col("ghi_cams") > TRANSFER_MIN_IRRADIANCE_W_M2)
            & (pl.col("nwp_ghi") > TRANSFER_MIN_IRRADIANCE_W_M2)
        )
    )
    last = first + TRANSFER_DAYS - 1
    cams_mean = _window(
        rows=rows,
        table=cams_forecast.select("site", "time", power_mw=pl.col("ghi_cams")),
        lead_day=lead_day,
        first=first,
        last=last,
        min_count=TRANSFER_MIN_DAYS,
    )
    forecast_mean = _window(
        rows=rows,
        table=cams_forecast.select("site", "time", power_mw=pl.col("nwp_ghi")),
        lead_day=lead_day,
        first=first,
        last=last,
        min_count=TRANSFER_MIN_DAYS,
    )
    return pl.DataFrame(
        {
            "pc_power_to_cams_7d": power / satellite,
            "pc_cams_to_forecast_30d": cams_mean / forecast_mean,
        }
    )


def analogue_ensemble(
    *,
    rows: pl.DataFrame,
    hourly: pl.DataFrame,
    weather: pl.DataFrame,
    clear_sky: pl.DataFrame,
    lead_day: int,
) -> pl.DataFrame:
    """Return AN: the mean, forecast clear-sky index and spread of the closest analogue days.

    For each row, the candidates are the same clock hour on each of the last 30 whole days. A
    candidate's forecast clear-sky index is its lead-day forecast irradiance over its clear-sky
    irradiance. The 5 candidates whose index is closest to the target's are kept, and each one's
    observed power is rescaled by the target's clear-sky irradiance over the candidate's.

    Args:
        rows: Rows with `site`, `time` and `nwp_ghi`.
        hourly: The lag source.
        weather: The lead-day's `nwp_ghi`, with `site` and `time`.
        clear_sky: `clear_sky_table`'s result.
        lead_day: The forecast's lead-day.

    Returns:
        `an_mean`, `an_csi` and `an_spread`, null with fewer than `ANALOGUE_MIN` usable candidates.
    """
    hours = hourly.join(weather.select("site", "time", "nwp_ghi"), on=["site", "time"]).join(
        clear_sky, on=["site", "time"]
    )
    target = rows.select("site", "time", "nwp_ghi").join(
        clear_sky.rename({"clear_sky_w_m2": "target_cs"}),
        on=["site", "time"],
        how="left",
        maintain_order="left",
    )
    target_index = (target["nwp_ghi"] / target["target_cs"]).to_numpy().astype(np.float64)
    target_cs = target["target_cs"].to_numpy().astype(np.float64)
    power = np.full((rows.height, ANALOGUE_DAYS), np.nan)
    index = np.full_like(power, np.nan)
    scaled = np.full_like(power, np.nan)
    for day in range(1, ANALOGUE_DAYS + 1):
        joined = rows.select(
            "site", lag_time=pl.col("time") - pl.duration(days=lead_day + day)
        ).join(
            hours.rename({"time": "lag_time"}),
            on=["site", "lag_time"],
            how="left",
            maintain_order="left",
        )
        p = joined["power_mw"].to_numpy().astype(np.float64)
        cs = joined["clear_sky_w_m2"].to_numpy().astype(np.float64)
        ghi = joined["nwp_ghi"].to_numpy().astype(np.float64)
        usable = (cs > 0) & ~np.isnan(p) & ~np.isnan(ghi)
        power[:, day - 1] = np.where(usable, p, np.nan)
        index[:, day - 1] = np.where(usable, ghi / np.where(cs > 0, cs, 1.0), np.nan)
        scaled[:, day - 1] = np.where(usable, p * target_cs / np.where(cs > 0, cs, 1.0), np.nan)
    distance = np.abs(index - target_index[:, None])
    distance = np.where(np.isnan(distance), np.inf, distance)
    nearest = np.argsort(distance, axis=1)[:, :ANALOGUE_COUNT]
    chosen_scaled = np.take_along_axis(scaled, nearest, axis=1)
    chosen_index = np.take_along_axis(index, nearest, axis=1)
    enough = (~np.isnan(chosen_scaled)).sum(axis=1) >= ANALOGUE_MIN
    with np.errstate(all="ignore"):
        mean = np.nanmean(chosen_scaled, axis=1)
        csi = np.nanmean(chosen_index, axis=1)
        spread = np.nanstd(chosen_scaled, axis=1)
    return pl.DataFrame(
        {
            "an_mean": np.where(enough, mean, np.nan),
            "an_csi": np.where(enough, csi, np.nan),
            "an_spread": np.where(enough, spread, np.nan),
        }
    ).with_columns(pl.all().fill_nan(None))


def clipping_share(
    *,
    rows: pl.DataFrame,
    hourly: pl.DataFrame,
    weather: pl.DataFrame,
    clear_sky: pl.DataFrame,
    daily: pl.DataFrame,
    lead_day: int,
) -> pl.Series:
    """Return CK's share of clear hours in the last 30 days that sit near the clipping ceiling.

    A clear hour has a forecast clear-sky index of at least `CLEAR_FORECAST_INDEX`. It is near the
    ceiling at `NEAR_CEILING_SHARE` of the plant's expanding ceiling at the end of its own day.

    Args:
        rows: Rows with `site` and `time`.
        hourly: The lag source.
        weather: The lead-day's `nwp_ghi`.
        clear_sky: `clear_sky_table`'s result.
        daily: `daily_features`' result, for the ceiling.
        lead_day: The forecast's lead-day.

    Returns:
        `ck_clear_near_share`, null with fewer than `NEAR_MIN_DAYS` days that hold a clear hour.
    """
    hours = (
        hourly.join(weather.select("site", "time", "nwp_ghi"), on=["site", "time"])
        .join(clear_sky, on=["site", "time"])
        .with_columns(date=(pl.col("time") - pl.duration(minutes=30)).dt.date())
        .join(daily.select("site", "date", "ck_expanding_p995"), on=["site", "date"])
        .drop_nulls(["power_mw", "nwp_ghi", "ck_expanding_p995"])
        .filter(
            (pl.col("clear_sky_w_m2") >= CLEAR_MIN_CLEAR_SKY_W_M2)
            & (pl.col("nwp_ghi") / pl.col("clear_sky_w_m2") >= CLEAR_FORECAST_INDEX)
        )
    )
    counts = hours.group_by("site", "date").agg(
        clear=pl.len().cast(pl.Float64),
        near=(pl.col("power_mw") >= NEAR_CEILING_SHARE * pl.col("ck_expanding_p995"))
        .sum()
        .cast(pl.Float64),
    )
    full = (
        daily.select("site", "date")
        .join(counts, on=["site", "date"], how="left")
        .sort("site", "date")
    )
    rolled = full.with_columns(
        near_sum=pl.col("near")
        .rolling_sum(window_size=NEAR_WINDOW_DAYS, min_samples=NEAR_MIN_DAYS)
        .over("site"),
        clear_sum=pl.col("clear")
        .rolling_sum(window_size=NEAR_WINDOW_DAYS, min_samples=NEAR_MIN_DAYS)
        .over("site"),
    ).with_columns(
        ck_clear_near_share=pl.when(pl.col("clear_sum") > 0).then(
            pl.col("near_sum") / pl.col("clear_sum")
        )
    )
    return lookup_daily(
        rows=rows, daily=rolled, lead_day=lead_day, columns=["ck_clear_near_share"]
    )["ck_clear_near_share"]


def lag_context(
    *, rows: pl.DataFrame, hourly: pl.DataFrame, weather: pl.DataFrame, lead_day: int
) -> pl.DataFrame:
    """Return CTX7: the lag and the lag-hour forecast irradiance for days 1 to 7.

    Args:
        rows: Rows with `site` and `time`.
        hourly: The lag source.
        weather: The lead-day's `nwp_ghi`.
        lead_day: The forecast's lead-day.

    Returns:
        `lag_d1` to `lag_d7` and `lag_ghi_d1` to `lag_ghi_d7`.
    """
    columns: dict[str, pl.Series] = {}
    for day in range(1, CONTEXT_DAYS + 1):
        columns[f"lag_d{day}"] = _window(
            rows=rows, table=hourly, lead_day=lead_day, first=day, last=day, min_count=1
        )
        columns[f"lag_ghi_d{day}"] = rows.select(
            "site", lag_time=pl.col("time") - pl.duration(days=lead_day + day)
        ).join(
            weather.select("site", pl.col("time").alias("lag_time"), "nwp_ghi"),
            on=["site", "lag_time"],
            how="left",
            maintain_order="left",
        )["nwp_ghi"]
    return pl.DataFrame(columns)


def window_anchor_lines(
    *,
    rows: pl.DataFrame,
    lead_day: int,
    windows: dict[str, int],
    daily_columns: list[str],
    morning_latest: pl.Series | None,
) -> list[str]:
    """Assert that every window reads nothing after the issue time, and report each margin.

    Args:
        rows: Rows with `time`.
        lead_day: The forecast's lead-day.
        windows: Each same-clock-hour window's name and its nearest day (`first_days_back`).
        daily_columns: The daily-feature columns the frame holds, looked up at the latest whole day.
        morning_latest: Each row's latest issue-morning hour read, or `None` if IM is absent.

    Returns:
        Report lines, one per window, giving the smallest margin in hours between the latest hour
        read and the issue time.

    Raises:
        ValueError: If any window reads an hour that ends after the issue time.
    """
    margins = rows.select(
        issued=issue_time(
            day_start=(pl.col("time") - pl.duration(minutes=30)).dt.truncate("1d"), day=lead_day
        ),
        time=pl.col("time"),
        asof_end=latest_whole_day(lead_day=lead_day).cast(pl.Datetime("us", "UTC"))
        + pl.duration(days=1),
    )
    lines = [
        "| Column or group | Reads from | Smallest margin before the issue time (hours) |",
        "|---|---|---|",
    ]
    for name, first in windows.items():
        margin = margins.select(
            hours=(
                pl.col("issued") - (pl.col("time") - pl.duration(days=lead_day + first))
            ).dt.total_hours()
        )["hours"].min()
        _require(name=name, lead_day=lead_day, smallest=margin)
        lines.append(f"| `{name}` | {first} whole days back or earlier | {margin} |")
    if daily_columns:
        margin = margins.select(hours=(pl.col("issued") - pl.col("asof_end")).dt.total_hours())[
            "hours"
        ].min()
        _require(name="daily features", lead_day=lead_day, smallest=margin)
        group = ", ".join(f"`{c}`" for c in daily_columns)
        lines.append(f"| {group} | the latest whole day | {margin} |")
    if morning_latest is not None:
        gap = (
            margins.with_columns(latest=morning_latest)
            .select(hours=(pl.col("issued") - pl.col("latest")).dt.total_hours())["hours"]
            .drop_nulls()
        )
        if not gap.is_empty():
            _require(name="IM", lead_day=lead_day, smallest=gap.min())
            lines.append(
                "| `im_energy`, `im_ratio`, `im_hours` | the issue day's hours to 09:00 UTC | "
                f"{gap.min()} |"
            )
    return [
        (
            "Every window below is checked on every row to read nothing after the issue time "
            f"(lead-day {lead_day}); a negative margin raises."
        ),
        "",
        *lines,
        "",
    ]


def _require(*, name: str, lead_day: int, smallest: object) -> None:
    """Raise if a window's smallest margin is negative.

    Args:
        name: The window's name.
        lead_day: The lead-day.
        smallest: The smallest margin in hours.

    Raises:
        ValueError: If the margin is below zero.
    """
    if smallest is None or float(smallest) < 0:  # ty: ignore[invalid-argument-type]
        msg = (
            f"lead-day {lead_day}: {name} reads an hour after its issue time (margin {smallest} h)"
        )
        raise ValueError(msg)
