"""Per-substation input arrays for experiment 126's extra columns, built once on the grid.

**Every array has one entry per grid half-hour, like `studies.dp_features.SiteFeatures`.** The
arrays are the inputs of the arms that add columns to T-all:

- exponentially weighted temperatures (S-ewma) with time constants of 1.5 and 4 days;
- the previous 24 hours' minimum and maximum temperature, and their 7-day means (S-mm);
- trailing-mean temperatures after a wind-chill term (C-wind) or an apparent-temperature term
  (C-hw), each at the four smoothings of `studies.dp_calendar.SMOOTHING_HOURS`;
- the day's gap between a weather station's observed temperature and ERA5 (W-stn);
- the school-holiday weekday flag (D-sch), from the Lincolnshire term table.

**No array reads a value later than its own half-hour**, except the station gap, which is a daily
scalar of the 24 hours ending 06:00 UTC on the origin day, so it is known 3 hours before the
09:00 UTC origin.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import cache as memoise
from pathlib import Path
from typing import Final

import e00126_common as common
import numpy as np
import polars as pl
from scipy.signal import lfilter
from studies.dp_assimilation import day_ahead_initialisation
from studies.dp_calendar import SMOOTHING_HOURS, trailing_mean_temperatures
from studies.dp_features import ERA5_PATH, FeatureCache
from studies.sources import REPO_DATA_DIR

EWMA_TIME_CONSTANTS_DAYS: Final[tuple[float, float]] = (1.5, 4.0)
"""The two exponential time constants of S-ewma, in days."""
HALF_HOUR_HOURS: Final[float] = 0.5
WIND_CHILL_FLOOR_KMH: Final[float] = 4.8
"""The North American wind-chill formula's lowest wind speed, in km/h."""
WIND_CHILL_CEILING_C: Final[float] = 10.0
"""The wind-chill term is applied to `min(T, 10)`, so it is continuous and constant above 10 C."""
APPARENT_WIND_WEIGHT: Final[float] = 0.70
APPARENT_VAPOUR_WEIGHT: Final[float] = 0.33
MEAN_WINDOW: Final[tuple[datetime, datetime]] = (common.REPLICA_TUNING_CUT, common.TUNING_CUT)
"""The window of the means that centre the apparent temperature: the 12 months before the tuning
cut, so no later weather enters."""
STATION_DIR: Final[Path] = REPO_DATA_DIR / "studies/downloads/observations/MIDAS-OPEN"
MAX_STATION_DISTANCE_KM: Final[float] = 50.0
MAX_STATION_GAP_SHARE: Final[float] = 0.10
STATION_ORIGIN_OFFSET_HOURS: Final[int] = 3
"""The station gap's last hour is this long before the 09:00 UTC origin (06:00 UTC)."""
TERM_DATES_PATH: Final[Path] = (
    Path(__file__).resolve().parents[1]
    / "00011-school-term-and-return-to-work-columns-stage1a-demand"
    / "lincolnshire_term_dates.csv"
)
EARLIER_TERM_END: Final[date] = date(2021, 7, 21)
"""The last day of Lincolnshire County Council's 2020-21 summer term, as reported by the Grimsby
Telegraph's list of expected dates (not the council's own calendar)."""
TERM_TABLE_START: Final[date] = date(2021, 7, 22)
"""The first day for which the school-holiday flag is defined."""


@dataclass(frozen=True)
class SiteAux:
    """One substation's extra arrays on the grid.

    Attributes:
        ewma: Exponentially weighted temperatures, shape (T, 2), time constants 1.5 and 4 days.
        extremes: The previous 24 hours' minimum and maximum temperature and their 7-day means,
            shape (T, 4), in that order.
        wind_chill: Trailing means at each smoothing of the wind-chilled temperature, (T, 4).
        apparent: Trailing means at each smoothing of the re-centred apparent temperature, (T, 4).
        station_gap: The day's station-minus-ERA5 temperature gap in degrees C, `NaN` when the
            station has no reading or the grid row is past the archive.
        school_holiday: True on weekdays inside a school holiday of the term table.
    """

    ewma: np.ndarray
    extremes: np.ndarray
    wind_chill: np.ndarray
    apparent: np.ndarray
    station_gap: np.ndarray
    school_holiday: np.ndarray


def school_holiday_dates() -> np.ndarray:
    """Return the dates (`datetime64[D]`) between two terms, from `TERM_TABLE_START` on.

    Returns:
        Every calendar date after one term's last day and before the next term's first day,
        weekends included (the caller keeps weekdays).
    """
    table = pl.read_csv(TERM_DATES_PATH, try_parse_dates=True).sort("first_day")
    firsts = table["first_day"].to_list()
    lasts = table["last_day"].to_list()
    previous_last = [EARLIER_TERM_END, *lasts[:-1]]
    holidays: list[date] = []
    for last_day, next_first in zip(previous_last, firsts, strict=True):
        day = last_day + timedelta(days=1)
        while day < next_first:
            holidays.append(day)
            day += timedelta(days=1)
    return np.array(holidays, dtype="datetime64[D]")


def _fill(values: np.ndarray) -> np.ndarray:
    """Return `values` with `NaN` filled by the nearest earlier value (the first by the next)."""
    series = pl.Series(values).fill_nan(None).fill_null(strategy="forward")
    return series.fill_null(strategy="backward").to_numpy()


def ewma_temperatures(*, temperature_c: np.ndarray) -> np.ndarray:
    """Return exponentially weighted temperatures at each time constant, causal.

    Args:
        temperature_c: Temperature on the half-hourly grid.

    Returns:
        An array of shape (len(temperature_c), 2).
    """
    filled = _fill(temperature_c)
    columns = []
    for days in EWMA_TIME_CONSTANTS_DAYS:
        weight = 1.0 - np.exp(-HALF_HOUR_HOURS / (24.0 * days))
        columns.append(
            lfilter([weight], [1.0, -(1.0 - weight)], filled, zi=[(1 - weight) * filled[0]])[0]
        )
    return np.stack(columns, axis=1)


def temperature_extremes(*, temperature_c: np.ndarray) -> np.ndarray:
    """Return the previous 24 hours' minimum and maximum temperature and their 7-day means.

    Args:
        temperature_c: Temperature on the half-hourly grid.

    Returns:
        An array of shape (len(temperature_c), 4); the first 7 days hold `NaN`.
    """
    series = pl.Series(_fill(temperature_c))
    daily = 48
    lowest = series.rolling_min(window_size=daily)
    highest = series.rolling_max(window_size=daily)
    week = 7 * daily
    return np.stack(
        [
            lowest.to_numpy(),
            highest.to_numpy(),
            lowest.rolling_mean(window_size=week).to_numpy(),
            highest.rolling_mean(window_size=week).to_numpy(),
        ],
        axis=1,
    )


def wind_chill_adjustment(*, temperature_c: np.ndarray, wind_m_s: np.ndarray) -> np.ndarray:
    """Return `WC(min(T, 10), v) - min(T, 10)` with the North American formula.

    Args:
        temperature_c: Air temperature in degrees C.
        wind_m_s: 10 m wind speed in m/s.

    Returns:
        The adjustment in degrees C (negative when the wind chills the air).
    """
    capped = np.minimum(temperature_c, WIND_CHILL_CEILING_C)
    speed = np.maximum(3.6 * wind_m_s, WIND_CHILL_FLOOR_KMH) ** 0.16
    chill = 13.12 + 0.6215 * capped - 11.37 * speed + 0.3965 * capped * speed
    return chill - capped


def vapour_pressure_hpa(*, dew_point_c: np.ndarray) -> np.ndarray:
    """Return the vapour pressure in hPa from the dew point, by the Magnus formula."""
    return 6.1094 * np.exp(17.625 * dew_point_c / (243.04 + dew_point_c))


def _trailing(*, stamps: pl.Series, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Return the trailing means of a half-hourly series at each smoothing, on the grid."""
    frame = pl.DataFrame({"valid_time": stamps, "temperature_c": values})
    smoothed = trailing_mean_temperatures(series=frame)
    out = np.empty((len(grid), len(SMOOTHING_HOURS)))
    seconds = smoothed["valid_time"].dt.replace_time_zone(None).to_numpy().astype("datetime64[s]")
    grid_seconds = grid.astype("datetime64[s]")
    for k, hours in enumerate(SMOOTHING_HOURS):
        column = smoothed[f"temperature_c_k{hours}"].to_numpy()
        known = np.isfinite(column)
        out[:, k] = np.interp(
            grid_seconds.astype(np.int64),
            seconds.astype(np.int64)[known],
            column[known],
            left=np.nan,
            right=np.nan,
        )
    return out


def _era5_site(*, time_series_id: int) -> pl.DataFrame:
    return (
        pl.read_parquet(
            ERA5_PATH,
            columns=[
                "time_series_id",
                "valid_time",
                "temperature_2m",
                "dew_point_temperature_2m",
                "wind_speed_10m",
            ],
        )
        .filter(pl.col("time_series_id") == time_series_id)
        .sort("valid_time")
    )


def _haversine_km(*, lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    p1, p2 = np.radians(lat), np.radians(lats)
    dphi = p2 - p1
    dlam = np.radians(lons - lon)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


@memoise
def _station_coordinates() -> pl.DataFrame:
    """Return the MIDAS hourly-weather stations' coordinates (columns `src_id`, `lat`, `lon`)."""
    path = (
        STATION_DIR
        / "_station_metadata"
        / ("midas-open_uk-hourly-weather-obs_dv-202607_station-metadata.csv")
    )
    lines = path.read_text().splitlines()
    start = lines.index("data") + 1
    stop = lines.index("end data")
    table = pl.read_csv(
        "\n".join(lines[start:stop]).encode(), schema_overrides={"src_id": pl.String}
    )
    return table.select("src_id", lat="station_latitude", lon="station_longitude")


@memoise
def _station_hourly() -> pl.DataFrame:
    """Return the archive's hourly air temperature (`src_id`, `time` in UTC, `air_c`)."""
    return pl.read_parquet(
        STATION_DIR / "uk_hourly_weather_obs.parquet", columns=["src_id", "time", "air_temperature"]
    ).rename({"air_temperature": "air_c"})


def station_gap_by_day(
    *, cache: FeatureCache, site: str, era5: pl.DataFrame
) -> tuple[np.ndarray, dict[str, float]]:
    """Return the station-minus-ERA5 temperature gap of the 24 hours before each origin.

    The nearest station that holds hourly air temperature for at least 90% of the hours between
    ERA5's start and the archive's end is used. The site drops out (all `NaN`) when that station is
    further than 50 km. The gap of an origin day D is the mean over the 24 hours ending 06:00 UTC on
    D, with at least 18 hours present.

    Args:
        cache: The feature cache.
        site: The substation's label.
        era5: The site's ERA5 columns `valid_time` and `temperature_2m`.

    Returns:
        The per-row gap on the grid (`NaN` where unavailable), and a record of the station's
        distance in km and the share of rows with a gap.
    """
    table = cache.sites[site].table
    hourly = _station_hourly()
    coordinates = _station_coordinates()
    start, stop = hourly["time"].min(), hourly["time"].max()
    hours = (stop - start).total_seconds() / 3600.0 + 1.0
    coverage = hourly.group_by("src_id").agg(share=pl.col("air_c").is_not_null().sum() / hours)
    stations = coordinates.join(coverage, on="src_id").filter(
        pl.col("share") >= 1.0 - MAX_STATION_GAP_SHARE
    )
    distance = _haversine_km(
        lat=table.latitude,
        lon=table.longitude,
        lats=stations["lat"].to_numpy(),
        lons=stations["lon"].to_numpy(),
    )
    nearest = int(np.argmin(distance))
    record = {"distance_km": float(distance[nearest]), "available_share": 0.0}
    gap = np.full(len(cache.end_times), np.nan)
    if distance[nearest] > MAX_STATION_DISTANCE_KM:
        return gap, record
    chosen = stations["src_id"][nearest]
    observed = hourly.filter(pl.col("src_id") == chosen).select("time", "air_c")
    model = era5.select(
        time=pl.col("valid_time"), model_c=pl.col("temperature_2m").cast(pl.Float64)
    )
    joined = observed.join(model, on="time", how="inner").with_columns(
        gap=pl.col("air_c") - pl.col("model_c")
    )
    # The window of origin day D is the readings stamped 07:00 on D-1 to 06:00 on D inclusive, so
    # shifting by 7 hours puts the whole window on one date, D-1.
    daily = (
        joined.with_columns(
            day=(pl.col("time") - pl.duration(hours=7)).dt.date() + pl.duration(days=1)
        )
        .group_by("day")
        .agg(gap=pl.col("gap").mean(), n=pl.col("gap").is_not_null().sum())
        .filter(pl.col("n") >= 18)
    )
    days = daily["day"].to_numpy().astype("datetime64[D]")
    values = daily["gap"].to_numpy()
    order = np.argsort(days)
    days, values = days[order], values[order]
    origin = origin_day_positions(cache=cache)
    slot = np.searchsorted(days, origin)
    slot = np.clip(slot, 0, len(days) - 1)
    found = days[slot] == origin
    gap[found] = values[slot[found]]
    record["available_share"] = float(np.isfinite(gap).mean())
    return gap, record


def build_site_aux(*, cache: FeatureCache, site: str) -> SiteAux:
    """Build one substation's extra arrays.

    Args:
        cache: The feature cache.
        site: The substation's label.

    Returns:
        The arrays.
    """
    features = cache.sites[site]
    era5 = _era5_site(time_series_id=features.table.time_series_id)
    grid = cache.end_times
    stamps = era5["valid_time"]
    temperature = era5["temperature_2m"].to_numpy().astype(np.float64)
    dew = era5["dew_point_temperature_2m"].to_numpy().astype(np.float64)
    wind = era5["wind_speed_10m"].to_numpy().astype(np.float64)
    chilled = temperature + wind_chill_adjustment(temperature_c=temperature, wind_m_s=wind)
    vapour = vapour_pressure_hpa(dew_point_c=dew)
    window = (stamps >= MEAN_WINDOW[0]) & (stamps < MEAN_WINDOW[1])
    mask = window.to_numpy()
    apparent = (
        temperature
        + APPARENT_VAPOUR_WEIGHT * (vapour - vapour[mask].mean())
        - APPARENT_WIND_WEIGHT * (wind - wind[mask].mean())
    )
    holidays = school_holiday_dates()
    local_date = cache.calendar.local_date
    is_holiday = np.isin(local_date, holidays) & (cache.calendar.day_type == 0)
    return SiteAux(
        ewma=ewma_temperatures(temperature_c=features.temperature_c),
        extremes=temperature_extremes(temperature_c=features.temperature_c),
        wind_chill=_trailing(stamps=stamps, values=chilled, grid=grid),
        apparent=_trailing(stamps=stamps, values=apparent, grid=grid),
        station_gap=station_gap_by_day(cache=cache, site=site, era5=era5)[0],
        school_holiday=is_holiday,
    )


def origin_day_positions(*, cache: FeatureCache) -> np.ndarray:
    """Return each grid row's origin date (`datetime64[D]`) for a day-ahead forecast."""
    return day_ahead_initialisation(end_times=cache.end_times, days_ahead=1).astype("datetime64[D]")
