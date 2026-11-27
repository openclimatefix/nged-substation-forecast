"""The 60 input columns of the step-5 neural network, from the ENS summary and the cleaned power.

Every rule here reproduces the research code the neural network was trained on, value for value:
the golden tests compare each column against the bundle's fixtures. Times are period-ending
half-hour stamps on research's grid (`GRID_START`, 30-minute steps). The native origin O is the
00Z ENS run's initialisation time plus 9 hours, and one forecast holds the valid times from
O + 30 minutes to O + 339 hours.

The input matrix has five groups of columns:

- the 14 ENS means and standard deviations of the day's run at the valid time, precipitation as
  `log1p` of the value clipped at zero;
- 21 date-time columns (`sun_columns` and `calendar_columns`) and 2 lead columns;
- 8 lag columns (`lag_columns`): the day and week lag of cleaned power, at least 24 hours before
  O, each as a value, a missing flag, and an age code, then the two lags' grid indices, which pick
  each lag's expected-weather row (`build_expected_rows`);
- the knot position of the per-Primary offset (`knot_position`);
- 14 thermal-lag and solar-gain ("B1") columns (`b1_columns`), trailing aggregates over a series
  that holds the analysis proxy (`analysis_proxy`) up to O and the day's run after it.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final

import holidays
import numpy as np
import polars as pl
from pvlib import spa
from weather_utils import NWP_ANALYSIS_MEMBER, select_analysis_proxy

from substation_demand_forecaster.bundle import (
    B1_TRAILING_WINDOW_STEPS,
    CALENDAR_TIMEZONE,
    GRID_START,
    GRID_STEP,
    INPUT_HORIZON,
    PUBLICATION_DELAY,
    SUMMARY_HORIZON,
    Bundle,
    CalendarConfig,
    LagConfig,
)
from substation_demand_forecaster.ens_summary import SUMMARY_COLUMNS, WEATHER_VARIABLES

STEPS_PER_DAY: Final[int] = 48
GRID_START_US: Final[int] = int(GRID_START.timestamp() * 1_000_000)
STEP_US: Final[int] = int(GRID_STEP / timedelta(microseconds=1))
INPUT_STEPS: Final[int] = int(INPUT_HORIZON / GRID_STEP)
"""Input rows per Primary: 678 half-hours from O + 30 minutes to O + 339 hours."""
SUMMARY_STEPS: Final[int] = int(SUMMARY_HORIZON / GRID_STEP)
"""Summary rows per Primary and run: 702 half-hours from O + 30 minutes to O + 351 hours."""

ANALYSIS_PROXY_LEADS: Final[tuple[timedelta, timedelta]] = (
    timedelta(hours=9),
    timedelta(hours=33.5),
)
"""Leads after the 00Z initialisation, both exclusive, that the analysis proxy takes from each run.

The window is each run's first day after its own native origin, (O + 0 h, O + 24 h], so
consecutive daily runs' windows touch without overlapping.
"""

EXPECTED_WEATHER_DAYS: Final[tuple[int, int]] = (8, 1)
"""The expected-weather rows cover the half-hours in (O − 8 days, O − 1 day]."""

SUN_OFFSET: Final[timedelta] = timedelta(minutes=15)
"""The sun columns and the clear-sky irradiance are evaluated at the half-hour's midpoint."""

_SPA_PRESSURE_MBAR: Final[float] = 1013.25
_SPA_TEMPERATURE_C: Final[float] = 12.0
_SPA_ATMOSPHERIC_REFRACTION_DEG: Final[float] = 0.5667
_SPA_DELTA_T_S: Final[float] = 67.0
"""`pvlib.solarposition.get_solarposition`'s defaults at sea level, which research used.

`get_solarposition` passes no `delta_t`, so `spa_python` uses its fixed default of 67 seconds rather
than the year's estimate.
"""

_HAURWITZ_W_M2: Final[float] = 1098.0
_HAURWITZ_EXPONENT: Final[float] = -0.059

B1_COLUMNS: Final[tuple[str, ...]] = (
    "t_mean_3h", "t_mean_6h", "t_mean_24h", "t_change_3h", "t_change_6h", "t_min_day",
    "t_max_day", "t_cold_snap_days", "sw_mean_3h", "sw_mean_6h", "sw_mean_24h", "csi_change_3h",
    "cams_prev_day_mean", "cams_prev_day_csi",
)  # fmt: skip
CALENDAR_COLUMNS: Final[tuple[str, ...]] = (
    "sin_hour", "cos_hour", "sin_hour_2", "cos_hour_2", "dow_1", "dow_2", "dow_3", "dow_4",
    "dow_5", "dow_6", "dow_7", "covid", "christmas_inside", "christmas_days", "christmas_tent",
    "bank_holiday", "nearest_days", "nearest_inside",
)  # fmt: skip
SUN_COLUMNS: Final[tuple[str, ...]] = ("cos_zenith", "sin_azimuth", "cos_azimuth")
LEAD_COLUMNS: Final[tuple[str, ...]] = ("lead", "lead_sqrt")
LAG_COLUMNS: Final[tuple[str, ...]] = (
    "day_lag_value", "day_lag_missing", "day_lag_age",
    "week_lag_value", "week_lag_missing", "week_lag_age",
    "day_lag_grid_index", "week_lag_grid_index",
)  # fmt: skip
STEP5_COLUMNS: Final[tuple[str, ...]] = (
    *SUMMARY_COLUMNS, *SUN_COLUMNS, *CALENDAR_COLUMNS, *LEAD_COLUMNS, *LAG_COLUMNS,
    "knot_position", *B1_COLUMNS,
)  # fmt: skip
"""The 60 columns this module builds, in the order `columns.json` of a step-5 bundle lists."""
N_LAG_FREE: Final[int] = 37
"""The first 37 columns, which share one float32 standardiser."""
_B1_START: Final[int] = STEP5_COLUMNS.index(B1_COLUMNS[0])


@dataclass(frozen=True)
class InputMatrix:
    """The input rows of one forecast, raw and standardised."""

    keys: pl.DataFrame
    """One row per (Primary, valid half-hour), ordered by `primary_index` then `valid_time`:
    `primary_index` (Int64), `time_series_id` (Int32), `valid_time` (Datetime, UTC),
    `lead_hours` (Float64), and `target_grid_index` (Int64)."""
    raw: np.ndarray
    """Float32 (rows, 60). A lag's value is its power as a fraction of effective capacity, NaN
    when missing; every other column of the lag block, the knot position, and the grid indices
    hold the values the neural network reads."""
    standardised: np.ndarray
    """Float32 (rows, 60): what the neural network reads."""


@dataclass(frozen=True)
class ExpectedRows:
    """Each Primary's expected-weather row at every half-hour a lag can read."""

    primary_index: np.ndarray
    """Int64 (rows,)."""
    grid_index: np.ndarray
    """Int64 (rows,)."""
    run_init_time: np.ndarray
    """`datetime64[us]` (rows,): the 00Z initialisation of the ENS run behind the row; NaT where
    no run covers the half-hour."""
    raw: np.ndarray
    """Float32 (rows, 28): the seven ENS means (precipitation through `log1p`) and the 21
    date-time columns of the half-hour; NaN where no run covers the half-hour."""
    standardised: np.ndarray
    """Float16 (rows, 28): standardised like the first 37 input columns, clipped, NaN as 0."""


# --- time helpers ---------------------------------------------------------------------------


def grid_index(us: np.ndarray) -> np.ndarray:
    """Return research's grid index of UTC microsecond stamps (Int64)."""
    return ((us - GRID_START_US) // STEP_US).astype(np.int64)


def _stamps_us(grid: np.ndarray) -> np.ndarray:
    return GRID_START_US + grid.astype(np.int64) * STEP_US


def _series_of_us(us: np.ndarray) -> pl.Series:
    return pl.Series(us).cast(pl.Datetime("us", "UTC"))


def native_origin(nwp_init_time: datetime) -> datetime:
    """Return the native origin of a 00Z ENS run: its initialisation time plus 9 hours."""
    return nwp_init_time + PUBLICATION_DELAY


def _origin_index(nwp_init_time: datetime) -> int:
    return int((native_origin(nwp_init_time) - GRID_START) / GRID_STEP)


def _stamp_us(index: int) -> int:
    return GRID_START_US + index * STEP_US


# --- sun, calendar, lead, and knot columns --------------------------------------------------


def _solar_position(*, us: np.ndarray, latitude: float, longitude: float) -> np.ndarray:
    """Return pvlib's SPA apparent zenith and azimuth in degrees at UTC stamps, shape (2, n)."""
    position = spa.solar_position(
        us.astype(np.float64) / 1e6,
        latitude,
        longitude,
        0.0,
        _SPA_PRESSURE_MBAR,
        _SPA_TEMPERATURE_C,
        _SPA_DELTA_T_S,
        _SPA_ATMOSPHERIC_REFRACTION_DEG,
    )
    apparent_zenith, azimuth = position[0], position[4]
    return np.stack([apparent_zenith, azimuth])


def sun_columns(*, valid_us: np.ndarray, latitude: float, longitude: float) -> np.ndarray:
    """Return `cos_zenith`, `sin_azimuth`, and `cos_azimuth` at each half-hour's midpoint.

    The angles are pvlib's SPA apparent zenith and azimuth, rounded to float32 before the
    trigonometry, as research's derived-column table stores them.

    Args:
        valid_us: Period-ending UTC stamps in microseconds.
        latitude: Degrees north.
        longitude: Degrees east.

    Returns:
        Float64 (n, 3).
    """
    zenith, azimuth = _solar_position(
        us=valid_us - int(SUN_OFFSET / timedelta(microseconds=1)),
        latitude=latitude,
        longitude=longitude,
    )
    zenith = np.radians(zenith.astype(np.float32).astype(np.float64))
    azimuth = np.radians(azimuth.astype(np.float32).astype(np.float64))
    return np.stack([np.cos(zenith), np.sin(azimuth), np.cos(azimuth)], axis=1)


def clear_sky_w_m2(*, valid_us: np.ndarray, latitude: float, longitude: float) -> np.ndarray:
    """Return Haurwitz clear-sky irradiance at each half-hour's midpoint, 0 with the sun down."""
    zenith, _ = _solar_position(
        us=valid_us - int(SUN_OFFSET / timedelta(microseconds=1)),
        latitude=latitude,
        longitude=longitude,
    )
    cosine = np.cos(np.radians(zenith))
    safe = np.where(cosine > 0.0, cosine, 1.0)
    return np.where(cosine > 0.0, _HAURWITZ_W_M2 * safe * np.exp(_HAURWITZ_EXPONENT / safe), 0.0)


def _signed_days_to_nearest(*, days: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return target minus day, in days, for the nearest target; a tie takes the earlier target."""
    gaps = (targets[None, :] - days[:, None]).astype("timedelta64[D]").astype(np.int64)
    return gaps[np.arange(len(days)), np.abs(gaps).argmin(axis=1)]


def _holiday_gaps(*, local_dates: np.ndarray, calendar: CalendarConfig) -> np.ndarray:
    """Return days to the nearest England bank holiday (clipped) and to Christmas (NaN far away).

    Both are positive before the day and negative after it. The holiday calendar spans one year
    either side of the dates, which reaches every holiday within the clip.

    Args:
        local_dates: Europe/London dates, `datetime64[D]`.
        calendar: The bundle's clip constants.

    Returns:
        Float64 (n, 2).
    """
    years = local_dates.astype("datetime64[Y]").astype(np.int64) + 1970
    span = range(int(years.min()) - 1, int(years.max()) + 2)
    holiday_dates = np.array(
        sorted(holidays.country_holidays("GB", subdiv="ENG", years=span)), dtype="datetime64[D]"
    )
    christmas_dates = np.array([np.datetime64(f"{year}-12-25") for year in span])
    nearest = _signed_days_to_nearest(days=local_dates, targets=holiday_dates)
    christmas = _signed_days_to_nearest(days=local_dates, targets=christmas_dates)
    clip, christmas_clip = calendar.nearest_holiday_clip_days, calendar.christmas_clip_days
    return np.stack(
        [
            np.clip(nearest, -clip, clip).astype(np.float64),
            np.where(np.abs(christmas) <= christmas_clip, christmas, np.nan).astype(np.float64),
        ],
        axis=1,
    )


def calendar_columns(*, valid_us: np.ndarray, calendar: CalendarConfig) -> np.ndarray:
    """Return the 18 `CALENDAR_COLUMNS` of period-ending UTC stamps.

    The local hour, weekday, and date come from the stamp itself in Europe/London. The lockdown
    flag compares the UTC stamp with the lockdown's first and last days.

    Args:
        valid_us: Period-ending UTC stamps in microseconds.
        calendar: The bundle's holiday and lockdown constants.

    Returns:
        Float64 (n, 18).
    """
    stamps = _series_of_us(valid_us)
    local = stamps.dt.convert_time_zone(CALENDAR_TIMEZONE)
    hour = (local.dt.hour().cast(pl.Float64) + local.dt.minute().cast(pl.Float64) / 60.0).to_numpy()
    weekday = local.dt.weekday().to_numpy()
    first_us = int(calendar.covid_first_day.timestamp() * 1_000_000)
    stop_us = int((calendar.covid_last_day + timedelta(days=1)).timestamp() * 1_000_000)
    covid = ((valid_us >= first_us) & (valid_us < stop_us)).astype(np.float64)
    local_dates = local.dt.date().to_numpy().astype("datetime64[D]")
    unique_dates, position = np.unique(local_dates, return_inverse=True)
    gaps = _holiday_gaps(local_dates=unique_dates, calendar=calendar)[position]
    # float32 rounding of the two day counts, as research's derived-column table stores them
    nearest = gaps[:, 0].astype(np.float32).astype(np.float64)
    christmas = gaps[:, 1].astype(np.float32).astype(np.float64)
    inside = np.isfinite(christmas)
    scaled = np.where(inside, christmas / calendar.christmas_clip_days, 0.0)
    columns = [
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * 2.0 * hour / 24.0),
        np.cos(2 * np.pi * 2.0 * hour / 24.0),
        *((weekday == day).astype(np.float64) for day in range(1, 8)),
        covid,
        inside.astype(np.float64),
        scaled,
        np.where(inside, 1.0 - np.abs(scaled), 0.0),
        (nearest == 0).astype(np.float64),
        nearest / calendar.nearest_holiday_clip_days,
        (np.abs(nearest) < calendar.nearest_holiday_clip_days).astype(np.float64),
    ]
    return np.stack(columns, axis=1)


def lead_columns(*, lead_hours: np.ndarray, lead_scale_hours: float) -> np.ndarray:
    """Return `lead` (lead over the scale) and `lead_sqrt`, float64 (n, 2)."""
    lead = lead_hours / lead_scale_hours
    return np.stack([lead, np.sqrt(np.clip(lead, 0, None))], axis=1)


def knot_position(*, valid_us: np.ndarray, first_knot_month: str) -> np.ndarray:
    """Return months since the first knot month plus the fraction of the valid time's month.

    The position is not clamped to the knot range here; the forward pass clamps it, so a valid
    time after the last knot's month holds the last knot's offset.

    Args:
        valid_us: Period-ending UTC stamps in microseconds.
        first_knot_month: `YYYY-MM` of knot 0.

    Returns:
        Float32 (n,).
    """
    t = valid_us.astype("datetime64[us]")
    start = t.astype("datetime64[M]")
    lo = start.astype("datetime64[us]")
    hi = (start + np.timedelta64(1, "M")).astype("datetime64[us]")
    fraction = (t - lo) / (hi - lo)
    first = np.datetime64(first_knot_month, "M")
    return ((start - first).astype(np.float64) + fraction).astype(np.float32)


# --- lags -----------------------------------------------------------------------------------


def lag_indices(
    *, target_index: np.ndarray, origin_index: np.ndarray, lags: LagConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Return the day and week lag grid indices of each target.

    The day lag is the latest same-half-hour-of-day reading at least `cap_steps` before the
    origin; the week lag the latest same-half-hour-of-week reading.
    """
    newest = origin_index - lags.cap_steps
    gap = target_index - newest
    day = target_index - STEPS_PER_DAY * np.ceil(gap / STEPS_PER_DAY).astype(np.int64)
    week = target_index - lags.week_steps * np.ceil(gap / lags.week_steps).astype(np.int64)
    return day, week


def lag_columns(
    *, power: np.ndarray, lag_index: np.ndarray, target_index: np.ndarray, which: int,
    bundle: Bundle,
) -> np.ndarray:  # fmt: skip
    """Return one lag's standardised value, missing flag, and age code, float32 (rows, 3).

    A missing lag (no cleaned reading, or a negative index) reads value 0, flag 1, and the largest
    age code, the encoding the neural network was trained with.

    Args:
        power: The lag's cleaned power as a fraction of effective capacity, NaN when missing.
        lag_index: The lag's grid index.
        target_index: The target's grid index.
        which: 0 for the day lag, 1 for the week lag.
        bundle: The bundle's lag rule and statistics.

    Returns:
        Float32 (rows, 3).
    """
    config = bundle.lags
    missing = ~np.isfinite(power) | (lag_index < 0)
    mean = np.float64(bundle.standardisers.lag_mean[which])
    std = max(np.float64(bundle.standardisers.lag_std[which]), np.float64(config.std_floor))
    z = np.clip((power.astype(np.float64) - mean) / std, -config.clip_sd, config.clip_sd)
    age = (target_index - lag_index) * 0.5 / config.age_scale_hours
    missing_age = (config.missing_age_code_day, config.missing_age_code_week)[which]
    out = np.zeros((len(power), 3), dtype=np.float32)
    out[:, 0] = np.where(missing, 0.0, z)
    out[:, 1] = missing
    out[:, 2] = np.where(missing, missing_age, age)
    return out


def _power_on_grid(
    *, cleaned_power: pl.DataFrame, bundle: Bundle, first: int, length: int
) -> np.ndarray:
    """Return each Primary's cleaned power over effective capacity on a grid window, NaN if absent.

    Args:
        cleaned_power: `time_series_id`, `time`, and `power` of the cleaned readings.
        bundle: The bundle's Primaries and their effective capacities.
        first: The window's first grid index.
        length: The window's length in half-hours.

    Returns:
        Float32 (primaries, length), covering grid indices `first` to `first + length - 1`.
    """
    frame = (
        cleaned_power.select("time_series_id", "time", "power")
        .join(
            bundle.primaries.select("time_series_id", "primary_index", "effective_capacity_mw"),
            on="time_series_id",
            how="inner",
        )
        .with_columns(position=pl.col("time").dt.epoch("us"))
    )
    position = grid_index(frame["position"].to_numpy()) - first
    inside = (position >= 0) & (position < length)
    out = np.full((bundle.primaries.height, length), np.nan, dtype=np.float32)
    fraction = frame["power"].cast(pl.Float64) / frame["effective_capacity_mw"]
    out[frame["primary_index"].to_numpy()[inside], position[inside]] = fraction.to_numpy()[inside]
    return out


# --- B1 columns -----------------------------------------------------------------------------


def trailing_mean(*, a: np.ndarray, n: int, at: np.ndarray) -> np.ndarray:
    """Return the mean of the finite values of `a` over the `n` steps ending at each position.

    A window holding no finite value gives 0, and a window reaching before the start of `a` uses
    what exists.
    """
    finite = np.isfinite(a)
    total = np.concatenate([[0.0], np.cumsum(np.where(finite, a, 0.0))])
    count = np.concatenate([[0], np.cumsum(finite)])
    start = np.maximum(at + 1 - n, 0)
    n_found = count[at + 1] - count[start]
    return (total[at + 1] - total[start]) / np.maximum(n_found, 1)


def b1_columns(
    *, temperature: np.ndarray, shortwave: np.ndarray, clear_sky: np.ndarray,
    local_date_codes: np.ndarray, at: np.ndarray, bundle: Bundle,
) -> np.ndarray:  # fmt: skip
    """Return the 14 `B1_COLUMNS` at positions `at` of one Primary's window, float64 (len(at), 14).

    The window starts `lookback_steps` (12 days) before the native origin and holds the analysis
    proxy up to the origin and the day's run after it, with NaN where neither has a value. Every
    aggregate skips the NaN values: a trailing mean of no finite value is 0, a change across a NaN
    is 0, and a daily minimum or maximum of no finite value stays NaN, which the B1 standardiser
    maps to 0. The two previous-day CAMS columns are 0, as the CAMS-free bundle (arm `drop`) was
    trained.

    Args:
        temperature: Mean 2 m temperature over the window, degrees C.
        shortwave: Mean shortwave irradiance over the window, W/m2.
        clear_sky: Haurwitz clear-sky irradiance over the window, W/m2.
        local_date_codes: The Europe/London date of the half-hour ending at each stamp, as days
            since 1970-01-01.
        at: Window positions of the forecast's valid times; each at least 12 steps in.
        bundle: The bundle's B1 constants.

    Returns:
        Float64 (len(at), 14).
    """
    b1 = bundle.b1
    positions = np.arange(len(temperature))
    out = np.zeros((len(at), len(B1_COLUMNS)))
    for k, n_steps in enumerate(B1_TRAILING_WINDOW_STEPS):
        out[:, k] = trailing_mean(a=temperature, n=n_steps, at=at)
        out[:, 8 + k] = trailing_mean(a=shortwave, n=n_steps, at=at)
    out[:, 3] = np.nan_to_num(temperature[at] - temperature[at - 6], nan=0.0)
    out[:, 4] = np.nan_to_num(temperature[at] - temperature[at - 12], nan=0.0)
    new_day = np.r_[False, local_date_codes[1:] != local_date_codes[:-1]]
    group_starts = np.flatnonzero(np.r_[True, new_day[1:]])
    group = np.cumsum(new_day)
    out[:, 5] = np.fmin.reduceat(temperature, group_starts)[group[at]]
    out[:, 6] = np.fmax.reduceat(temperature, group_starts)[group[at]]
    mean_24h = trailing_mean(a=temperature, n=STEPS_PER_DAY, at=positions)
    warm = np.nan_to_num(mean_24h, nan=-99.0) > b1.cold_snap_threshold_c
    last_warm = np.maximum.accumulate(np.where(warm, positions, -1))
    days = np.where(last_warm >= 0, (positions - last_warm) / STEPS_PER_DAY, b1.cold_snap_cap_days)
    out[:, 7] = np.minimum(days, b1.cold_snap_cap_days)[at]
    csi = np.where(
        (clear_sky > b1.min_clear_sky_w_m2) & np.isfinite(shortwave),
        np.clip(shortwave / np.maximum(clear_sky, 1.0), 0.0, b1.max_clear_sky_index),
        np.nan,
    )
    held = csi[np.maximum.accumulate(np.where(np.isfinite(csi), positions, 0))]
    out[:, 11] = np.nan_to_num(held[at] - held[at - 6], nan=0.0)
    return out


def analysis_proxy(summary: pl.LazyFrame) -> pl.LazyFrame:
    """Return the analysis proxy of the ENS mean temperature and shortwave.

    Each run contributes its leads inside `ANALYSIS_PROXY_LEADS`, and `select_analysis_proxy` keeps
    the freshest run with a value at each valid time.

    Args:
        summary: `summarise_runs`' output.

    Returns:
        One row per (`time_series_id`, `valid_time`): those two, `nwp_init_time`,
        `mean_temperature_2m`, and `mean_shortwave`.
    """
    lead = pl.col("valid_time") - pl.col("nwp_init_time")
    first_day = summary.filter(
        lead > ANALYSIS_PROXY_LEADS[0], lead < ANALYSIS_PROXY_LEADS[1]
    ).select(
        "nwp_init_time",
        "time_series_id",
        "valid_time",
        "mean_temperature_2m",
        "mean_shortwave",
        ensemble_member=pl.lit(NWP_ANALYSIS_MEMBER),
    )
    return select_analysis_proxy(
        first_day, group_key="time_series_id", init_time_col="nwp_init_time"
    )


def _values_on_grid(
    *, frame: pl.DataFrame, columns: tuple[str, ...], bundle: Bundle, first: int, length: int
) -> np.ndarray:
    """Return `columns` of a (`time_series_id`, `valid_time`) frame on a grid window, NaN if absent.

    Args:
        frame: Rows keyed by `time_series_id` and `valid_time`.
        columns: The columns to place on the grid.
        bundle: The bundle's Primaries.
        first: The window's first grid index.
        length: The window's length in half-hours.

    Returns:
        Float64 (len(columns), primaries, length).
    """
    joined = frame.join(
        bundle.primaries.select("time_series_id", "primary_index"), on="time_series_id"
    )
    position = grid_index(joined["valid_time"].dt.epoch("us").to_numpy()) - first
    inside = (position >= 0) & (position < length)
    primary = joined["primary_index"].to_numpy()[inside]
    out = np.full((len(columns), bundle.primaries.height, length), np.nan)
    for k, column in enumerate(columns):
        out[k, primary, position[inside]] = joined[column].cast(pl.Float64).to_numpy()[inside]
    return out


def _local_date_codes(*, valid_us: np.ndarray) -> np.ndarray:
    """Return the Europe/London date of the half-hour ending at each stamp, as days since 1970."""
    starts = _series_of_us(valid_us - STEP_US)
    return starts.dt.convert_time_zone(CALENDAR_TIMEZONE).dt.date().cast(pl.Int32).to_numpy()


def _run_b1(
    *, summary: pl.DataFrame, nwp_init_time: datetime, origin_index: int, bundle: Bundle
) -> np.ndarray:
    """Return the B1 columns of every Primary at the run's 702 summary valid times.

    Args:
        summary: `summarise_runs`' output for the day's run and the runs before it.
        nwp_init_time: The day's 00Z run.
        origin_index: The grid index of the day's native origin.
        bundle: The bundle's Primaries and B1 constants.

    Returns:
        Float64 (primaries, SUMMARY_STEPS, 14).
    """
    lookback = bundle.b1.lookback_steps
    first = origin_index - lookback
    length = lookback + 1 + SUMMARY_STEPS
    history = analysis_proxy(
        summary.lazy().filter(pl.col("nwp_init_time") < nwp_init_time)
    ).collect()
    weather = ("mean_temperature_2m", "mean_shortwave")
    series = _values_on_grid(
        frame=history, columns=weather, bundle=bundle, first=first, length=length
    )
    run = summary.filter(pl.col("nwp_init_time") == nwp_init_time)
    today = _values_on_grid(frame=run, columns=weather, bundle=bundle, first=first, length=length)
    series[:, :, lookback + 1 :] = today[:, :, lookback + 1 :]
    window_us = _stamps_us(np.arange(first, first + length))
    local_dates = _local_date_codes(valid_us=window_us)
    at = np.arange(lookback + 1, length)
    out = np.zeros((bundle.primaries.height, SUMMARY_STEPS, len(B1_COLUMNS)))
    for primary in bundle.primaries.iter_rows(named=True):
        index = primary["primary_index"]
        clear = clear_sky_w_m2(
            valid_us=window_us, latitude=primary["latitude"], longitude=primary["longitude"]
        )
        out[index] = b1_columns(
            temperature=series[0, index],
            shortwave=series[1, index],
            clear_sky=clear,
            local_date_codes=local_dates,
            at=at,
            bundle=bundle,
        )
    return out


# --- expected-weather rows ------------------------------------------------------------------


def _date_time_columns(
    *, valid_us: np.ndarray, latitude: float, longitude: float, bundle: Bundle
) -> np.ndarray:
    """Return the 21 date-time columns (sun, then calendar), float64 (n, 21)."""
    return np.concatenate(
        [
            sun_columns(valid_us=valid_us, latitude=latitude, longitude=longitude),
            calendar_columns(valid_us=valid_us, calendar=bundle.calendar),
        ],
        axis=1,
    )


def _weather_columns(values: np.ndarray) -> np.ndarray:
    """Return the network's weather columns from summary values: precipitation through `log1p`.

    Args:
        values: Float64 (..., 7 or 14), in the order of `WEATHER_VARIABLES` (and again for the
            standard deviations).

    Returns:
        Float64, the shape of `values`.
    """
    out = values.copy()
    for k in range(values.shape[-1]):
        if WEATHER_VARIABLES[k % len(WEATHER_VARIABLES)] == "precipitation":
            out[..., k] = np.log1p(np.clip(values[..., k], 0, None))
    return out


def build_expected_rows(
    *, bundle: Bundle, summary: pl.DataFrame, nwp_init_time: datetime
) -> ExpectedRows:
    """Return every Primary's expected-weather row at each half-hour a lag of this forecast reads.

    For each half-hour g in (O − 8 days, O − 1 day], the row comes from the ENS summary of the
    freshest run whose native origin is strictly before g; a missing run falls back to the
    next-freshest run that covers g. The row holds that run's seven means (precipitation through
    `log1p`) and the 21 date-time columns of g. Where no run covers g, the row is NaN, which
    standardises to 0.

    Args:
        bundle: The bundle's Primaries and standardisers.
        summary: `summarise_runs`' output for the runs up to the day's run.
        nwp_init_time: The day's 00Z run.

    Returns:
        One row per Primary and half-hour, ordered by `primary_index` then `grid_index`.
    """
    origin_index = _origin_index(nwp_init_time)
    last = origin_index - EXPECTED_WEATHER_DAYS[1] * STEPS_PER_DAY
    length = (EXPECTED_WEATHER_DAYS[0] - EXPECTED_WEATHER_DAYS[1]) * STEPS_PER_DAY
    first = last - length + 1
    freshest = (
        summary.lazy()
        .filter(
            pl.col("nwp_init_time") + PUBLICATION_DELAY < pl.col("valid_time"),
            pl.col("valid_time").dt.epoch("us").is_between(_stamp_us(first), _stamp_us(last)),
        )
        .group_by("time_series_id", "valid_time")
        .agg(pl.all().sort_by("nwp_init_time").last())
        .collect()
    )
    means = tuple(f"mean_{variable}" for variable in WEATHER_VARIABLES)
    weather = _values_on_grid(
        frame=freshest, columns=means, bundle=bundle, first=first, length=length
    )
    init = _values_on_grid(
        frame=freshest.with_columns(pl.col("nwp_init_time").dt.epoch("us").cast(pl.Float64)),
        columns=("nwp_init_time",),
        bundle=bundle,
        first=first,
        length=length,
    )[0]
    grid = np.arange(first, first + length)
    valid_us = _stamps_us(grid)
    rows = []
    for primary in bundle.primaries.iter_rows(named=True):
        date_time = _date_time_columns(
            valid_us=valid_us, latitude=primary["latitude"], longitude=primary["longitude"],
            bundle=bundle,
        )  # fmt: skip
        values = np.concatenate(
            [_weather_columns(weather[:, primary["primary_index"]].T), date_time], axis=1
        )
        # A half-hour no run covers holds no weather, and research's table held no row there. A
        # covered half-hour keeps its row, with NaN only in a null weather column, as in research.
        values[~np.isfinite(init[primary["primary_index"]])] = np.nan
        rows.append(values)
    raw = np.concatenate(rows).astype(np.float32)
    positions = list(bundle.network.expected_weather_positions)
    mean = bundle.standardisers.x37_mean[positions]
    std = bundle.standardisers.x37_std[positions]
    clip = bundle.network.input_clip_sd
    z = np.clip((raw - mean) / std, -clip, clip)
    n = bundle.primaries.height
    return ExpectedRows(
        primary_index=np.repeat(np.arange(n, dtype=np.int64), length),
        grid_index=np.tile(grid, n),
        run_init_time=init.reshape(-1).astype("datetime64[us]"),
        raw=raw,
        standardised=np.nan_to_num(z, nan=0.0).astype(np.float16),
    )


# --- the input matrix -----------------------------------------------------------------------


def _check_columns(bundle: Bundle) -> None:
    if bundle.column_names != STEP5_COLUMNS:
        msg = "the bundle's column order differs from the step-5 recipe's"
        raise ValueError(msg)


def raw_input_matrix(
    *, bundle: Bundle, summary: pl.DataFrame, cleaned_power: pl.DataFrame, nwp_init_time: datetime
) -> tuple[pl.DataFrame, np.ndarray]:
    """Return the keys and the raw (unstandardised) 60 input columns of one forecast.

    Args:
        bundle: The bundle.
        summary: `summarise_runs`' output for the day's run and the runs of the 13 days before it.
            It must hold no run initialised after `nwp_init_time`.
        cleaned_power: `time_series_id`, `time`, and `power` of the Primaries' cleaned readings
            over the lag window, the 7 days ending 1 day before the native origin.
        nwp_init_time: The day's 00Z run.

    Returns:
        The keys and the raw columns, as `InputMatrix.keys` and `InputMatrix.raw` describe them.
    """
    _check_columns(bundle)
    origin_index = _origin_index(nwp_init_time)
    origin_us = _stamp_us(origin_index)
    n_primaries = bundle.primaries.height
    run = summary.filter(pl.col("nwp_init_time") == nwp_init_time)
    summary_values = _values_on_grid(
        frame=run, columns=SUMMARY_COLUMNS, bundle=bundle, first=origin_index + 1,
        length=SUMMARY_STEPS,
    )[:, :, :INPUT_STEPS]  # fmt: skip
    target = origin_index + 1 + np.arange(INPUT_STEPS)
    valid_us = _stamps_us(target)
    lead_hours = (valid_us - origin_us) / 3.6e9
    calendar = calendar_columns(valid_us=valid_us, calendar=bundle.calendar)
    lead = lead_columns(lead_hours=lead_hours, lead_scale_hours=bundle.network.lead_scale_hours)
    knots = knot_position(valid_us=valid_us, first_knot_month=bundle.network.first_knot_month)
    origin = np.full(INPUT_STEPS, origin_index)
    day, week = lag_indices(target_index=target, origin_index=origin, lags=bundle.lags)
    lag_window = bundle.lags.week_steps
    lag_first = origin_index - bundle.lags.cap_steps - lag_window + 1
    power = _power_on_grid(
        cleaned_power=cleaned_power, bundle=bundle, first=lag_first, length=lag_window
    )
    b1 = _run_b1(
        summary=summary, nwp_init_time=nwp_init_time, origin_index=origin_index, bundle=bundle
    )
    raw = np.zeros((n_primaries, INPUT_STEPS, len(STEP5_COLUMNS)), dtype=np.float32)
    for primary in bundle.primaries.iter_rows(named=True):
        p = primary["primary_index"]
        sun = sun_columns(
            valid_us=valid_us, latitude=primary["latitude"], longitude=primary["longitude"]
        )
        lag_block = []
        for which, index in enumerate((day, week)):
            fraction = power[p, index - lag_first]
            encoded = lag_columns(
                power=fraction, lag_index=index, target_index=target, which=which, bundle=bundle
            )
            lag_block.append(np.stack([fraction, encoded[:, 1], encoded[:, 2]], axis=1))
        raw[p] = np.concatenate(
            [
                _weather_columns(summary_values[:, p].T),
                sun,
                calendar,
                lead,
                *lag_block,
                np.stack([day, week], axis=1),
                knots[:, None],
                b1[p, :INPUT_STEPS],
            ],
            axis=1,
        )
    keys = pl.DataFrame(
        {
            "primary_index": np.repeat(np.arange(n_primaries, dtype=np.int64), INPUT_STEPS),
            "time_series_id": np.repeat(bundle.primaries["time_series_id"].to_numpy(), INPUT_STEPS),
            "valid_time": _series_of_us(np.tile(valid_us, n_primaries)),
            "lead_hours": np.tile(lead_hours, n_primaries),
            "target_grid_index": np.tile(target, n_primaries),
        }
    )
    return keys, raw.reshape(n_primaries * INPUT_STEPS, len(STEP5_COLUMNS))


def standardise_input_matrix(
    *, raw: np.ndarray, target_index: np.ndarray, bundle: Bundle
) -> np.ndarray:
    """Return the standardised matrix the neural network reads from the raw columns.

    The 37 lag-free columns use float32 arithmetic, a clip at the bundle's input clip, and NaN as
    0. Each lag value is standardised in float64 and clipped, and a missing lag gets the missing
    encoding. The grid indices and the knot position pass through. The B1 columns use float32
    arithmetic, no clip, and a non-finite result as 0.

    Args:
        raw: `InputMatrix.raw`.
        target_index: The grid index of each row's valid time.
        bundle: The bundle's standardisers.

    Returns:
        Float32 (rows, 60).
    """
    standardisers = bundle.standardisers
    clip = bundle.network.input_clip_sd
    x = np.zeros(raw.shape, dtype=np.float32)
    z37 = np.clip(
        (raw[:, :N_LAG_FREE] - standardisers.x37_mean) / standardisers.x37_std, -clip, clip
    )
    x[:, :N_LAG_FREE] = np.nan_to_num(z37, nan=0.0)
    slices = bundle.network.column_slices
    for which, name in enumerate(("day_lag", "week_lag")):
        start, stop = slices[name]
        index_column = bundle.network.lag_grid_index_columns[which]
        lag_index = raw[:, index_column].astype(np.int64)
        x[:, start:stop] = lag_columns(
            power=raw[:, start], lag_index=lag_index, target_index=target_index, which=which,
            bundle=bundle,
        )  # fmt: skip
        x[:, index_column] = raw[:, index_column]
    knot = bundle.network.knot_position_column
    x[:, knot] = raw[:, knot]
    b1 = (raw[:, _B1_START:] - standardisers.b1_mean) / standardisers.b1_std
    x[:, _B1_START:] = np.where(np.isfinite(b1), b1, 0.0)
    return x


def build_input_matrix(
    *, bundle: Bundle, summary: pl.DataFrame, cleaned_power: pl.DataFrame, nwp_init_time: datetime
) -> InputMatrix:
    """Return the 60 input columns of one forecast, raw and standardised.

    Args:
        bundle: The bundle.
        summary: `summarise_runs`' output for the day's run and the runs of the 13 days before it.
        cleaned_power: `time_series_id`, `time`, and `power` of the Primaries' cleaned readings
            over the lag window.
        nwp_init_time: The day's 00Z run.

    Returns:
        The keys, the raw columns, and the standardised columns.

    Raises:
        ValueError: If the bundle's column order differs from the step-5 recipe's.
    """
    keys, raw = raw_input_matrix(
        bundle=bundle, summary=summary, cleaned_power=cleaned_power, nwp_init_time=nwp_init_time
    )
    standardised = standardise_input_matrix(
        raw=raw, target_index=keys["target_grid_index"].to_numpy(), bundle=bundle
    )
    return InputMatrix(keys=keys, raw=raw, standardised=standardised)
