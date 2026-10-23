"""Per-substation input arrays for experiment 279's extra columns, built once on the grid.

**Every array has one entry per grid half-hour, like `studies.dp_features.SiteFeatures`, and no
entry reads a value later than its own half-hour.** The arrays are:

- exponentially weighted temperatures with time constants of 1 and 3 days (causal filters);
- the mean temperature of the previous day and of the day before that, that is the trailing
  24-hour mean temperature 48 and 96 half-hours earlier;
- the trailing 24-hour and 72-hour mean 10 m wind speed (ERA5).

ERA5 arrives about 5 days late, so each array is a ceiling for a forecast made at the origin.
"""

from dataclasses import dataclass
from typing import Final

import e00279_common as common  # noqa: F401 (keeps the folder's import order uniform)
import numpy as np
import polars as pl
from scipy.signal import lfilter
from studies.dp_calendar import SMOOTHING_HOURS, trailing_mean_temperatures
from studies.dp_features import ERA5_PATH, FeatureCache

EWMA_TIME_CONSTANTS_DAYS: Final[tuple[float, float]] = (1.0, 3.0)
"""The two exponential time constants of the lag arms, in days."""
HALF_HOUR_HOURS: Final[float] = 0.5
DAY_HALF_HOURS: Final[int] = 48
WIND_COLUMNS: Final[tuple[int, int]] = (1, 2)
"""The columns of `SMOOTHING_HOURS` kept for wind: the 24-hour and 72-hour means."""


@dataclass(frozen=True)
class SiteAux:
    """One substation's extra arrays on the grid.

    Attributes:
        ewma: Exponentially weighted temperatures, shape (T, 2), time constants 1 and 3 days.
        previous_days: The mean temperature of the previous day and of the day before, (T, 2),
            `NaN` for the first four days.
        wind: Trailing 24-hour and 72-hour mean wind speed in m/s, shape (T, 2).
    """

    ewma: np.ndarray
    previous_days: np.ndarray
    wind: np.ndarray


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


def previous_day_means(*, trailing_24h: np.ndarray) -> np.ndarray:
    """Return the previous day's and the day before's mean temperature.

    The trailing 24-hour mean 48 half-hours earlier is the mean over the 24 hours that end a day
    before the row; 96 half-hours earlier, two days before.

    Args:
        trailing_24h: The trailing 24-hour mean temperature on the half-hourly grid.

    Returns:
        An array of shape (len(trailing_24h), 2).
    """
    out = np.full((len(trailing_24h), 2), np.nan)
    for column, shift in enumerate((DAY_HALF_HOURS, 2 * DAY_HALF_HOURS)):
        out[shift:, column] = trailing_24h[:-shift]
    return out


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


def build_site_aux(*, cache: FeatureCache, site: str) -> SiteAux:
    """Build one substation's extra arrays.

    Args:
        cache: The feature cache.
        site: The substation's label.

    Returns:
        The arrays.
    """
    features = cache.sites[site]
    era5 = (
        pl.read_parquet(ERA5_PATH, columns=["time_series_id", "valid_time", "wind_speed_10m"])
        .filter(pl.col("time_series_id") == features.table.time_series_id)
        .sort("valid_time")
    )
    wind = _trailing(
        stamps=era5["valid_time"],
        values=era5["wind_speed_10m"].to_numpy().astype(np.float64),
        grid=cache.end_times,
    )[:, list(WIND_COLUMNS)]
    return SiteAux(
        ewma=ewma_temperatures(temperature_c=features.temperature_c),
        previous_days=previous_day_means(trailing_24h=features.smoothed_temperature_c[:, 1]),
        wind=wind,
    )
