"""Climatology and manual-heuristic reference forecasts at a BSP, for absolute skill.

Both are written by this experiment's own code because no stored reference exists for a BSP.

- **Climatology.** The empirical quantiles of the BSP's default-row readings before the scored
  year, by local month, half-hour of the local day, and weekend flag.
- **Manual heuristic.** The quantiles of an ensemble of the readings at the same weekday and time
  of day 1 to 6 weeks earlier and 49 to 55 weeks earlier (as `ManualHeuristicForecaster` does),
  keeping a member only if its reading ended at least 3.5 hours before the origin. A row with no
  member is dropped.
"""

from datetime import UTC, datetime
from typing import Final

import e00161_common as common
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache, naive_utc

WEEK: Final[np.timedelta64] = np.timedelta64(7, "D")
WEEKLY_LAGS: Final[tuple[int, ...]] = tuple(range(1, 7))
ANNUAL_LAGS: Final[tuple[int, ...]] = tuple(range(49, 56))
TELEMETRY_DELAY: Final[np.timedelta64] = np.timedelta64(210, "m")
CLIMATOLOGY_END: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)


def _positions(*, cache: FeatureCache, stamps: np.ndarray) -> np.ndarray:
    """Return the grid index of each stamp, `-1` where it is off the grid."""
    index = np.searchsorted(cache.end_times, stamps)
    index = np.clip(index, 0, len(cache.end_times) - 1)
    return np.where(cache.end_times[index] == stamps, index, -1)


def manual_heuristic(*, cache: FeatureCache, bsp: str, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the frame with its mean and quantiles replaced by the manual heuristic's."""
    y = cache.sites[bsp].y
    valid = frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    origin = frame["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy()
    members = np.full((len(frame), len(WEEKLY_LAGS) + len(ANNUAL_LAGS)), np.nan)
    for column, lag in enumerate((*WEEKLY_LAGS, *ANNUAL_LAGS)):
        stamps = valid - lag * WEEK
        position = _positions(cache=cache, stamps=stamps)
        ok = (position >= 0) & (stamps <= origin - TELEMETRY_DELAY)
        members[ok, column] = y[position[ok]]
    has_member = np.isfinite(members).any(axis=1)
    quantiles = np.full((len(frame), len(common.QUANTILES)), np.nan)
    quantiles[has_member] = np.nanquantile(members[has_member], common.LEVELS, axis=1).T
    mean = np.full(len(frame), np.nan)
    mean[has_member] = np.nanmean(members[has_member], axis=1)
    out = frame.with_columns(
        pl.Series("mean", mean.astype(np.float32)),
        *[
            pl.Series(name, quantiles[:, i].astype(np.float32))
            for i, name in enumerate(common.QUANTILES)
        ],
    )
    return out.filter(pl.Series(has_member))


def climatology(*, cache: FeatureCache, bsp: str, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the frame with its mean and quantiles replaced by the climatology's."""
    features = cache.sites[bsp]
    train = features.default_rows & np.isfinite(features.y)
    train &= cache.end_times < naive_utc(moment=CLIMATOLOGY_END)

    def keys(stamps: np.ndarray) -> np.ndarray:
        series = pl.Series(stamps).dt.replace_time_zone("UTC").dt.convert_time_zone("Europe/London")
        month = series.dt.month().to_numpy().astype(np.int64)
        half_hour = (series.dt.hour() * 2 + series.dt.minute() // 30).to_numpy().astype(np.int64)
        weekend = (series.dt.weekday() >= 6).to_numpy().astype(int)
        return month * 1000 + half_hour * 10 + weekend

    train_keys = keys(cache.end_times[train])
    table = {
        int(k): np.quantile(features.y[train][train_keys == k], common.LEVELS)
        for k in np.unique(train_keys)
    }
    row_keys = keys(frame["valid_time"].dt.replace_time_zone(None).to_numpy())
    quantiles = np.array([table[int(k)] for k in row_keys])
    return frame.with_columns(
        pl.Series("mean", quantiles[:, 5].astype(np.float32)),
        *[
            pl.Series(name, quantiles[:, i].astype(np.float32))
            for i, name in enumerate(common.QUANTILES)
        ],
    )
