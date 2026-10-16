"""The half-hour panel, the same-clock reference, and the model-free step features F1 to F7.

All arrays have one row per series (in `SERIES_IDS` order) and one column per half-hour stamp of
the grid that starts at 2022-01-01 00:00 UTC. A stamp marks the end of its half-hour.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

import numpy as np
import polars as pl
from e00041_common import MVA_SERIES, SERIES_IDS, WORK_DIR, logged_intervals

GRID_START: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
GRID_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
N_STAMPS: Final[int] = int((GRID_END - GRID_START).total_seconds() // 1800)
WEEK: Final[int] = 336
REF_WEEKS: Final[tuple[int, ...]] = (-4, -3, -2, -1, 1, 2, 3, 4)
MIN_REF_VALUES: Final[int] = 3
N_NULL: Final[int] = 400
PAD_STAMPS: Final[int] = 48
ADJACENCY: Final[dict[int, tuple[int, ...]]] = {
    3: (15, 16, 26),
    7: (24,),
    8: (10, 14, 24),
    9: (11, 24),
    10: (8, 12, 13, 14, 24, 25),
    11: (9, 25),
    12: (10, 13, 14, 16, 25),
    13: (10, 12, 14, 16, 25),
    14: (8, 10, 12, 13, 15, 16),
    15: (3, 14, 17, 18),
    16: (3, 12, 13, 14, 26),
    17: (15, 18, 26),
    18: (15, 17),
    24: (7, 8, 9, 10, 25),
    25: (10, 11, 12, 13, 24),
    26: (3, 16, 17),
}
INDEX: Final[dict[int, int]] = {sid: i for i, sid in enumerate(SERIES_IDS)}


def stamp_of(*, utc_time: datetime) -> int:
    """Return the index of the half-hour containing `utc_time` (floor)."""
    return int((utc_time - GRID_START).total_seconds() // 1800)


def time_of(*, stamp: int) -> datetime:
    """Return the UTC instant at the end of `stamp`."""
    return GRID_START + timedelta(minutes=30 * stamp)


def first_affected(*, close_utc: datetime) -> int:
    """First stamp whose half-hour ends after the close."""
    return stamp_of(utc_time=close_utc) + 1


def last_affected(*, open_utc: datetime) -> int:
    """Last stamp affected by a closure (inclusive): the half-hour after the open's own."""
    return stamp_of(utc_time=open_utc) + 1


@dataclass
class Panel:
    """Raw power, the hindsight quantiles, and the exclusion mask of logged intervals."""

    obs: np.ndarray
    q10: np.ndarray
    q50: np.ndarray
    q90: np.ndarray
    excluded: np.ndarray
    offset_half_hours: np.ndarray
    local_hour: np.ndarray
    local_month: np.ndarray


def _local_columns() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stamps = pl.DataFrame(
        {"t": [time_of(stamp=i) for i in range(N_STAMPS)]}, schema={"t": pl.Datetime("us", "UTC")}
    ).with_columns(local=pl.col("t").dt.convert_time_zone("Europe/London"))
    local = stamps["local"]
    offset = (
        local.dt.replace_time_zone(None) - stamps["t"].dt.replace_time_zone(None)
    ).dt.total_minutes().to_numpy() // 30
    hour = (local.dt.hour().to_numpy() + local.dt.minute().to_numpy() / 60.0).astype(float)
    return offset.astype(int), hour, local.dt.month().to_numpy()


def logged_exclusion() -> np.ndarray:
    """Boolean [series, stamp]: inside any logged interval touching the series, padded 24 hours."""
    excluded = np.zeros((len(SERIES_IDS), N_STAMPS), dtype=bool)
    for row in logged_intervals().iter_rows(named=True):
        lo = max(first_affected(close_utc=row["start_utc"]) - PAD_STAMPS, 0)
        hi = min(last_affected(open_utc=row["end_utc"]) + PAD_STAMPS + 1, N_STAMPS)
        if hi <= 0 or lo >= N_STAMPS:
            continue
        for side in (row["a"], row["b"]):
            if side is not None and side.isdigit() and int(side) in INDEX:
                excluded[INDEX[int(side)], lo:hi] = True
    return excluded


def load_panel() -> Panel:
    """Read the hindsight files into arrays; MVA series become |power| with midday NaN in summer."""
    shape = (len(SERIES_IDS), N_STAMPS)
    arrays = {name: np.full(shape, np.nan) for name in ("obs", "q10", "q50", "q90")}
    for sid in SERIES_IDS:
        frame = pl.read_parquet(WORK_DIR / "hindsight" / f"{sid}.parquet")
        stamp = (frame["valid_time"].dt.epoch("s").to_numpy() - int(GRID_START.timestamp())) // 1800
        ok = (stamp >= 0) & (stamp < N_STAMPS)
        for name, column in (
            ("obs", "obs_mw"),
            ("q10", "q10_mw"),
            ("q50", "q50_mw"),
            ("q90", "q90_mw"),
        ):
            arrays[name][INDEX[sid], stamp[ok]] = frame[column].to_numpy()[ok]
    offset, hour, month = _local_columns()
    midday_summer = (hour >= 10) & (hour < 15) & (month >= 4) & (month <= 9)
    for sid in MVA_SERIES:
        for array in arrays.values():
            array[INDEX[sid]] = np.abs(array[INDEX[sid]])
            array[INDEX[sid], midday_summer] = np.nan
    return Panel(
        **arrays,
        excluded=logged_exclusion(),
        offset_half_hours=offset,
        local_hour=hour,
        local_month=month,
    )


def reference(*, panel: Panel, row: int, stamps: np.ndarray) -> np.ndarray:
    """Median of the same local half-hour and weekday over weeks -4..-1 and +1..+4.

    Candidates inside an excluded interval are dropped; fewer than `MIN_REF_VALUES` gives NaN.
    """
    stamps = np.asarray(stamps)
    candidates = np.stack([stamps + k * WEEK for k in REF_WEEKS])
    inside = (candidates >= 0) & (candidates < N_STAMPS)
    safe = np.clip(candidates, 0, N_STAMPS - 1)
    own = panel.offset_half_hours[np.clip(stamps, 0, N_STAMPS - 1)]
    shifted = safe + (own[None, :] - panel.offset_half_hours[safe])
    inside &= (shifted >= 0) & (shifted < N_STAMPS)
    shifted = np.clip(shifted, 0, N_STAMPS - 1)
    values = np.where(inside & ~panel.excluded[row, shifted], panel.obs[row, shifted], np.nan)
    count = np.sum(~np.isnan(values), axis=0)
    with np.errstate(all="ignore"):
        median = np.nanmedian(values, axis=0)
    return np.where(count >= MIN_REF_VALUES, median, np.nan)


def excess(*, panel: Panel, row: int, lo: int, hi: int) -> np.ndarray:
    """`obs - ref` for stamps `lo` to `hi` (exclusive)."""
    stamps = np.arange(lo, hi)
    valid = (stamps >= 0) & (stamps < N_STAMPS)
    out = np.full(len(stamps), np.nan)
    inside = stamps[valid]
    out[valid] = panel.obs[row, inside] - reference(panel=panel, row=row, stamps=inside)
    return out


def _mean(values: np.ndarray) -> float:
    return float(np.nanmean(values)) if np.any(~np.isnan(values)) else float("nan")


def window_length(*, first: int, last: int) -> int:
    """Half-hours either side of a step: 4 hours, shorter for short events, 24 hours from a day."""
    length = last - first + 1
    if length >= 48:
        return 48
    return int(min(8, max(4, length)))


def step_at(*, panel: Panel, row: int, edge: int, window: int) -> float:
    """Mean excess over `window` stamps from `edge`, minus the mean over the `window` before."""
    after = excess(panel=panel, row=row, lo=edge, hi=edge + window)
    before = excess(panel=panel, row=row, lo=edge - window, hi=edge)
    return _mean(after) - _mean(before)


def null_steps(
    *,
    panel: Panel,
    row: int,
    edge: int,
    window: int,
    avoid: np.ndarray,
    rng: np.random.Generator,
    n_draws: int = N_NULL,
) -> np.ndarray:
    """Up to `n_draws` steps at the same local clock time on quiet days, with the same window.

    `avoid` is a boolean stamp mask of the card's own interval and its padding; draws whose
    windows touch it, or touch this series' logged intervals, are dropped.
    """
    days = np.arange(-180, 181)
    days = days[days != 0]
    rng.shuffle(days)
    own = panel.offset_half_hours[np.clip(edge, 0, N_STAMPS - 1)]
    cand = edge + days * 48
    cand = cand + own - panel.offset_half_hours[np.clip(cand, 0, N_STAMPS - 1)]
    cand = cand[(cand - window - 1 >= 4 * WEEK) & (cand + window + 1 <= N_STAMPS - 4 * WEEK)]
    bad = (avoid | panel.excluded[row]).astype(np.int32)
    csum = np.concatenate([[0], np.cumsum(bad)])
    touches = csum[cand + window + 1] - csum[cand - window - 1] > 0
    cand = cand[~touches][:n_draws]
    if len(cand) == 0:
        return np.array([])
    offsets = np.arange(-window, window)
    stamps = (cand[:, None] + offsets[None, :]).ravel()
    exc = (panel.obs[row, stamps] - reference(panel=panel, row=row, stamps=stamps)).reshape(
        len(cand), -1
    )
    with np.errstate(all="ignore"):
        steps = np.nanmean(exc[:, window:], axis=1) - np.nanmean(exc[:, :window], axis=1)
    return steps[~np.isnan(steps)]


def percentile_of(*, value: float, null: np.ndarray) -> float:
    """Share of the null's absolute steps that are no larger than `abs(value)`."""
    if np.isnan(value) or len(null) < 30:
        return float("nan")
    return float(np.mean(np.abs(null) <= abs(value)))
