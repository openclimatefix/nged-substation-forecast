"""Daily effective-capacity feature tables for experiment 00026.

A series' feature table has one row per cutoff day. The row for cutoff day `c` is computed from the
whole UTC days that end before day `c` starts, so it uses nothing at or after the start of day `c`.
A forecast origin maps to the cutoff day that is `CUTOFF_DAYS` days before the origin's UTC day, so
a feature never reads power or weather less than 5 days old (ERA5 arrives about 5 days late).
Training and serving rows both go through `cutoff_rows`; `check_no_lookahead` proves the table
ignores everything after the cutoff.
"""

from datetime import UTC, datetime
from typing import Final

import numpy as np
import polars as pl

FEATURE_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
FEATURE_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
STEPS_PER_DAY: Final[int] = 48
N_DAYS: Final[int] = (FEATURE_END - FEATURE_START).days
N_STEPS: Final[int] = N_DAYS * STEPS_PER_DAY
CUTOFF_DAYS: Final[int] = 5
"""A feature reads data at least this many whole days older than the forecast origin's UTC day."""
LAST_YEAR_DAYS: Final[int] = 365
LAST_YEAR_HALF_WIDTH_DAYS: Final[int] = 14
MIN_TOP_STEPS: Final[int] = 10
MIN_OBSERVED_SHARE: Final[float] = 0.5
MAX_ZENITH_DEGREES: Final[float] = 85.0
CAMS_FLOOR_W_M2: Final[float] = 200.0
WIND_CUBE_RANGE_M_S: Final[tuple[float, float]] = (3.0, 8.0)
TRAINING_LEAD_HOURS: Final[np.ndarray] = np.arange(0, 361, 3)
"""The leads a training row's origin is drawn from, uniformly."""

FEATURE_NAMES: Final[dict[str, tuple[str, ...]]] = {
    "solar": (
        "top_decile_28d",
        "top_decile_7d",
        "top_decile_90d",
        "power_to_cams_q90_28d",
        "top_decile_last_year",
    ),
    "wind": (
        "top_decile_28d",
        "p995_28d",
        "p995_90d",
        "power_to_wind3_q75_28d",
        "p995_last_year",
    ),
}
ARM_FEATURES: Final[dict[str, tuple[int, ...]]] = {
    "a0": (),
    "a2": (0,),
    "a3": (0, 1, 2),
    "a4": (0, 3),
    "a5": (0, 4),
    "a6": (0, 1, 2),
}
"""Each arm's columns as indices into `FEATURE_NAMES`; `a6` reads them from the shuffled table."""


def day_index(*, times: np.ndarray) -> np.ndarray:
    """Return the index of the UTC day containing each `datetime64` time, from `FEATURE_START`."""
    start = np.datetime64(FEATURE_START.replace(tzinfo=None), "us")
    return np.floor((times.astype("datetime64[us]") - start) / np.timedelta64(1, "D")).astype(int)


def _window(*, power: np.ndarray, other: np.ndarray, zenith: np.ndarray, c: int, days: int):  # noqa: ANN202
    """Return the arrays of the `days` whole days before cutoff day `c`, or None before the grid."""
    end = c * STEPS_PER_DAY
    start = end - days * STEPS_PER_DAY
    if start < 0:
        return None
    return power[start:end], other[start:end], zenith[start:end]


def _centred(*, power: np.ndarray, other: np.ndarray, zenith: np.ndarray, c: int):  # noqa: ANN202
    """Return the 28-day window centred `LAST_YEAR_DAYS` before cutoff day `c`, or None."""
    centre = c - LAST_YEAR_DAYS
    start = (centre - LAST_YEAR_HALF_WIDTH_DAYS) * STEPS_PER_DAY
    end = (centre + LAST_YEAR_HALF_WIDTH_DAYS) * STEPS_PER_DAY
    if start < 0:
        return None
    return power[start:end], other[start:end], zenith[start:end]


def _top_decile_p99(*, power: np.ndarray, rank: np.ndarray, keep: np.ndarray) -> float:
    """Return the 99th percentile of power at the steps in the top 10% of `rank`, or NaN."""
    valid = keep & np.isfinite(power) & np.isfinite(rank)
    if valid.sum() < 10 * MIN_TOP_STEPS:
        return np.nan
    threshold = np.quantile(rank[valid], 0.9)
    chosen = valid & (rank >= threshold)
    return float(np.quantile(power[chosen], 0.99)) if chosen.sum() >= MIN_TOP_STEPS else np.nan


def _p995(*, power: np.ndarray) -> float:
    """Return the 99.5th percentile of the observed power, or NaN if under half is observed."""
    seen = power[np.isfinite(power)]
    return float(np.quantile(seen, 0.995)) if seen.size >= MIN_OBSERVED_SHARE * power.size else np.nan


def _ratio_quantile(*, ratio: np.ndarray, keep: np.ndarray, level: float) -> float:
    valid = keep & np.isfinite(ratio)
    return float(np.quantile(ratio[valid], level)) if valid.sum() >= MIN_TOP_STEPS else np.nan


def _solar_row(*, power: np.ndarray, cams: np.ndarray, zenith: np.ndarray, c: int) -> list[float]:
    out = []
    for days in (28, 7, 90):
        window = _window(power=power, other=cams, zenith=zenith, c=c, days=days)
        out.append(np.nan if window is None else _solar_top(window))
    window = _window(power=power, other=cams, zenith=zenith, c=c, days=28)
    if window is None:
        out.append(np.nan)
    else:
        p, w, _ = window
        with np.errstate(divide="ignore", invalid="ignore"):
            out.append(_ratio_quantile(ratio=p / (w / 1000.0), keep=w > CAMS_FLOOR_W_M2, level=0.9))
    window = _centred(power=power, other=cams, zenith=zenith, c=c)
    out.append(np.nan if window is None else _solar_top(window))
    return out


def _solar_top(window: tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
    p, w, z = window
    return _top_decile_p99(power=p, rank=w, keep=z < MAX_ZENITH_DEGREES)


def _wind_row(*, power: np.ndarray, speed: np.ndarray, zenith: np.ndarray, c: int) -> list[float]:
    low, high = WIND_CUBE_RANGE_M_S
    window = _window(power=power, other=speed, zenith=zenith, c=c, days=28)
    out = []
    if window is None:
        out += [np.nan, np.nan, np.nan]
    else:
        p, w, _ = window
        out += [_top_decile_p99(power=p, rank=w, keep=np.ones(p.size, dtype=bool)), _p995(power=p)]
        window90 = _window(power=power, other=speed, zenith=zenith, c=c, days=90)
        out.append(np.nan if window90 is None else _p995(power=window90[0]))
    if window is None:
        out.append(np.nan)
    else:
        p, w, _ = window
        with np.errstate(divide="ignore", invalid="ignore"):
            out.append(_ratio_quantile(ratio=p / w**3, keep=(w >= low) & (w <= high), level=0.75))
    last_year = _centred(power=power, other=speed, zenith=zenith, c=c)
    out.append(np.nan if last_year is None else _p995(power=last_year[0]))
    return out


def daily_table(
    *, technology: str, power: np.ndarray, other: np.ndarray, zenith: np.ndarray
) -> np.ndarray:
    """Return the feature table of one series, shape (`N_DAYS`, 5), float32.

    Args:
        technology: `solar` or `wind`.
        power: Cleaned power over effective capacity on the half-hourly grid from `FEATURE_START`,
            NaN where unusable.
        other: CAMS shortwave (W/m2, solar) or ERA5 100 m wind speed (m/s, wind) on the same grid.
        zenith: Solar zenith in degrees on the same grid (used by solar only).

    Returns:
        Row `c` holds `FEATURE_NAMES[technology]` for cutoff day `c`.
    """
    row = _solar_row if technology == "solar" else _wind_row
    key = "cams" if technology == "solar" else "speed"
    return np.array(
        [row(power=power, **{key: other}, zenith=zenith, c=c) for c in range(N_DAYS)],
        dtype=np.float32,
    )


def check_no_lookahead(*, technology: str, power, other, zenith, cutoff: int) -> None:  # noqa: ANN001
    """Raise if the table row for any cutoff day up to `cutoff` changes when later data is removed.

    Raises:
        ValueError: If a row up to `cutoff` depends on data at or after day `cutoff`.
    """
    full = daily_table(technology=technology, power=power, other=other, zenith=zenith)
    cut = cutoff * STEPS_PER_DAY
    kept = [a.copy() for a in (power, other, zenith)]
    for a in kept:
        a[cut:] = np.nan
    truncated = daily_table(technology=technology, power=kept[0], other=kept[1], zenith=kept[2])
    if not np.array_equal(full[: cutoff + 1], truncated[: cutoff + 1], equal_nan=True):
        msg = "a feature row depends on data at or after its cutoff day"
        raise ValueError(msg)


def cutoff_rows(*, table: np.ndarray, origins: np.ndarray) -> np.ndarray:
    """Return the feature row for each forecast origin, NaN where the cutoff is off the table.

    Args:
        table: A series' (possibly shuffled) feature table.
        origins: `datetime64` forecast origins, one per row.

    Returns:
        Shape (rows, table width) float32.
    """
    cutoff = day_index(times=origins) - CUTOFF_DAYS
    inside = (cutoff >= 0) & (cutoff < table.shape[0])
    out = np.full((origins.size, table.shape[1]), np.nan, dtype=np.float32)
    out[inside] = table[cutoff[inside]]
    return out


def training_origins(*, valid: np.ndarray, seed: int) -> np.ndarray:
    """Return an origin for each training row: its valid time minus a seeded uniform lead."""
    leads = np.random.default_rng(seed).choice(TRAINING_LEAD_HOURS, size=valid.size)
    return valid.astype("datetime64[us]") - leads.astype("timedelta64[h]").astype("timedelta64[us]")


def shuffled_table(*, table: np.ndarray, seed: int, last_source_day: int) -> np.ndarray:
    """Return the table with whole calendar months permuted among the months before the end.

    Every day takes the row of the day at the same day of month in its month's replacement month.
    Replacement months come only from days before `last_source_day`, so no day after it feeds a
    training row. Days in months after the last source month take a random source month.

    Args:
        table: A series' feature table.
        seed: Seed of the permutation.
        last_source_day: Index of the first day that may not be a source.

    Returns:
        A table of the same shape.
    """
    days = np.datetime64(FEATURE_START.replace(tzinfo=None), "D") + np.arange(table.shape[0])
    months = days.astype("datetime64[M]")
    starts = np.flatnonzero(np.r_[True, months[1:] != months[:-1]])
    lengths = np.diff(np.r_[starts, table.shape[0]])
    usable = np.flatnonzero(starts + lengths <= last_source_day)
    generator = np.random.default_rng(seed)
    replacement = np.empty(starts.size, dtype=int)
    replacement[usable] = generator.permutation(usable)
    others = np.setdiff1d(np.arange(starts.size), usable)
    replacement[others] = generator.choice(usable, size=others.size)
    out = np.empty_like(table)
    for m in range(starts.size):
        source = replacement[m]
        offset = np.minimum(np.arange(lengths[m]), lengths[source] - 1)
        out[starts[m] : starts[m] + lengths[m]] = table[starts[source] + offset]
    return out


def series_grids(*, analysis: pl.DataFrame, series: int, technology: str):  # noqa: ANN201
    """Return the weather and zenith of one series on the half-hourly grid from `FEATURE_START`.

    Returns:
        `other` (CAMS shortwave or ERA5 100 m wind speed) and `zenith`, float32 arrays of
        `N_STEPS`, NaN where the product has no value.
    """
    column = "shortwave_cams" if technology == "solar" else "wind_speed_100m"
    part = analysis.filter(pl.col("time_series_id") == series).select(
        "valid_time", column, "solar_zenith"
    )
    index = day_index(times=part["valid_time"].to_numpy()) * STEPS_PER_DAY
    half = (part["valid_time"].dt.minute().to_numpy() // 30).astype(int)
    position = index + half
    inside = (position >= 0) & (position < N_STEPS)
    other = np.full(N_STEPS, np.nan, dtype=np.float32)
    zenith = np.full(N_STEPS, np.nan, dtype=np.float32)
    other[position[inside]] = part[column].to_numpy()[inside]
    zenith[position[inside]] = part["solar_zenith"].to_numpy()[inside]
    return other, zenith
