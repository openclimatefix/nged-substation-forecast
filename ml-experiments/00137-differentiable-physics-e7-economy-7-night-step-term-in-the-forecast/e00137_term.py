"""The Economy 7 night term: night features from ENS-mean temperature, the ridge fit, and its use.

A night half-hour is one of 24 slots that end between 21:30 and 09:00 UTC. The term's columns are,
for each slot, a constant, `cold / 10`, and a weekend flag, where `cold = max(0, 15.5 - T)` and `T`
is the ENS-mean temperature over the 24 hours from 09:00 UTC on the evening date to 09:00 UTC on
the next day.
"""

from datetime import datetime, timedelta

import e00137_common as common
import numpy as np
import polars as pl

N_COLUMNS = common.N_NIGHT_SLOTS * 3
LAMBDA_GRID: tuple[float, ...] = (10.0, 100.0, 1000.0, 10000.0)
HALF_HOUR = np.timedelta64(30, "m")
MINUTE = pl.col("valid_time").dt.hour().cast(pl.Int32) * 60 + pl.col("valid_time").dt.minute().cast(
    pl.Int32
)
"""The UTC minute of the day at which each half-hour ends."""


def minute_of_day(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the UTC minute of the day at which each row's half-hour ends."""
    return frame.select(MINUTE).to_series().to_numpy().astype(np.int64)


def evening_temperature(*, site: str, frame: pl.DataFrame) -> np.ndarray:
    """Return `T` for each row's night, NaN outside the night slots or past a run's coverage."""
    origins, temperature = common.ens_mean_temperature(site=site)
    valid = frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    minute = minute_of_day(frame=frame)
    shifted = valid - np.timedelta64(21 * 60 + 30, "m")
    evening = shifted.astype("datetime64[D]")
    runs = frame["run"].to_numpy()
    out = np.full(len(frame), np.nan)
    in_night = (minute >= common.NIGHT_START_MINUTE) | (minute <= 9 * 60)
    cache: dict[tuple[int, np.datetime64], float] = {}
    for row in np.flatnonzero(in_night):
        key = (int(runs[row]), evening[row])
        if key not in cache:
            origin = origins[int(runs[row])]
            first = evening[row].astype("datetime64[us]") + np.timedelta64(9 * 60 + 30, "m")
            start = int((first - origin) // HALF_HOUR) - 1
            stop = start + 48
            window = temperature[key[0], max(start, 0) : stop]
            cache[key] = (
                float(window.mean()) if start >= 0 and stop <= temperature.shape[1] else np.nan
            )
        out[row] = cache[key]
    return out


def add_night_columns(*, site: str, frame: pl.DataFrame) -> pl.DataFrame:
    """Add `slot` (0 to 23, -1 outside the night), `evening`, `temp_c`, `cold`, and `weekend`."""
    minute = minute_of_day(frame=frame)
    slot = ((minute - common.NIGHT_START_MINUTE) % 1440) // 30
    slot = np.where(slot < common.N_NIGHT_SLOTS, slot, -1)
    temp = evening_temperature(site=site, frame=frame)
    evening = (frame["valid_time"] - timedelta(hours=21, minutes=30)).dt.date()
    return frame.with_columns(
        slot=pl.Series(slot.astype(np.int16)),
        evening=evening,
        temp_c=pl.Series(temp),
    ).with_columns(
        cold=(common.HEATING_THRESHOLD_C - pl.col("temp_c")).clip(lower_bound=0.0),
        weekend=(pl.col("evening").dt.weekday() >= 6).cast(pl.Float64),
        window=(pl.col("slot") >= 0) & (MINUTE.is_in(list(common.WINDOW_ENDS_MINUTES))),
    )


def design(*, frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return the design matrix of the night rows with finite `T`, and the mask of those rows."""
    slot = frame["slot"].to_numpy()
    cold = frame["cold"].to_numpy()
    weekend = frame["weekend"].to_numpy()
    keep = (slot >= 0) & np.isfinite(cold)
    x = np.zeros((len(frame), N_COLUMNS))
    rows = np.flatnonzero(keep)
    s = slot[rows]
    x[rows, 3 * s] = 1.0
    x[rows, 3 * s + 1] = cold[rows] / 10.0
    x[rows, 3 * s + 2] = weekend[rows]
    return x[keep], keep


def fit_ridge(*, frames: list[pl.DataFrame], ridge: float) -> np.ndarray:
    """Return the ridge coefficients for the error `actual - mean` on the night rows of `frames`."""
    xtx = np.zeros((N_COLUMNS, N_COLUMNS))
    xty = np.zeros(N_COLUMNS)
    for frame in frames:
        x, keep = design(frame=frame)
        y = (frame["actual"] - frame["mean"]).to_numpy().astype(np.float64)[keep]
        xtx += x.T @ x
        xty += x.T @ y
    return np.linalg.solve(xtx + ridge * np.eye(N_COLUMNS), xty)


def correction(*, frame: pl.DataFrame, beta: np.ndarray) -> np.ndarray:
    """Return the correction per row: `x @ beta` on night rows with finite `T`, else 0."""
    x, keep = design(frame=frame)
    out = np.zeros(len(frame))
    out[keep] = x @ beta
    return out


def shift_quantiles(*, frame: pl.DataFrame, shift: np.ndarray) -> pl.DataFrame:
    """Return `frame` with every quantile and the mean moved by `shift`."""
    return frame.with_columns(
        [(pl.col(c) + pl.Series(shift)).cast(pl.Float32) for c in (*common.QUANTILES, "mean")]
    )


def month_starts(*, frame: pl.DataFrame) -> list:
    """Return the first-of-month (UTC) of every origin month in `frame`, ascending."""
    return sorted(frame["power_fcst_init_time"].dt.truncate("1mo").unique().to_list())


def training_rows(*, frame: pl.DataFrame, cut: datetime) -> pl.DataFrame:
    """Return the rows whose run origin and valid time are both before `cut`."""
    return frame.filter((pl.col("power_fcst_init_time") < cut) & (pl.col("valid_time") < cut))


def monthly_fit_with_betas(
    *, groups: dict[str, pl.DataFrame], ridge: float, pooled: bool
) -> tuple[dict[str, pl.DataFrame], dict[tuple[str, datetime], np.ndarray]]:
    """Refit monthly on the rows before each cut and correct that month, per site or pooled.

    Args:
        groups: Each site's night-annotated base rows.
        ridge: The ridge penalty.
        pooled: Share one set of coefficients across the sites.

    Returns:
        Each site's corrected rows, and the coefficients of each (site or "pool", cut).
    """
    months = sorted({m for f in groups.values() for m in month_starts(frame=f)})
    corrected: dict[str, list[pl.DataFrame]] = {s: [] for s in groups}
    betas: dict[tuple[str, datetime], np.ndarray] = {}
    for cut in months:
        fits = {}
        if pooled:
            train = [training_rows(frame=f, cut=cut) for f in groups.values()]
            if sum(t.height for t in train):
                fits = {s: fit_ridge(frames=train, ridge=ridge) for s in groups}
                betas[("pool", cut)] = fits[next(iter(groups))]
        else:
            for site, f in groups.items():
                train = training_rows(frame=f, cut=cut)
                if train.height:
                    fits[site] = fit_ridge(frames=[train], ridge=ridge)
                    betas[(site, cut)] = fits[site]
        for site, f in groups.items():
            month = f.filter(pl.col("power_fcst_init_time").dt.truncate("1mo") == cut)
            if month.is_empty():
                continue
            shift = (
                correction(frame=month, beta=fits[site]) if site in fits else np.zeros(month.height)
            )
            corrected[site].append(shift_quantiles(frame=month, shift=shift))
    result = {s: pl.concat(p) for s, p in corrected.items()}
    return result, betas


def monthly_fit_forecast(
    *, groups: dict[str, pl.DataFrame], ridge: float, pooled: bool
) -> dict[str, pl.DataFrame]:
    """Return each site's rows corrected by the monthly refits of `monthly_fit_with_betas`."""
    return monthly_fit_with_betas(groups=groups, ridge=ridge, pooled=pooled)[0]
