"""The GAM-stage network's training rows and input matrix.

A training row is a half-hour of a site's residual stream: the convex model's forecast on the
weather it was driven by, and the residual it left. The rows come from two streams that agree
where they overlap in design. Before 2024-07 the stream is experiment 201's chained monthly fits on
the cache's own ERA5 and CAMS weather (`past_weather`). From 2024-04 it is experiment 136's arm A2
stream on the analysis proxy (`proxy`), which is the stream the Kalman filter reads. Where both
exist the proxy stream is used. The scored year (valid time from 2025-07-01) is never a training
row.

The input matrix has 24 columns: the convex model's forecast `g`, the driver temperature and its
square, three harmonics of the local half-hour of day, one harmonic of the weekday and a weekend
flag, two harmonics of the day of year, bank-holiday and Christmas flags, and a six-way site
indicator.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

import e00222_common as common
import e00222_stream as stream_module
import e00222_tall as tall
import numpy as np
import polars as pl
from studies.dp_calendar import bank_holiday_names

LONDON: Final[str] = "Europe/London"
SCORED_START: Final[np.datetime64] = np.datetime64("2025-07-01T00:00:00", "us")
LAST_EARLY_CUT: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
N_FOLDS: Final[int] = 4
N_INPUTS: Final[int] = 24
SITES: Final[tuple[str, ...]] = common.SCORED_SITES


def fold_of(*, times: np.ndarray) -> np.ndarray:
    """Return the cross-fitting fold of each time: the calendar month index modulo `N_FOLDS`."""
    return (times.astype("datetime64[M]").astype(np.int64) % N_FOLDS).astype(np.int64)


def design_matrix(
    *, g: np.ndarray, temp: np.ndarray, times: np.ndarray, site_index: int
) -> np.ndarray:
    """Return the network's input matrix, float64, shape (rows, `N_INPUTS`).

    Args:
        g: The convex model's forecast, in points.
        temp: The temperature the convex model was driven by, in degrees Celsius.
        times: The UTC valid times, `datetime64[us]`.
        site_index: The site's position in `SITES`.

    Returns:
        One row per time.
    """
    local = pl.Series(times).dt.replace_time_zone("UTC").dt.convert_time_zone(LONDON)
    hh = (local.dt.hour().cast(pl.Int32) * 2 + local.dt.minute().cast(pl.Int32) // 30).to_numpy()
    weekday = local.dt.weekday().to_numpy().astype(np.float64)
    doy = local.dt.ordinal_day().to_numpy().astype(np.float64)
    month, day = local.dt.month().to_numpy(), local.dt.day().to_numpy()
    dates = local.dt.date().to_list()
    holidays = bank_holiday_names()
    bank = np.array([d in holidays for d in dates], dtype=np.float64)
    christmas = (((month == 12) & (day >= 24)) | ((month == 1) & (day <= 1))).astype(np.float64)
    columns = [g, temp, temp**2]
    for harmonic in (1, 2, 3):
        angle = 2 * np.pi * harmonic * hh / 48.0
        columns += [np.sin(angle), np.cos(angle)]
    columns += [
        np.sin(2 * np.pi * weekday / 7.0),
        np.cos(2 * np.pi * weekday / 7.0),
        (weekday >= 5).astype(np.float64),
    ]
    for harmonic in (1, 2):
        angle = 2 * np.pi * harmonic * doy / 365.25
        columns += [np.sin(angle), np.cos(angle)]
    columns += [bank, christmas]
    sites = np.zeros((len(g), len(SITES)))
    sites[:, site_index] = 1.0
    out = np.column_stack([*columns, sites])
    if out.shape[1] != N_INPUTS:
        msg = f"expected {N_INPUTS} inputs, built {out.shape[1]}"
        raise ValueError(msg)
    return out


@dataclass(frozen=True)
class SiteRows:
    """One site's stream rows on the cache grid.

    Attributes:
        site: The site's label.
        stream: The training stream (proxy where it exists, else past weather), `NaN` where none.
        proxy_stream: Experiment 136's proxy stream only (what the filter reads), `NaN` where none.
        g: The convex model's forecast on the stream's weather, `NaN` where no stream.
        temp: The driver temperature, `NaN` where no stream.
        switching: Whether the half-hour lies in a hand-filtered switching event.
        times: The grid's UTC times.
    """

    site: str
    stream: np.ndarray
    proxy_stream: np.ndarray
    g: np.ndarray
    temp: np.ndarray
    switching: np.ndarray
    times: np.ndarray


def early_stream(*, site: str, cache) -> np.ndarray:  # noqa: ANN001
    """Return experiment 201's early past-weather residual stream on the cache grid.

    Fit `i` supplies the rows after its cut and up to the next cut (the last fit up to
    2024-07-01), so no residual is in-sample.
    """
    fits = tall.load_pickle(path=common.E201_WORK_DIR / "tall_fits" / f"{site}.pkl")
    from e00222_stream import residual_stream

    return residual_stream(cache=cache, fits=fits, site=site, last=LAST_EARLY_CUT)


def site_rows(*, site: str, site_stream: stream_module.SiteStream) -> SiteRows:
    """Return a site's stream rows."""
    cache = site_stream.cache
    early = early_stream(site=site, cache=cache)
    proxy = site_stream.stream
    use_proxy = np.isfinite(proxy)
    stream = np.where(use_proxy, proxy, early)
    y = cache.sites[site].y
    temp = np.where(
        use_proxy, site_stream.proxy.sites[site].temperature_c, cache.sites[site].temperature_c
    )
    ok = np.isfinite(stream) & np.isfinite(temp)
    return SiteRows(
        site=site,
        stream=np.where(ok, stream, np.nan),
        proxy_stream=proxy,
        g=np.where(ok, y - stream, np.nan),
        temp=np.where(ok, temp, np.nan),
        switching=cache.sites[site].switching_mask,
        times=cache.end_times,
    )
