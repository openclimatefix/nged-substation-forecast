"""A past-weather forecast harness for the 15 Primary series: GAM forecast plus Kalman filter.

**Each row is one forecast of one half-hour, made at an origin every 6 hours, with the filter
allowed only the readings that ended at least 3.5 hours before the origin.** The location of the
forecast is the frozen convex GAM forecast of experiment 130 (past weather: ERA5 and CAMS, refitted
on the 1st of each month) plus the correction that the Kalman filter of
`studies.dp_assimilation` holds for the half-hour's hour band at the origin's data cutoff. The
quantiles of every arm are that location plus one fixed offset table per series, hour band, and
lead band, fitted on the no-mask forecast's residuals in odd months, so a contrast between arms is
a difference in location only.

A mask arm sets the readings inside every alarm span set aside, as experiment 227 does: a span
opens at the step stamp and applies once its onset is known (`start`), reaches to the end of the
grid while its end is unknown, and covers only `[onset, end)` once the end is known (`known`).
Rows whose cutoffs sit between the same two span-status changes share one filter run.

Nothing here reads a score or tunes a setting.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00308_data as dat
import numpy as np
import polars as pl
from studies.dp_assimilation import (
    CorrectionTrace,
    FilterHyperparameters,
    correction_at,
    run_kalman,
)
from studies.dp_calendar import LOCAL_TIME_ZONE, band_of_hour

GAM_DIR: Final[Path] = dat.DATA / "00130-ro-nc-detector-on-the-gam-k-residual" / "gam"
SERIES_PATH: Final[Path] = (
    dat.DATA.parent
    / "studies"
    / "per_study"
    / "stage1_pilot"
    / "demand"
    / "inputs"
    / "series.parquet"
)
HARNESS_DIR: Final[Path] = (
    dat.DATA / "00258-lsc-mask-out-of-sample-all-primaries-whole-history" / "harness"
)
"""Per-series correction arrays, offset tables, and the README; never committed."""
SERIES: Final[tuple[int, ...]] = (3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 24, 25, 26)
"""The 15 Primary series with a stored filter (series 12 has no readings)."""
ORIGIN_STEP: Final[int] = 12
"""Half-hours between origins: every 6 hours."""
DELAY_HALF_HOURS: Final[int] = 7
"""Telemetry delay: a half-hour is usable when it ends at least 3.5 hours before the origin."""
N_LEADS: Final[int] = 336
"""Targets per origin: leads 0 to 7 days in half-hours."""
FIRST_ORIGIN: Final[int] = 1488
"""The first origin, 31 days after the grid starts, so the filter has settled."""
LEAD_BAND_UPPER: Final[tuple[int, ...]] = (48, 144, N_LEADS)
"""Upper bounds of the lead bands in half-hours: under 24 hours, under 72 hours, up to 7 days."""
END_LAG: Final[int] = 60
"""Half-hours from the end of a load shift to the stamp its end is known (experiment 148's rule)."""
LEVELS: Final[np.ndarray] = np.array(
    [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
)
TUNING_MONTHS: Final[frozenset[int]] = frozenset({1, 3, 5, 7, 9, 11})
"""Odd calendar months, in which thresholds, hyperparameters, and offsets are fitted."""
N_GRID: Final[int] = dat.GRID_LENGTH


@dataclass(frozen=True)
class Span:
    """One interval whose readings the filter sets aside, in grid positions.

    Attributes:
        onset: First position set aside (the step stamp).
        end: First position no longer set aside, used once the end is known.
        start: First cutoff at which the span is known.
        known: First cutoff at which the end is known.
    """

    onset: int
    end: int
    start: int
    known: int


@dataclass(frozen=True)
class SeriesData:
    """One series on the half-hour grid, in points of effective capacity.

    Attributes:
        series: The series id.
        capacity_mw: The effective capacity.
        actual: The reading, NaN where missing.
        expected: The GAM forecast.
        hyper: The filter's hyperparameters.
    """

    series: int
    capacity_mw: float
    actual: np.ndarray
    expected: np.ndarray
    hyper: FilterHyperparameters

    @property
    def residual(self) -> np.ndarray:
        """Reading minus GAM forecast, NaN where there is no reading."""
        return self.actual - self.expected


def band_grid() -> np.ndarray:
    """The hour band of every grid half-hour, from the local hour at its start."""
    index = pl.datetime_range(
        dat.GRID_START, dat.GRID_END, interval="30m", closed="left", time_zone="UTC", eager=True
    )
    hour = index.dt.convert_time_zone(LOCAL_TIME_ZONE).dt.hour().to_numpy()
    return band_of_hour(hour=hour).astype(np.int64)


def utc_month_grid() -> np.ndarray:
    """The UTC calendar month (1 to 12) of every grid half-hour."""
    index = pl.datetime_range(
        dat.GRID_START, dat.GRID_END, interval="30m", closed="left", time_zone="UTC", eager=True
    )
    return index.dt.month().to_numpy()


def capacities() -> dict[int, float]:
    """The effective capacity in MW of each series, from the stage 1 pilot's series table."""
    table = pl.read_parquet(SERIES_PATH)
    return {
        int(r["time_series_id"]): float(r["effective_capacity_mw"])
        for r in table.iter_rows(named=True)
    }


def load_series(*, series: int) -> SeriesData:
    """Load one series' power, GAM forecast, and filter hyperparameters, all in points."""
    with (GAM_DIR / "panel_1.pkl").open("rb") as handle:
        frozen = pickle.load(handle)["frozen"]
    raw = json.loads((GAM_DIR / "hyper.json").read_text())
    cap = capacities()[series]
    scale = 100.0 / cap
    return SeriesData(
        series=series,
        capacity_mw=cap,
        actual=np.asarray(frozen["power"][series], dtype=np.float64) * scale,
        expected=np.asarray(frozen["expected"][series], dtype=np.float64) * scale,
        hyper=FilterHyperparameters.from_array(values=np.array(raw[str(series)])),
    )


def origin_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The origins, their data cutoffs, and the target of every (origin, lead) pair.

    Returns:
        The origin positions, the newest usable grid position of each origin, and the targets
        (origins by `N_LEADS`), which exceed the grid for the last origins.
    """
    origins = np.arange(FIRST_ORIGIN, N_GRID, ORIGIN_STEP)
    cutoffs = origins - DELAY_HALF_HOURS - 1
    targets = origins[:, None] + np.arange(N_LEADS)[None, :]
    return origins, cutoffs, targets


def lead_band_of_lead() -> np.ndarray:
    """The lead band (0 to 2) of each of the `N_LEADS` leads."""
    return np.searchsorted(np.array(LEAD_BAND_UPPER), np.arange(N_LEADS), side="right")


def spans_from_table(*, table: pl.DataFrame, series: int) -> list[Span]:
    """Mask spans of one series from an alarm interval table (experiment 227's rule).

    The span opens at the step stamp, is known from the table's `onset_known`, and its end is
    known at the later of the table's `known_from`, the end plus `END_LAG`, and one stamp after
    the onset is known.
    """
    out = []
    for row in table.filter(pl.col("series") == series).iter_rows(named=True):
        onset, end = int(row["onset"]), int(row["end"])
        start = int(row["onset_known"])
        known = max(int(row["known_from"]), end + END_LAG, start + 1)
        out.append(Span(onset, end, start, known))
    return out


def oracle_spans(*, events: pl.DataFrame, series: int) -> list[Span]:
    """Spans of the logged load shifts that involve the series: known at once, end known at once."""
    return [
        Span(int(r["onset"]), int(r["end"]), int(r["onset"]), int(r["end"]))
        for r in events.iter_rows(named=True)
        if series in r["series"]
    ]


def labelled_events() -> pl.DataFrame:
    """All 66 labelled real load shifts: card_id, onset, end, hours, pair, series (a list)."""
    return (
        pl.read_parquet(dat.REAL_RESULTS)
        .filter(pl.col("setting") == "XL")
        .select("card_id", "onset", "end", "hours", pair=pl.col("series"))
        .with_columns(series=pl.col("pair").str.split(",").list.eval(pl.element().cast(pl.Int64)))
    )


def cleaned_stream(*, stream: np.ndarray, spans: list[Span], cutoff: int) -> np.ndarray:
    """The stream with the readings inside every span known at `cutoff` set to NaN."""
    out = stream.copy()
    for s in spans:
        if s.start > cutoff:
            continue
        stop = s.end if s.known <= cutoff else len(out)
        out[s.onset : stop] = np.nan
    return out


def _rows_of(
    *,
    trace: CorrectionTrace,
    cutoffs: np.ndarray,
    targets: np.ndarray,
    band: np.ndarray,
) -> np.ndarray:
    """The correction of every (origin, lead) pair of the origins given, shape (origins, leads)."""
    shape = targets.shape
    flat_target = np.minimum(targets.ravel(), N_GRID - 1)
    flat_cutoff = np.repeat(cutoffs, shape[1])
    total = correction_at(
        trace=trace, cutoff=flat_cutoff, target=flat_target, band=band[flat_target]
    )
    return total.total.reshape(shape).astype(np.float32)


def k0_trace(*, data: SeriesData, band: np.ndarray) -> CorrectionTrace:
    """The no-mask filter run over the whole record."""
    return run_kalman(residual=data.residual, band=band, hyper=data.hyper)


def mask_correction(
    *, data: SeriesData, spans: list[Span], band: np.ndarray, k0: CorrectionTrace
) -> np.ndarray:
    """The mask arm's correction at every (origin, lead) pair, shape (origins, leads).

    Origins whose cutoffs sit between the same two span-status changes share one filter run, run
    over the prefix of the record up to the newest cutoff among them.
    """
    _, cutoffs, targets = origin_grid()
    breaks = np.sort(np.array([v for s in spans for v in (s.start, s.known)], dtype=np.int64))
    state = (
        np.searchsorted(breaks, cutoffs, side="right")
        if len(breaks)
        else np.zeros(len(cutoffs), int)
    )
    out = np.zeros(targets.shape, dtype=np.float32)
    stream = data.residual
    for status in np.unique(state):
        chosen = np.flatnonzero(state == status)
        if status == 0:
            trace = k0
        else:
            cleaned = cleaned_stream(stream=stream, spans=spans, cutoff=int(breaks[status - 1]))
            stop = int(cutoffs[chosen].max()) + 1
            trace = run_kalman(residual=cleaned[:stop], band=band[:stop], hyper=data.hyper)
        out[chosen] = _rows_of(
            trace=trace, cutoffs=cutoffs[chosen], targets=targets[chosen], band=band
        )
    return out


def k0_correction(*, trace: CorrectionTrace, band: np.ndarray) -> np.ndarray:
    """The no-mask correction at every (origin, lead) pair."""
    _, cutoffs, targets = origin_grid()
    return _rows_of(trace=trace, cutoffs=cutoffs, targets=targets, band=band)


def fit_offsets(*, data: SeriesData, k0: np.ndarray, band: np.ndarray) -> np.ndarray:
    """The 11 quantile offsets of the no-mask residual per hour band and lead band, odd months only.

    Args:
        data: The series' readings and GAM forecast.
        k0: The no-mask correction at every (origin, lead) pair.
        band: The hour band of every grid half-hour.

    Returns:
        An array of shape (4, 3, 11): the residual's quantiles at `LEVELS`, in points.
    """
    _, _, targets = origin_grid()
    valid = targets < N_GRID
    clipped = np.minimum(targets, N_GRID - 1)
    residual = data.actual[clipped] - data.expected[clipped] - k0
    month = utc_month_grid()[clipped]
    odd = np.isin(month, list(TUNING_MONTHS)) & valid & np.isfinite(residual)
    band_of_target = band[clipped]
    lead_band = np.broadcast_to(lead_band_of_lead()[None, :], targets.shape)
    table = np.full((4, len(LEAD_BAND_UPPER), len(LEVELS)), np.nan)
    for b in range(4):
        for lb in range(len(LEAD_BAND_UPPER)):
            chosen = odd & (band_of_target == b) & (lead_band == lb)
            if chosen.sum() > 100:
                table[b, lb] = np.quantile(residual[chosen], LEVELS)
    return table
