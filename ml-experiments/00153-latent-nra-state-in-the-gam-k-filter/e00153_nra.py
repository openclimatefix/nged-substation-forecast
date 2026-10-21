# ruff: noqa: ANN001, E501
"""The arms of experiment 153: filters that forecast the normal running arrangement (NRA).

(The subtraction arms are experiment 151's; experiment 153 adds the latent-NRA filter at the end.)

Every arm is GAM-K's cached member forecasts plus a correction from a Kalman filter run on a
stream. The arms differ in the stream and in the regime prior the filter sees:

- K0: the stream with the injected transfers in it, no correction.
- CLEAN: the stream without the injected megawatts (MW), the ceiling.
- S-KF, S-ORACLE: experiment 141's two-regime filter, fed detected or true intervals.
- SUB-DET: the stream with the detected MW subtracted over each interval known at the forecast
  origin. The filter is re-run once for each distinct set of known intervals.
- SUB-TRUE-LAG: the same with the true MW and interval, known after the detector's delays.
- SUB-PLACEBO: SUB-DET with the intervals shifted by whole weeks.

Every score is against the true NRA series, `data.actual`, without the injected MW.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import e00153_fed as fed
import e00153_g_arms as arms134
import e00153_kf as kf
import numpy as np
import polars as pl

ONSET_LAG: Final[int] = fed.ONSET_LAG
"""Half-hours from a step to the stamp its alarm is known (the 24-hour statistic and a peak test)."""
END_LAG: Final[int] = 60
"""Half-hours from the end of a transfer to the stamp its end is known: the call lag of the
6-hour lookahead rule of experiment 148 (median 60, 90th percentile 96)."""
TAIL: Final[int] = 96
"""48 hours of target time after the end of a transfer."""
WEEK: Final[int] = 336
ONSET_TOLERANCE: Final[int] = fed.ONSET_TOLERANCE


@dataclass(frozen=True)
class Span:
    """One interval whose MW is subtracted from the stream, in grid positions.

    Attributes:
        onset: First position the MW applies to.
        end: First position it no longer applies to (used once the end is known).
        start: First cutoff at which the interval is known.
        known: First cutoff at which the end is known.
        amount: The points to subtract: a constant, or an array over the whole grid.
    """

    onset: int
    end: int
    start: int
    known: int
    amount: float | np.ndarray


def detected_spans(
    *,
    data,
    site: str,
    intervals: pl.DataFrame,
    setting: str = "R0_M48",
    end_lag: int = END_LAG,
) -> list[Span]:
    """Detected spans of a site from the offline-end intervals, known as the 6-hour rule knows them.

    The onset is known `ONSET_LAG` after the step. The end is known at the later of the interval's
    stored `known_from` and `END_LAG` after the end. The megawatts are the stored `step_mw` of the
    detector on the site's series, a constant per interval.
    """
    offset = fed.stamp_offset(data)
    series = fed.SERIES_OF_SITE[site]
    spans = []
    for row in intervals.filter(
        (pl.col("setting") == setting) & (pl.col("series") == series)
    ).iter_rows(named=True):
        points = 100.0 * row["step_mw"] / fed.CAPACITY_MW[site]
        if not np.isfinite(points):
            continue
        onset, end = row["onset"] + offset, row["end"] + offset
        start = onset + ONSET_LAG
        known = max(row["known_from"] + offset, end + end_lag, start + 1)
        spans.append(Span(onset, end, start, known, points))
    return spans


def true_spans(*, events: list[tuple[int, int]], delta_points: np.ndarray, data) -> list[Span]:
    """True spans from the injected events, known after the same delays as a detection.

    `events` are (onset, end) in detector stamps; `delta_points` is over grid positions.
    """
    offset = fed.stamp_offset(data)
    spans = []
    for on, en in events:
        onset, end = on + offset, en + offset
        masked = np.zeros_like(delta_points)
        masked[onset:end] = delta_points[onset:end]
        spans.append(Span(onset, end, onset + ONSET_LAG, end + END_LAG, masked))
    return spans


def constant_true_spans(*, spans: list[Span], delta_points: np.ndarray) -> list[Span]:
    """True spans with the MW held at the event's mean instead of its daily shape."""
    out = []
    for s in spans:
        mean = float(np.nanmean(delta_points[s.onset : s.end]))
        out.append(Span(s.onset, s.end, s.start, s.known, mean))
    return out


def shifted(*, spans: list[Span], weeks: int, n: int) -> list[Span]:
    """Shift every span by whole weeks (a control), dropping those that leave the grid."""
    out = []
    for s in spans:
        d = weeks * WEEK
        if s.onset + d < 0 or s.end + d >= n or s.known + d >= n:
            continue
        out.append(Span(s.onset + d, s.end + d, s.start + d, s.known + d, s.amount))
    return out


def cleaned_stream(*, stream: np.ndarray, spans: list[Span], cutoff: int | None) -> np.ndarray:
    """The stream with every span known at `cutoff` subtracted, using the end where it is known."""
    out = stream.copy()
    if cutoff is None:
        return out
    for s in spans:
        if s.start > cutoff:
            continue
        stop = s.end if s.known <= cutoff else len(out)
        if isinstance(s.amount, np.ndarray):
            out[s.onset : stop] -= s.amount[s.onset : stop]
        else:
            out[s.onset : stop] -= s.amount
    return out


def subtraction_correction(
    *, data, stream: np.ndarray, spans: list[Span], cutoffs: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    """The total filter correction at `rows`, each from the stream cleaned as known at its cutoff.

    Rows share a filter run when no span changes status between their cutoffs, so the filter runs
    once per distinct status of the spans among the rows. The result is NaN elsewhere.
    """
    out = np.full(len(data.run_of_row), np.nan)
    breaks = np.sort(np.array([v for s in spans for v in (s.start, s.known)] or [0]))
    state = np.searchsorted(breaks, cutoffs[rows], side="right")
    for s in np.unique(state):
        chosen = rows[state == s]
        cutoff = int(breaks[s - 1]) if s > 0 else None
        correction = fed.correction_for(
            data=data,
            stream=cleaned_stream(stream=stream, spans=spans, cutoff=cutoff),
            prior=np.zeros(len(stream)),
            multiplier=1.0,
        )
        out[chosen] = correction[chosen]
    return out


def score_nra(
    *, data, site: str, corrections: dict[str, np.ndarray], rows: np.ndarray
) -> pl.DataFrame:
    """Score every arm on `rows` against the true NRA series: quantile score and errors."""
    actual = data.actual[rows]
    spread = fed.k0_spread(site)
    frames = []
    for arm, correction in corrections.items():
        q = fed.quantiles_at(data=data, correction=correction, spread=spread, rows=rows)
        mean_forecast = data.member_mean[rows] + correction[rows]
        frames.append(
            pl.DataFrame(
                {
                    "arm": arm,
                    "site": site,
                    "row": rows,
                    "position": data.position[rows],
                    "run": data.runs[data.run_of_row[rows]],
                    "lead_hours": data.lead_hours[rows],
                    "qs": arms134.quantile_score(actual=actual, quantiles=q),
                    "err_median": q[:, 5] - actual,
                    "err_mean": mean_forecast - actual,
                    "q01": q[:, 0],
                    "q05": q[:, 1],
                    "q95": q[:, 9],
                    "q99": q[:, 10],
                    "actual": actual,
                }
            )
        )
    return pl.concat(frames)


def oracle_spans(*, events: list[tuple[int, int]], data) -> list[Span]:
    """True spans known at once: the alarm at the onset, the end one day after the true end."""
    offset = fed.stamp_offset(data)
    return [Span(on + offset, en + offset, on + offset, en + offset + 48, 0.0) for on, en in events]


CAP: Final[int] = 14 * 48
"""Half-hours after its onset at which an interval's offset is folded into the NRA level."""


def open_mask(
    *, spans: list[Span], cutoff: int | None, n: int, cap: int | None = CAP
) -> tuple[np.ndarray, np.ndarray]:
    """Positions where the transfer offset is switched on, and where it is folded away.

    An interval known at `cutoff` switches the offset on from its onset (the filter is re-run
    from there) until its end is called, or to the cutoff if the end is not yet called. With a
    `cap`, an interval still open `cap` half-hours after its onset has its offset folded into the
    NRA level there, so a step that never returns is not held out of the level for ever.
    """
    mask = np.zeros(n, dtype=bool)
    fold = np.zeros(n, dtype=bool)
    if cutoff is None:
        return mask, fold
    for s in spans:
        if s.start > cutoff:
            continue
        stop = s.known if s.known <= cutoff else n
        if cap is not None and stop - s.onset > cap:
            stop = s.onset + cap
            if stop < n:
                fold[stop] = True
        mask[s.onset : stop] = True
    return mask, fold


def _by_status(*, spans: list[Span], cutoffs: np.ndarray, rows: np.ndarray, run) -> np.ndarray:
    """Run `run(cutoff)` once per distinct status of the spans among `rows`; keep each row's."""
    out = np.full(len(cutoffs), np.nan)
    breaks = np.sort(np.array([v for s in spans for v in (s.start, s.known)] or [0]))
    state = np.searchsorted(breaks, cutoffs[rows], side="right")
    for st in np.unique(state):
        chosen = rows[state == st]
        out[chosen] = run(int(breaks[st - 1]) if st > 0 else None)[chosen]
    return out


def latent_correction(
    *,
    data,
    stream: np.ndarray,
    spans: list[Span],
    cutoffs: np.ndarray,
    rows: np.ndarray,
    ratio: float,
    cap: int | None = CAP,
) -> np.ndarray:
    """The latent-NRA filter's correction at `rows`, one filter run per distinct interval status."""
    cutoff_of_row = data.cutoffs(delay_hours=fed.DELAY_HOURS)[data.run_of_row]

    def run(cutoff: int | None) -> np.ndarray:
        on, fold = open_mask(spans=spans, cutoff=cutoff, n=len(stream), cap=cap)
        trace = kf.run_latent_nra(
            residual=stream,
            band=data.cache.calendar.band.astype(np.int64),
            open_interval=on,
            reabsorb=fold,
            hyper=fed.hyper_of(data),
            ratio=ratio,
        )
        return fed.correction_at(
            trace=trace, cutoff=cutoff_of_row, target=data.position, band=data.hour_band
        ).total

    return _by_status(spans=spans, cutoffs=cutoffs, rows=rows, run=run)


def mask_correction(
    *, data, stream: np.ndarray, spans: list[Span], cutoffs: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    """K0 with the readings inside the known intervals missing, re-run from each onset."""

    def run(cutoff: int | None) -> np.ndarray:
        on, _ = open_mask(spans=spans, cutoff=cutoff, n=len(stream), cap=None)
        return fed.correction_for(
            data=data,
            stream=np.where(on, np.nan, stream),
            prior=np.zeros(len(stream)),
            multiplier=1.0,
        )

    return _by_status(spans=spans, cutoffs=cutoffs, rows=rows, run=run)


def true_end_spans(*, spans: list[Span], events: list[tuple[int, int]], data) -> list[Span]:
    """Detected spans whose end is known at the true end (the alarm keeps its detection delay).

    A detected span is matched to the injected event whose onset is within the tolerance;
    unmatched spans keep their causal end.
    """
    offset = fed.stamp_offset(data)
    out = []
    for s in spans:
        match = [en for on, en in events if abs(on + offset - s.onset) <= ONSET_TOLERANCE]
        if not match:
            out.append(s)
            continue
        end = match[0] + offset
        out.append(Span(s.onset, end, s.start, max(end + 1, s.start + 1), s.amount))
    return out
