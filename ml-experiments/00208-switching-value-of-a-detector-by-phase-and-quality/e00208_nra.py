"""The arms of experiment 151: filters that forecast the normal running arrangement (NRA).

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
from typing import Any, Final

import e00208_fed as fed
import e00208_g_arms as arms134
import numpy as np
import polars as pl

ONSET_LAG: Final[int] = fed.ONSET_LAG
"""Half-hours from a step to the stamp its alarm is known (24-hour statistic, peak test)."""
END_LAG: Final[int] = 60
"""Half-hours from the end of a transfer to the stamp its end is known: the call lag of the
6-hour lookahead rule of experiment 148 (median 60, 90th percentile 96)."""
TAIL: Final[int] = 144
"""72 hours of target time after the end of a transfer."""
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
        open_amount: The array used while the end is not yet known: the amount held at its value
            when the interval becomes known, so the true end is not used before it is known.
        weight: The alarm's reliability weight in [0, 1], which a soft mask turns into an
            inflation of the observation noise (1 when the alarm has no reliability score).
    """

    onset: int
    end: int
    start: int
    known: int
    amount: float | np.ndarray
    open_amount: np.ndarray | None = None
    weight: float = 1.0


def detected_spans(
    *, data: Any, site: str, intervals: pl.DataFrame, end_lag: int = END_LAG
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
        (pl.col("setting") == "R0_M48") & (pl.col("series") == series)
    ).iter_rows(named=True):
        points = 100.0 * row["step_mw"] / fed.CAPACITY_MW[site]
        if not np.isfinite(points):
            continue
        onset, end = row["onset"] + offset, row["end"] + offset
        stored = row.get("onset_known")
        start = onset + ONSET_LAG if stored is None else stored + offset
        known = max(row["known_from"] + offset, end + end_lag, start + 1)
        weight = row.get("reliability_weight")
        spans.append(
            Span(onset, end, start, known, points, weight=1.0 if weight is None else weight)
        )
    return spans


def true_spans(*, events: list[tuple[int, int]], delta_points: np.ndarray, data: Any) -> list[Span]:
    """True spans from the injected events, known after the same delays as a detection.

    `events` are (onset, end) in detector stamps; `delta_points` is over grid positions.
    """
    offset = fed.stamp_offset(data)
    spans = []
    for on, en in events:
        onset, end = on + offset, en + offset
        masked = np.zeros_like(delta_points)
        masked[onset:end] = delta_points[onset:end]
        start = onset + ONSET_LAG
        held = masked.copy()
        held[end:] = masked[min(start, end - 1)]
        spans.append(Span(onset, end, start, end + END_LAG, masked, held))
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
        out.append(Span(s.onset + d, s.end + d, s.start + d, s.known + d, s.amount, s.open_amount))
    return out


def cleaned_stream(
    *, stream: np.ndarray, spans: list[Span], cutoff: int | None, mask: bool = False
) -> np.ndarray:
    """The stream with every span known at `cutoff` removed, using the end where it is known.

    By default the span's MW is subtracted. With `mask` the readings inside the span are set to
    missing instead, so the filter treats them as absent.
    """
    out = stream.copy()
    if cutoff is None:
        return out
    for s in spans:
        if s.start > cutoff:
            continue
        ended = s.known <= cutoff
        stop = s.end if ended else len(out)
        if mask:
            out[s.onset : stop] = np.nan
            continue
        amount = s.amount if (ended or s.open_amount is None) else s.open_amount
        if isinstance(amount, np.ndarray):
            out[s.onset : stop] -= amount[s.onset : stop]
        else:
            out[s.onset : stop] -= amount
    return out


def noise_scale_of(*, spans: list[Span], cutoff: int | None, factor: float, n: int) -> np.ndarray:
    """The observation-noise inflation of every grid position, as known at `cutoff`.

    A span known at `cutoff` inflates the noise variance of the readings from its onset to its end
    (or to the end of the grid while the end is not yet known) by `1 + (factor - 1) * weight`, and
    where spans overlap the larger inflation applies. Outside every known span the factor is 1.
    """
    out = np.ones(n)
    if cutoff is None:
        return out
    for s in spans:
        if s.start > cutoff:
            continue
        stop = s.end if s.known <= cutoff else n
        out[s.onset : stop] = np.maximum(out[s.onset : stop], 1.0 + (factor - 1.0) * s.weight)
    return out


def subtraction_correction(
    *,
    data: Any,
    stream: np.ndarray,
    spans: list[Span],
    cutoffs: np.ndarray,
    rows: np.ndarray,
    mask: bool = False,
    soft_factor: float | None = None,
) -> np.ndarray:
    """The total filter correction at `rows`, each from the stream cleaned as known at its cutoff.

    Rows share a filter run when no span changes status between their cutoffs, so the filter runs
    once per distinct status of the spans among the rows. The result is NaN elsewhere. With
    `soft_factor` the readings stay in the stream and the filter inflates their observation noise
    inside each known span instead (see `noise_scale_of`).
    """
    out = np.full(len(data.run_of_row), np.nan)
    breaks = np.sort(np.array([v for s in spans for v in (s.start, s.known)] or [0]))
    state = np.searchsorted(breaks, cutoffs[rows], side="right")
    for s in np.unique(state):
        chosen = rows[state == s]
        cutoff = int(breaks[s - 1]) if s > 0 else None
        soft = soft_factor is not None
        correction = fed.correction_for(
            data=data,
            stream=stream
            if soft
            else cleaned_stream(stream=stream, spans=spans, cutoff=cutoff, mask=mask),
            prior=np.zeros(len(stream)),
            multiplier=1.0,
            noise_scale=noise_scale_of(
                spans=spans, cutoff=cutoff, factor=soft_factor, n=len(stream)
            )
            if soft_factor is not None
            else None,
        )
        out[chosen] = correction[chosen]
    return out


def score_nra(
    *, data: Any, site: str, corrections: dict[str, np.ndarray], rows: np.ndarray
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
