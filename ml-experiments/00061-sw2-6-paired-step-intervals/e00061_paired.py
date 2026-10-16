"""The paired-step detector: an onset step and its opposite return step found as one interval.

Nothing here reads a switching log except `labelled_duration_prior`, which reads the durations of
the labelled real transfers (experiment 41). The step statistic is D0's 24-hour level shift, in
noise units.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cache
from typing import Final

import numpy as np
import polars as pl
from e00061_common import GRID_LENGTH, LABELLED_PATH
from e00061_detectors import (
    REFIT_PERIOD_WINDOW,
    DetectorInputs,
    local_peaks,
    refit_stamps,
)
from e00061_intervals import BLOCK, CAP_STAMPS, Interval, level_track, relative_residual

CANDIDATE_Z: Final[float] = 2.5
"""Local peaks of the onset step at or above this many noise units are candidate onsets."""
RETURN_Z_MIN: Final[float] = 1.5
"""A return step must reach this many noise units in the opposite direction to count."""
MIN_DURATION: Final[int] = BLOCK
"""Shortest duration searched, in half-hours (24 hours)."""
PRIOR_WEIGHT: Final[float] = 1.0
PRIOR_FLOOR: Final[float] = -1.0
"""The log of the prior's density ratio to its peak is floored here."""
SQRT2: Final[float] = float(np.sqrt(2.0))
GRID: Final[np.ndarray] = np.arange(MIN_DURATION, CAP_STAMPS + 1)
"""The durations searched, in half-hours."""


def flat_prior() -> np.ndarray:
    """The flat prior: zero log density ratio at every duration searched."""
    return np.zeros(len(GRID))


def labelled_duration_prior(*, exclude_cards: frozenset[str] = frozenset()) -> np.ndarray:
    """Log density ratio, floored, of the labelled real transfers' durations over `GRID`.

    A Gaussian kernel density of the log duration in hours (Silverman bandwidth) over the labelled
    real operations of 2 hours to 30 days, without the cards in `exclude_cards`.
    """
    frame = pl.read_parquet(LABELLED_PATH).filter(pl.col("kind") == "real_operation")
    frame = frame.filter(~pl.col("card_id").is_in(list(exclude_cards)))
    hours = (
        (frame["logged_open_utc"] - frame["logged_close_utc"]).dt.total_minutes() / 60
    ).to_numpy()
    log_hours = np.log(hours[(hours >= 2) & (hours <= 30 * 24)])
    bandwidth = 1.06 * log_hours.std() * len(log_hours) ** -0.2
    at = np.log(GRID / 2.0)
    density = np.exp(-0.5 * ((at[:, None] - log_hours[None, :]) / bandwidth) ** 2).sum(axis=1)
    return np.maximum(np.log(density / density.max()), PRIOR_FLOOR)


@cache
def near_refit() -> np.ndarray:
    """True within 24 hours of a monthly refit, where the residual model's jump can mimic a step."""
    mask = np.zeros(GRID_LENGTH, dtype=bool)
    for stamp in refit_stamps():
        mask[max(stamp - BLOCK + 1, 0) : stamp + BLOCK] = True
    return mask


@dataclass(frozen=True)
class Candidate:
    """The best reading of one candidate onset: with a return step, or without one."""

    series: int
    onset: int
    onset_z: float
    end: int | None
    return_z: float
    score: float


def best_candidates(
    *, z: np.ndarray, series: int, window: int, prior: np.ndarray
) -> list[Candidate]:
    """Score every candidate onset of one series and keep its best total.

    The pair score of a return stamp `e` is `(|z(s)| + return_z) / sqrt(2)` plus the prior at the
    duration `e - s`, where `return_z` is `z(e)` in the opposite direction to `z(s)`. Without a
    return step the score is `|z(s)| / sqrt(2)` plus the prior floor.
    """
    refits = refit_stamps() if window >= REFIT_PERIOD_WINDOW else np.array([], dtype=int)
    out: list[Candidate] = []
    for s in local_peaks(z=z, threshold=CANDIDATE_Z, window=window):
        if len(refits) and (np.abs(refits - s) < window).any():
            continue
        sign = np.sign(z[s])
        alone = abs(z[s]) / SQRT2 + PRIOR_WEIGHT * prior.min()
        best = Candidate(series, int(s), float(z[s]), None, 0.0, float(alone))
        stop = min(s + CAP_STAMPS + 1, GRID_LENGTH)
        ret = -sign * z[s + MIN_DURATION : stop]
        if ret.size:
            usable = (
                np.isfinite(ret) & (ret >= RETURN_Z_MIN) & ~near_refit()[s + MIN_DURATION : stop]
            )
            total = np.where(
                usable,
                (abs(z[s]) + np.nan_to_num(ret)) / SQRT2 + PRIOR_WEIGHT * prior[: ret.size],
                -np.inf,
            )
            k = int(np.argmax(total))
            if total[k] > alone:
                best = Candidate(
                    series,
                    int(s),
                    float(z[s]),
                    int(s + MIN_DURATION + k),
                    float(ret[k]),
                    float(total[k]),
                )
        out.append(best)
    return out


def paired_alarms(*, inputs: DetectorInputs, prior: np.ndarray) -> dict[int, list[Candidate]]:
    """Every series' scored candidates, ready to be cut at any threshold."""
    return {
        s: best_candidates(
            z=inputs.shift[s] / inputs.scale[s], series=s, window=inputs.window, prior=prior
        )
        for s in inputs.shift
    }


def cut(*, scored: dict[int, list[Candidate]], threshold: float) -> list[Candidate]:
    """The candidates at or above `threshold`, in time order per series.

    A candidate of the opposite sign within 24 hours of an accepted pair's end is the pair's own
    return step and is dropped.
    """
    kept: list[Candidate] = []
    for candidates in scored.values():
        accepted: list[Candidate] = []
        for c in sorted(candidates, key=lambda c: c.onset):
            if c.score < threshold:
                continue
            if any(
                a.end is not None
                and np.sign(a.onset_z) != np.sign(c.onset_z)
                and abs(c.onset - a.end) <= BLOCK
                for a in accepted
            ):
                continue
            accepted.append(c)
        kept.extend(accepted)
    return sorted(kept, key=lambda c: (c.series, c.onset))


def alarm_table(*, kept: list[Candidate], shift: dict[int, np.ndarray]) -> pl.DataFrame:
    """One row per alarm: series, stamp, step_mw, z, score, end, and return_z."""
    rows = [
        {
            "series": c.series,
            "stamp": c.onset,
            "step_mw": float(shift[c.series][c.onset]),
            "z": c.onset_z,
            "score": c.score,
            "end": c.end,
            "return_z": c.return_z,
        }
        for c in kept
    ]
    schema = {
        "series": pl.Int64,
        "stamp": pl.Int64,
        "step_mw": pl.Float64,
        "z": pl.Float64,
        "score": pl.Float64,
        "end": pl.Int64,
        "return_z": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


def paired_intervals(
    *,
    kept: list[Candidate],
    residual: dict[int, np.ndarray],
    expected: dict[int, np.ndarray],
    window: int,
) -> list[Interval]:
    """One interval per alarm: the paired return stamp as the end, else D0M's tracked end."""
    relative = {s: relative_residual(residual=residual[s], expected=expected[s]) for s in residual}
    out: list[Interval] = []
    for c in kept:
        if c.end is not None:
            out.append(
                Interval(c.series, c.onset, c.end, c.end + BLOCK, c.series, "P", "alarm", c.onset_z)
            )
            continue
        rho, end, known = level_track(q=relative[c.series], onset=c.onset, window=window)
        if np.isfinite(rho):
            out.append(Interval(c.series, c.onset, end, known, c.series, "P", "alarm", rho))
    return out


END_SEARCH_STAMPS: Final[int] = 14 * BLOCK
"""PE searches for the return step up to 14 days after the onset."""


def end_only_intervals(*, intervals: list[Interval], inputs: DetectorInputs) -> list[Interval]:
    """Replace each interval's end by the strongest return step in the next 14 days (PE).

    The alarms stay as they are. When no stamp reaches `RETURN_Z_MIN` in the opposite direction, the
    interval keeps its tracked end.
    """
    out: list[Interval] = []
    for iv in intervals:
        z = inputs.shift[iv.series] / inputs.scale[iv.series]
        sign = np.sign(z[iv.onset])
        stop = min(iv.onset + END_SEARCH_STAMPS + 1, GRID_LENGTH)
        ret = -sign * z[iv.onset + MIN_DURATION : stop]
        blocked = near_refit()[iv.onset + MIN_DURATION : stop]
        usable = np.where(np.isfinite(ret) & (ret >= RETURN_Z_MIN) & ~blocked, ret, -np.inf)
        if usable.size == 0 or not np.isfinite(usable.max()):
            out.append(iv)
            continue
        end = int(iv.onset + MIN_DURATION + np.argmax(usable))
        out.append(replace(iv, end=end, known_from=end + BLOCK))
    return out
