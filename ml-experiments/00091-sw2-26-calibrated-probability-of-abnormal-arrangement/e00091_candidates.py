"""The RO-NC detector run causally, keeping every candidate above a low score floor.

`e00091_causal.causal_intervals` keeps only candidates above the RO-NC threshold. A probability
needs the weaker candidates too, each with its score, so this module repeats that construction
with the threshold lowered by `FLOOR_BELOW_THRESHOLD` log-odds and returns the score of every
interval's alarm. Only residual values before the cut are visible. Nothing here reads a
switching log.
"""

from dataclasses import dataclass
from typing import Final

import e00091_causal as causal
import numpy as np
import polars as pl
from e00091_combined import candidates
from e00091_detectors import _balance
from e00091_endrule import first_return_intervals
from e00091_intervals import build_intervals
from e00091_run_constants import CAP_DAYS, WINDOW
from e00091_variants import tracked_relative, variant_inputs

FLOOR_BELOW_THRESHOLD: Final[float] = 4.0
"""Candidates scoring at least this many log-odds below the RO-NC threshold are dropped."""


@dataclass(frozen=True)
class Candidate:
    """One interval of a candidate at a cut.

    Attributes:
        series: The series the interval masks.
        onset: First stamp of the interval.
        end: First stamp after the interval; the cut for an open interval.
        score: The RO-NC log-odds score of the alarm that started the interval.
        above_threshold: Whether the score reaches the RO-NC threshold (the binary flag).
        x1: The alarm series' 24-hour step in noise units.
        is_open: True when the interval was assumed to continue.
        return_z: Size in noise units of the step statistic at a closed interval's end, else 0.
    """

    series: int
    onset: int
    end: int
    score: float
    above_threshold: bool
    x1: float
    is_open: bool
    return_z: float


def causal_candidates(
    *, state: dict, residual: causal.Arrays, offset: causal.Arrays, cut: int
) -> list[Candidate]:
    """Return every interval above the score floor that RO-NC reports at a cut.

    Args:
        state: `e00091_harness.make_state()`.
        residual: The (injected) residual of every series on the stored model, MW.
        offset: The same for the model refitted on the 16th.
        cut: The first stamp that is not visible (exclusive).

    Returns:
        One entry per interval (one per series and onset), open intervals running to `cut`.
    """
    res, off = causal.visible(values=residual, cut=cut), causal.visible(values=offset, cut=cut)
    inputs = variant_inputs(variant="X", residual=res, offset=off, scale=state["scales"]["X"])
    for shift in inputs.shift.values():
        shift[max(cut - WINDOW + 1, 0) :] = np.nan
    found = candidates(inputs=inputs)
    if found.is_empty():
        return []
    threshold = state["threshold"]
    scores = state["model"].score(table=found)
    found = found.with_columns(score=pl.Series(scores))
    kept = found.filter(pl.col("score") > threshold - FLOOR_BELOW_THRESHOLD)
    if kept.is_empty():
        return []
    partner_text = []
    for series, stamp in kept.select("series", "stamp").iter_rows():
        _, partners = _balance(inputs=inputs, series=series, stamp=stamp)
        partner_text.append(";".join(f"{p}:{step:.6f}" for p, step in partners.items()))
    alarm_table = kept.select("series", "stamp", "step_mw").with_columns(
        partners=pl.Series(partner_text, dtype=pl.String)
    )
    relative = tracked_relative(
        variant="X",
        panel=state["panel"],
        offset_panel=state["offset_panel"],
        residual=res,
        offset=off,
    )
    tracked = build_intervals(
        alarm_table=alarm_table, relative=relative, window=WINDOW, detector="D0"
    )
    intervals = first_return_intervals(intervals=tracked, inputs=inputs, cap_days=CAP_DAYS)
    by_alarm = {
        (r["series"], r["stamp"]): (r["score"], r["x1"]) for r in kept.iter_rows(named=True)
    }
    out: list[Candidate] = []
    seen: set[tuple[int, int]] = set()
    for iv in sorted(intervals, key=lambda i: (i.series, i.onset, i.role != "alarm")):
        if (iv.series, iv.onset) in seen:
            continue
        seen.add((iv.series, iv.onset))
        score, x1 = by_alarm[iv.alarm_series, iv.onset]
        is_open = iv.end + WINDOW > cut
        return_z = 0.0
        if not is_open:
            z = inputs.shift[iv.series][iv.end] / inputs.scale[iv.series]
            return_z = float(abs(z)) if np.isfinite(z) else 0.0
        out.append(
            Candidate(
                series=iv.series,
                onset=iv.onset,
                end=cut if is_open else iv.end,
                score=float(score),
                above_threshold=bool(score > threshold),
                x1=float(x1),
                is_open=is_open,
                return_z=return_z,
            )
        )
    return out


ROW_SCHEMA: Final[dict[str, type[pl.DataType]]] = {
    "series": pl.Int64,
    "lag_stamp": pl.Int64,
    "flag": pl.Boolean,
    "has_latest": pl.Boolean,
    "l_score": pl.Float64,
    "l_age_days": pl.Float64,
    "l_since_days": pl.Float64,
    "l_returned": pl.Boolean,
    "l_return_z": pl.Float64,
    "l_x1": pl.Float64,
    "has_best": pl.Boolean,
    "b_score": pl.Float64,
    "b_age_days": pl.Float64,
    "b_since_days": pl.Float64,
    "b_returned": pl.Boolean,
    "b_return_z": pl.Float64,
    "b_x1": pl.Float64,
}


def lag_rows(
    *, found: list[Candidate], series_ids: tuple[int, ...], lags: np.ndarray, cut: int
) -> pl.DataFrame:
    """Return one row per (series, lag stamp) with the arms' inputs, from the candidates at a cut.

    A candidate covers a lag stamp `t` of its series when `onset <= t < end` (an open interval
    runs to the cut). The binary flag is true when a covering candidate is above the RO-NC
    threshold. The `l_` columns describe the covering candidate with the latest onset, and the
    `b_` columns the covering candidate with the highest score (a post hoc variant). A lag stamp
    after a closed interval's end is not covered by it.

    Args:
        found: `causal_candidates` at the cut.
        series_ids: The series to give rows to.
        lags: The lag stamps, all before `cut`.
        cut: The first stamp that is not visible.

    Returns:
        A frame with the columns of `ROW_SCHEMA`.
    """
    nan = float("nan")
    rows: dict[str, list] = {name: [] for name in ROW_SCHEMA}
    for series in series_ids:
        mine = [c for c in found if c.series == series]
        for t in lags:
            covering = [c for c in mine if c.onset <= t < c.end]
            latest = max(covering, key=lambda c: c.onset, default=None)
            best = max(covering, key=lambda c: c.score, default=None)
            rows["series"].append(series)
            rows["lag_stamp"].append(int(t))
            rows["flag"].append(any(c.above_threshold for c in covering))
            for prefix, c in (("l", latest), ("b", best)):
                rows[f"has_{'latest' if prefix == 'l' else 'best'}"].append(c is not None)
                rows[f"{prefix}_score"].append(c.score if c else nan)
                rows[f"{prefix}_age_days"].append((cut - c.onset) / 48 if c else nan)
                rows[f"{prefix}_since_days"].append((t - c.onset) / 48 if c else nan)
                rows[f"{prefix}_returned"].append(bool(c and not c.is_open))
                rows[f"{prefix}_return_z"].append(c.return_z if c else nan)
                rows[f"{prefix}_x1"].append(c.x1 if c else nan)
    return pl.DataFrame(rows, schema=ROW_SCHEMA)
