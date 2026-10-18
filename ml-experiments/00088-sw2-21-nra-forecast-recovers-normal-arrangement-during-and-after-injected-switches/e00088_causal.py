"""The XL detector run causally: only residual values before a cut stamp are visible.

Experiment 00082 built `xl_intervals.parquet` from the whole record. This module repeats that
construction on residuals whose values at or after the cut are missing and whose values more than
`HISTORY_STAMPS` before it are missing, with four causal rules added to the functions of
experiment 00082:

- the step statistic is evaluated only where its 48-half-hour after-window is complete, so a stamp
  after the cut minus 48 half-hours has no statistic;
- an interval counts as ended only when its end lies at least 48 half-hours before the cut, so the
  evidence for the end (a failed level block or a return step) was fully observed;
- every other interval is open and runs to the cut;
- the first-return search reads only the statistic, which is missing after the cut.

Nothing here reads a switching log.
"""

from dataclasses import dataclass
from typing import Final

import e00088_grid as grid
import numpy as np
import polars as pl
from e00088_combined import candidates
from e00088_detectors import _balance
from e00088_endrule import first_return_intervals
from e00088_intervals import build_intervals
from e00088_run_constants import CAP_DAYS, WINDOW
from e00088_variants import tracked_relative, variant_inputs

HISTORY_STAMPS: Final[int] = 45 * 48
"""Residual values older than this before the cut are hidden; 45 days cover a 30-day transfer."""
Arrays = dict[int, np.ndarray]


@dataclass(frozen=True)
class CausalInterval:
    """One interval the causal detector reports at a cut.

    Attributes:
        series: The series id.
        onset: First stamp of the interval.
        end: First stamp after the interval; the cut for an open interval.
        step_mw: The series' signed 24-hour step at the onset, in MW.
        role: `alarm` (the series the alarm fired on) or `partner`.
        is_open: True when the interval was assumed to continue.
    """

    series: int
    onset: int
    end: int
    step_mw: float
    role: str
    is_open: bool


def visible(*, values: Arrays, cut: int) -> Arrays:
    """Return each series' values with every stamp outside `[cut - HISTORY_STAMPS, cut)` missing."""
    low = max(cut - HISTORY_STAMPS, 0)
    out: Arrays = {}
    for series, array in values.items():
        hidden = np.full(grid.GRID_LENGTH, np.nan)
        hidden[low:cut] = array[low:cut]
        out[series] = hidden
    return out


def causal_intervals(
    *, state: dict, residual: Arrays, offset: Arrays, cut: int
) -> list[CausalInterval]:
    """Return the intervals that XL reports at a cut, from residuals up to the cut only.

    Args:
        state: `e00088_harness.make_state()`.
        residual: The (injected) residual of every series on the stored model, MW.
        offset: The same for the model refitted on the 16th.
        cut: The first stamp that is not visible (exclusive).

    Returns:
        One entry per interval, open intervals running to `cut`.
    """
    res, off = visible(values=residual, cut=cut), visible(values=offset, cut=cut)
    inputs = variant_inputs(variant="X", residual=res, offset=off, scale=state["scales"]["X"])
    for shift in inputs.shift.values():
        shift[max(cut - WINDOW + 1, 0) :] = np.nan
    found = candidates(inputs=inputs)
    if found.is_empty():
        return []
    kept = found.filter(pl.Series(state["model"].score(table=found) > state["threshold"]))
    if kept.is_empty():
        return []
    kept = kept.select("series", "stamp", "step_mw")
    partner_steps: dict[tuple[int, int], dict[int, float]] = {}
    partner_text = []
    for series, stamp in kept.select("series", "stamp").iter_rows():
        _, partners = _balance(inputs=inputs, series=series, stamp=stamp)
        partner_steps[series, stamp] = partners
        partner_text.append(";".join(f"{p}:{step:.6f}" for p, step in partners.items()))
    alarm_table = kept.with_columns(partners=pl.Series(partner_text, dtype=pl.String))
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
    own_step = {(r["series"], r["stamp"]): r["step_mw"] for r in kept.iter_rows(named=True)}
    out: list[CausalInterval] = []
    for iv in intervals:
        key = (iv.alarm_series, iv.onset)
        step = own_step[key] if iv.role == "alarm" else partner_steps[key][iv.series]
        is_open = iv.end + WINDOW > cut
        out.append(
            CausalInterval(
                series=iv.series,
                onset=iv.onset,
                end=cut if is_open else iv.end,
                step_mw=float(step),
                role=iv.role,
                is_open=is_open,
            )
        )
    return out
